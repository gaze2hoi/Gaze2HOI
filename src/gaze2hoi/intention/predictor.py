"""Deterministic, grammar-constrained intention inference."""

from __future__ import annotations

import gc
import json
import threading
from pathlib import Path
from typing import Any

from .grammar import build_json_schema
from .inputs import IntentionInput, read_json
from .serialization import parse_prediction


class IntentionPredictor:
    """Load Qwen3-VL and the Gaze2HOI adapter on first use.

    ``device="auto"`` enables Accelerate placement. ``max_memory`` may reserve
    memory for other processes, for example ``{0: "10GiB", 1: "10GiB"}``.
    A predictor serializes requests because TGSM features are sample-specific.
    """

    def __init__(
        self,
        base_model: str | Path,
        adapter: str | Path,
        device: str = "auto",
        *,
        schema: dict[str, Any] | str | Path | None = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 320,
        device_map: str | dict[str, Any] | None = None,
        max_memory: dict[int | str, int | str] | None = None,
        local_files_only: bool = False,
        dataset_root: str | Path | None = None,
    ) -> None:
        if not str(base_model).strip():
            raise ValueError("base_model must be a local model directory or Hugging Face model ID")
        self.base_model = str(base_model)
        self.adapter_path = Path(adapter).resolve()
        config = read_json(self.adapter_path / "adapter_config.json")
        if "gaze_feature_film" not in (config.get("modules_to_save") or []):
            raise ValueError("The adapter must include the learned gaze_feature_film module")
        if not (self.adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing adapter_model.safetensors in {self.adapter_path}")
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.max_memory = max_memory
        self.local_files_only = local_files_only
        self.schema = schema
        self.dataset_root = dataset_root
        self._lock = threading.RLock()
        self._model = self._processor = self._compiler = self._gaze_adapter = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import xgrammar as xgr
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Install the intention inference dependencies with pip install -e '.[intention]'"
            ) from exc
        from .tgsm import GazeFeatureAdapter

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for the requested device {self.device}")
        placement = self.device_map
        if placement is None:
            placement = "auto" if self.device == "auto" else {"": self.device}
        kwargs: dict[str, Any] = {
            "dtype": getattr(torch, self.dtype),
            "device_map": placement,
            "local_files_only": self.local_files_only,
        }
        if self.max_memory is not None:
            kwargs["max_memory"] = self.max_memory
        processor = AutoProcessor.from_pretrained(
            self.base_model, local_files_only=self.local_files_only
        )
        model = AutoModelForImageTextToText.from_pretrained(self.base_model, **kwargs)
        gaze_adapter = GazeFeatureAdapter()
        gaze_adapter.attach(model)
        model = PeftModel.from_pretrained(model, str(self.adapter_path), local_files_only=True)
        model.eval()
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            processor.tokenizer, vocab_size=model.config.get_text_config().vocab_size
        )
        compiler = xgr.GrammarCompiler(tokenizer_info)
        self._model, self._processor = model, processor
        self._compiler, self._gaze_adapter = compiler, gaze_adapter

    def predict(self, sample_dir: str | Path) -> dict[str, Any]:
        """Generate the two-hand intention record from observation-only sample inputs."""
        sample = IntentionInput.load(sample_dir, schema=self.schema, dataset_root=self.dataset_root)
        heatmaps = sample.load_gaze()
        with self._lock:
            self._load()
            import torch
            import xgrammar as xgr

            from .tgsm import pool_gaze_features

            messages, images = sample.open_chat()
            try:
                chat = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(text=[chat], images=images, return_tensors="pt").to(
                    self._model.device
                )
                features = pool_gaze_features(
                    heatmaps, inputs["image_grid_thw"], self._processor.image_processor.merge_size
                )
                self._gaze_adapter.set_pending(features.to(self._model.device))
                compiled = self._compiler.compile_json_schema(
                    json.dumps(build_json_schema(sample.context))
                )
                with torch.inference_mode():
                    generated = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        logits_processor=[xgr.contrib.hf.LogitsProcessor(compiled)],
                    )
                tokens = generated[0][inputs["input_ids"].shape[1] :]
                raw = self._processor.decode(tokens, skip_special_tokens=True)
                parsed = parse_prediction(raw, sample.context)
                return {
                    "sample_id": sample.sample_id,
                    "raw": raw,
                    "parse_ok": parsed.ok,
                    "parse_errors": parsed.errors,
                    "intent": parsed.intent,
                }
            finally:
                self._gaze_adapter.set_pending(None)
                for image in images:
                    image.close()

    def close(self) -> None:
        """Release model references before running a subsequent pipeline stage."""
        with self._lock:
            if self._gaze_adapter is not None:
                self._gaze_adapter.set_pending(None)
            self._model = self._processor = self._compiler = self._gaze_adapter = None
            gc.collect()
            import sys

            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self) -> IntentionPredictor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
