"""Public inference API for intention-conditioned hand-object forecasting."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .assets import NormStats
from .binding import bind_observation, model_inputs, parse_intention
from .config import ModelConfig
from .contact import OnlineACG
from .model import OursDiT
from .sampling import DDIMSampler, draw_seed, initial_noise


class MotionPredictor:
    """Run online contact grounding and the released motion diffusion model."""

    def __init__(self, checkpoint_dir: Path, mano_dir: Path, device: str = "cuda:0") -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; pass device='cpu' explicitly"
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assets = self.checkpoint_dir / "assets"
        self.stats = NormStats(assets / "norm_stats.npz")
        self.vocab = json.loads((assets / "intent_vocab.json").read_text(encoding="utf-8"))
        self.object_vocab = json.loads((assets / "object_vocab.json").read_text(encoding="utf-8"))
        furniture = json.loads((assets / "furniture_vocab.json").read_text(encoding="utf-8"))
        raw_config = json.loads(
            (self.checkpoint_dir / "motion_config.json").read_text(encoding="utf-8")
        )
        fields = {field.name for field in dataclasses.fields(ModelConfig)}
        config = ModelConfig(**{key: value for key, value in raw_config.items() if key in fields})
        self.model = OursDiT(
            config,
            {key: len(values) for key, values in self.vocab.items()},
            len(furniture["types"]),
            self.stats,
            assets,
        )
        weights = torch.load(
            self.checkpoint_dir / "motion.pt", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.sampler = DDIMSampler(self.stats).to(self.device).eval()
        self.contact = OnlineACG(self.checkpoint_dir, Path(mano_dir), self.device)

    @torch.inference_mode()
    def prepare(self, sample_dir: Path, intention: dict) -> tuple[dict, dict, dict, dict]:
        """Construct conditions from observations without opening reference files."""
        shard, bindings, metadata = bind_observation(
            sample_dir, intention, self.vocab, self.object_vocab
        )
        intention = parse_intention(intention)
        fields = self.contact.predict(shard, intention, sample_id=metadata["sample_id"])
        inputs = model_inputs(shard, self.stats)
        inputs.update(fields)
        batch = {key: value.unsqueeze(0).to(self.device) for key, value in inputs.items()}
        return batch, shard, bindings, metadata

    @torch.inference_mode()
    def predict(
        self,
        sample_dir: Path,
        intention: dict,
        draws: list[int],
        seed: int = 999,
        noise_profile: str | None = None,
    ) -> dict:
        """Generate zero-based draws with metric translations and native pose parameters."""
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not draws or any(type(draw) is not int or draw < 0 for draw in draws):
            raise ValueError("draws must contain non-negative integer indices")
        if len(set(draws)) != len(draws):
            raise ValueError("draw indices must be unique")
        batch, shard, bindings, sample = self.prepare(sample_dir, intention)
        sample_id = str(sample["sample_id"])
        profile = noise_profile or sample.get("homepage", {}).get(
            "noise_profile", "sample_draw_cpu"
        )
        if profile not in {"sample_draw_cpu", "sequential_device"}:
            raise ValueError(f"Unknown noise profile: {profile}")
        noises, seeds = {}, {}
        if profile == "sequential_device":
            payload = f"{seed}:vlm_rebind:{sample_id}".encode()
            stream_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            generator = torch.Generator(device=self.device).manual_seed(stream_seed)
            for index in range(max(draws) + 1):
                noise = torch.randn((1, 60, 66), device=self.device, generator=generator)
                if index in draws:
                    noises[index] = noise
                    seeds[index] = stream_seed
        else:
            for index in draws:
                noises[index] = initial_noise(sample_id, index, seed)
                seeds[index] = draw_seed(seed, sample_id, index)
        outputs = []
        for index in draws:
            state = self.sampler(self.model, batch, noises[index])[0]
            if not torch.isfinite(state).all():
                raise FloatingPointError(
                    f"Non-finite forecast for sample {sample_id}, draw {index}"
                )
            outputs.append(state.cpu().numpy().astype(np.float32))
        states = np.stack(outputs)
        hand_valid = np.stack(
            [np.asarray(shard[f"{side}_hand_valid"], dtype=bool) for side in ("left", "right")],
            axis=-1,
        )
        object_valid = np.stack(
            [np.asarray(shard[f"{side}_obj_valid"], dtype=bool) for side in ("left", "right")],
            axis=-1,
        )
        hand_valid[30:] = hand_valid[:30].any(axis=0)
        object_valid[30:] = object_valid[:30].any(axis=0)
        result = {
            "state": states,
            "mano_betas": np.asarray(shard["mano_betas"], dtype=np.float32),
            "hand_state": states[:, 30:, :48].reshape(len(draws), 30, 2, 24),
            "bound_object_state": states[:, 30:, 48:].reshape(len(draws), 30, 2, 9),
            "object_slots": np.array(
                [int(shard["left_obj_slot"]), int(shard["right_obj_slot"])], dtype=np.int64
            ),
            "hand_valid": hand_valid,
            "object_valid": object_valid,
            "draw_indices": np.asarray(draws, dtype=np.int64),
            "seeds": np.asarray([seeds[index] for index in draws], dtype=np.uint64),
            "metadata": {
                "sample_id": sample_id,
                "intention": parse_intention(intention),
                "bindings": bindings,
                "base_seed": seed,
                "noise_profile": profile,
                "draw_indices": draws,
                "ddim_steps": 50,
                "guidance_scale": 1.5,
                "ddim_eta": 0.0,
                "device": str(self.device),
                "torch_version": torch.__version__,
                "future_validity": "any_observed_presence",
                "deterministic_algorithms": True,
                "postprocessing": "none",
            },
        }
        for key in (
            "acg_contact_cond",
            "acg_contact_uncond",
            "acg_bottom",
            "acg_bottom_valid",
            "acg_landing",
            "acg_landing_valid",
            "acg_hand_valid",
        ):
            result[key] = batch[key][0].cpu().numpy()
        return result
