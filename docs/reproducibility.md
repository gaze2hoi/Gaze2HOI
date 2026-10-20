# Reproducibility

The release preserves the models, observed inputs, binding rules, contact computation, sampling configuration, and rendering settings used for the project-page examples. Reproduction means running these models on the supplied observations; archived predictions are never substituted for fresh inference.

## Default settings

| Component | Configuration |
|---|---|
| Intention model | Qwen3-VL-8B-Instruct with the released LoRA and TGSM adapter |
| Intention decoding | Grammar constrained, `do_sample=False` |
| Contact grounding | Online ACG using the predicted intention and observed scene |
| Motion model | Released EMA weights, checkpoint step 22500 |
| Sampling | DDIM, 50 steps, guidance 1.5, eta 0 |
| Base seed | 999 |
| Draw | Per-example homepage draw recorded in `sample.json` |
| Motion postprocessing | None |

Draw numbers on the CLI and in documentation start at one. Array `draw_indices` start at zero. Passing `--num-draws 5` produces indices 0 through 4. The renderer selects the example's recorded homepage draw unless `--draw` is specified.

Two historical noise conventions are preserved:

- `sample_draw_cpu`, used for qualitative and failure examples, derives each draw seed from the sample ID, base seed, and draw index and generates its noise on CPU. Draw ordering does not change the result.
- `sequential_device`, used for overview examples, generates a sequence from a sample-specific stream on the inference device. Earlier draws are consumed even when only a later draw is requested. Use CUDA to reproduce the overview references.

DDIM eta zero does not remove dependence on the initial noise. A new seed can change a forecast even when the inferred intention is unchanged.

## Verified numerical agreement

The release was checked against all 11 supplied project-page examples on NVIDIA RTX A4500 hardware with Python 3.10.20, PyTorch 2.10.0 / CUDA 12.8, Transformers 4.57.6, PEFT 0.19.1, XGrammar 0.1.29, NumPy 1.26.4, and SMPL-X 0.1.28.

- Fresh VLM inference matched all 11 archived structured intentions exactly.
- Motion inference conditioned on those fresh intentions reproduced all 11 selected draws, with maximum translation difference **1.20 × 10⁻⁷ m** and maximum raw state-channel difference **7.16 × 10⁻⁷**. The latter mixes pose coefficients and rotation channels with translation and is not a distance metric.
- Repeated motion inference with the same seed was bitwise identical in the verified environment. Changing the seed changed the forecast; requesting the same draws in another order preserved their values.
- Motion inference succeeded from a standalone copy of observed inputs with the reference and ground-truth directories absent and without importing the original research project.
- Two native-resolution renderer frames, from a qualitative example and a failure example, matched the original renderer pixel for pixel. Complete prediction and GT clips were rendered for all 11 examples.

The measured numerical results are recorded in [verification.json](verification.json). They are checks of these released examples, not claims of bitwise equality on arbitrary hardware or environments, or of full-benchmark reproduction.

Different PyTorch/CUDA versions, attention kernels, precision, GPU models, or device-based random streams can change results. Intention inference is greedy, but small numerical changes near token-decision boundaries can still change its output.

## Installation and resource checks

A separate Python 3.10 environment was built from a source-only copy, without the research environment on its import path. Installed-package CLI checks, model and example integrity checks, Qwen processor loading, actual CPU contact/motion inference, and EGL video rendering were exercised. Qwen's processor requires torchvision; it is included in the intention extra.

The full intention model was exercised using automatic placement across four GPUs with a 6 GiB placement limit per GPU. These are placement limits, not maximum peak memory measurements. Allow additional headroom for image processing and generation. The BF16 base weights alone occupy approximately 16.3 GiB.

ACG and motion inference peaked at approximately **462 MiB of allocated CUDA tensor memory** in a separate single-example probe. CUDA context memory, reserved memory, and renderer memory are additional. CPU motion inference is supported with `--motion-device cpu`; it does not reproduce the CUDA-specific overview noise stream.

The supported default renderer uses EGL. FFmpeg must be installed separately. The renderer was also verified using PyOpenGL 3.1.0, the dependency required by pyrender 0.1.45; no incompatible PyOpenGL override is needed.

## Checking a local run

1. Run `gaze2hoi check --samples all --mano-dir ../mano --hashes` to verify the packaged files.
2. Run `gaze2hoi forecast --samples all --mano-dir ../mano --output outputs/reproduction`.
3. Inspect each sample's `metadata.json` for its seed, draw, bindings, and intention provenance.
4. Render those outputs using `gaze2hoi render --samples all --mano-dir ../mano --results outputs/reproduction --output outputs/gallery`.

Ground-truth files are used only for comparison and visualization. A supplied `--intention-file` is recorded as such; omit this option to run the VLM inside the full-pipeline command.
