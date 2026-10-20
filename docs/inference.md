# Inference guide

Run these commands from the repository root after completing the
[installation](../README.md#installation) and [asset setup](assets.md).
The default paths are `../egoaim-demo`, `../gaze2hoi-checkpoints`, and the
locally prepared MANO directory supplied with `--mano-dir`.

## Validate assets and select examples

```bash
gaze2hoi list
gaze2hoi check --samples all --mano-dir ../mano --hashes
```

`list` shows all 11 examples and their recorded project-page draws. `check`
validates packaged inputs, required model files, available checksums, and
MANO file presence when `--mano-dir` is supplied.

| Selector | Examples |
|---|---|
| `homepage` | Default: four qualitative examples and three failure cases |
| `qualitative` | Four qualitative examples |
| `failure` | Three failure cases |
| `overview` | Four overview examples |
| `all` | All 11 examples |

Use `--samples` with one or more groups or sample IDs:

```bash
gaze2hoi forecast --mano-dir ../mano \
  --samples P0009_8c0caf6c_000215_000274 \
  --output outputs/single-example
```

Keep the selection consistent between inference and rendering. For all examples:

```bash
gaze2hoi forecast --samples all --mano-dir ../mano --output outputs/reproduction
gaze2hoi render --samples all --mano-dir ../mano \
  --results outputs/reproduction --output outputs/gallery
python -m http.server 8000 --directory outputs/gallery
```

Open [the local gallery](http://localhost:8000/) to compare observations,
ground truth, and generated forecasts. Rendering requires FFmpeg and EGL/OpenGL.
Use `--data` for a custom example directory. Commands that use model assets
(`check`, `intention`, and `forecast`) also accept `--checkpoints`; the download
command accepts both paths as destinations.

## Optional: run intention inference separately

`forecast` generates intentions by default. A separate intention run is useful
for inspecting the structured predictions or reusing them across motion runs:

```bash
gaze2hoi intention --output outputs/intentions
gaze2hoi forecast --mano-dir ../mano \
  --intention-file outputs/intentions/predictions.jsonl \
  --output outputs/from-intentions
```

The intention command writes one `intention.json` per example and a combined
`predictions.jsonl`. Decoding is grammar constrained with `do_sample=False`.
The supplied file must contain valid intentions for every selected example;
use matching `--samples` selections when departing from the default seven.
Run metadata records whether intentions were generated or explicitly supplied.

### Local base model and GPU placement

Both inference commands accept `--base-model` and `--local-files-only`:

```bash
gaze2hoi intention \
  --base-model /path/to/Qwen3-VL-8B-Instruct --local-files-only \
  --max-memory 0=10GiB --max-memory 1=10GiB \
  --output outputs/local-intentions
```

The default base model is `Qwen/Qwen3-VL-8B-Instruct`, with automatic device
placement. Repeated `--max-memory` options limit model placement; they do not
cap peak runtime memory. BF16 base weights alone occupy approximately 16.3 GiB,
so allow additional headroom for visual tokens, attention, and generation.
The intention model is released before ACG and motion inference begin.

## Sampling and draw selection

Motion inference uses **50 DDIM steps**, **guidance scale 1.5**, **eta 0**, and
default **base seed 999**. Without a draw option, each example uses its recorded
project-page draw. CLI draw numbers start at one; stored `draw_indices` start
at zero. `--draw` and `--num-draws` are mutually exclusive.

```bash
gaze2hoi forecast --mano-dir ../mano --samples qualitative \
  --num-draws 5 --seed 999 --output outputs/five-draws
gaze2hoi render --mano-dir ../mano --samples qualitative \
  --results outputs/five-draws --draw 3 --output outputs/draw-three
```

`--num-draws 5` generates draws 1 through 5. Use `forecast --draw 3` to generate
only draw 3 across selected examples. The renderer defaults to each example's
recorded project-page draw; pass `render --draw` when selecting another saved
draw. Changing `--seed` explores different forecasts even when the intention
is unchanged. Eta zero still leaves a dependence on initial noise.

Qualitative and failure examples use sample/draw-specific CPU noise. Overview
examples preserve their historical sequential device noise stream; use CUDA
to reproduce their references. Motion inference also supports
`--motion-device cpu`, which does not reproduce the CUDA overview noise stream.

## Outputs and reproducibility

```text
outputs/homepage/
├── predictions.jsonl
├── run.json
└── <sample_id>/
    ├── intention.json
    ├── forecast.npz
    └── metadata.json
```

Inference refuses a nonempty output directory unless `--overwrite` is supplied.
Use a new directory to preserve a previous run. `forecast.npz` stores metric
states, hand and bound-object trajectories, validity masks, slots, draw indices,
seeds, and predicted contact fields. Metadata records bindings, sampling settings,
versions, and intention provenance. See [data and output format](data-format.md)
for array shapes and coordinates.

Fresh forecasts use observed inputs and online contact grounding. Archived
predictions and future ground truth are used only for comparison, and rendering
applies no temporal smoothing or motion refinement. Fixed seeds support
reproducibility in a matched environment; hardware, kernels, precision, and
software versions can change numerical results. The
[verification notes](reproducibility.md) report the measured agreement for all
11 released examples and its scope.

## Development checks

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check src tests tools
ruff format --check src tests tools
```

The default regression tests require no model downloads, example data, or GPU.
