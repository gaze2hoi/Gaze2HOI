# Models, examples, and local assets

The code repository is independent of the model and example packages. It does not contain large weights, observation media, or proprietary MANO model files.

## Checkpoint package

```text
gaze2hoi-checkpoints/
├── README.md
├── LICENSE.md
├── NOTICE.md
├── manifest.json
├── licenses/
├── intention/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── acg.pt
├── acg_config.json
├── motion.pt
├── motion_config.json
└── assets/
    ├── norm_stats.npz
    ├── object_templates.npz
    ├── contact_priors.npz
    ├── bottom_prior.npy
    └── *_vocab.json
```

The model state dictionaries contain only inference tensors. Optimizer states, duplicated EMA copies, and training logs are omitted. Motion weights merge EMA parameters with required fixed model buffers. Configuration files retain only architecture parameters used by the runtime. The adapter includes the trained gaze modulation module in addition to LoRA parameters.

The Qwen base model and its tokenizer/processor are loaded from `--base-model`. The adapter package does not duplicate them. Use `--local-files-only` when the base model is already available locally.

`manifest.json` records SHA-256 hashes and sizes for the checkpoint package. The example package has its own `checksums.sha256`. The `check --hashes` command verifies both formats.

## Downloading separately hosted packages

The [model package](https://huggingface.co/gaze2hoi/Gaze2HOI) and [example package](https://huggingface.co/datasets/gaze2hoi/EgoAIM-Demo) are hosted in separate Hugging Face repositories. From the code repository root, run:

```bash
gaze2hoi download \
  --model-repo gaze2hoi/Gaze2HOI \
  --data-repo gaze2hoi/EgoAIM-Demo
```

The default destinations are `../gaze2hoi-checkpoints` and `../egoaim-demo`. Override them with `--checkpoints` and `--data`, and pass the same paths to subsequent inference commands. Either repository option can be omitted to download only the other package.

`--model-revision` and `--data-revision` default to `main`. For reproducible downloads, pass the corresponding repository's commit hash to each option. After downloading, run `gaze2hoi check --samples all --hashes` to verify both packages. Add `--mano-dir ../mano` once MANO is prepared to validate those local files as well.

The initial release uses these fixed revisions:

| Package | Revision |
|---|---|
| Model checkpoints | `a2d240d77d2fa3597b373b5cc3f04d5e3f42e430` |
| Example data | `c3a78cc93bfbb37fcb69c6c9533b5c165ef993be` |

The `download` extra supplies the Hub client: `python -m pip install -e ".[download]"`. The main installation command already includes it. These downloads do not include the upstream Qwen base model or MANO files.

## MANO

Obtain MANO through its official distribution and prepare local NumPy-only model files with `tools/prepare_mano.py`. Both `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` must be present in `--mano-dir`. The conversion tool rejects identical source/output directories and does not overwrite existing model files.

MANO files are user-local assets and must not be added to this repository, the checkpoint package, or the example package.

## Scope and attribution

The example package documents its source data, transformations, retained identifiers, and component-specific terms. It includes all observations needed by the demonstrated inference workflow, as well as separate references for inspection. It is a compact demonstration package rather than the complete EgoAIM benchmark.

Keep each package's README, license files, and notices with its redistributed files. These attribution files are part of the release, not implementation comments.
