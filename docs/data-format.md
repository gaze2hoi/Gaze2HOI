# Data and output format

All JSON files use UTF-8. NumPy archives contain numeric or Unicode arrays and are loaded with `allow_pickle=False`. Paths inside the example package are relative; the inference runtime does not need the original research directory.

## Example selection

`manifest.json` lists sample IDs, relative sample directories, groups, phases, and homepage draws. `gaze2hoi list` shows these records. Select a group, an individual ID, or several selectors with `--samples`:

| Selector | Examples |
|---|---:|
| `homepage` (default) | Four qualitative examples and three failure cases |
| `qualitative` | 4 |
| `failure` | 3 |
| `overview` | 4 |
| `all` | 11 |

Each sample's `sample.json` preserves its sequence ID, observation and forecast frame ranges, phase, contact fallback, and sampling configuration. Original frame numbers and sample IDs remain unchanged.

## Intention inputs

`inputs/intention.json` stores system and user messages, ordered image paths, a relative gaze-file path, and the context used for constrained decoding. It contains no assistant answer. Image and schema references may reach shared assets but must stay inside the dataset root.

The first five images are observed RGB frames at offsets 5, 11, 17, 23, and 29 of the 30-frame observation window. They are followed by object reference views in prompt order. RGB inputs are 672 by 672 pixels; object reference images are 448 by 448 pixels.

`inputs/gaze.npz` contains `heatmaps`, shaped `[5, 2, 42, 42]`. The two channels represent current and historical gaze. The release preserves the original fixation and recency weighting and projection, rather than regenerating approximate gaze maps from video.

An intention has `left` and `right` entries. Each entry contains `action`, `object`, `part`, `deep_action`, `target`, and `point`. The input context determines allowed values. A hand with no intention has all six fields set to `null`. Placement points use a 0–1000 image grid; unused fields are `null`. Use the supplied schema and serializer when constructing new records.

`intention.json` outputs additionally record `sample_id`, the generated text, the parsed `intent`, and parsing diagnostics. `predictions.jsonl` contains one such record per line and can be passed explicitly to `forecast --intention-file`.

## Observed 3D inputs

The motion model consumes a 30-frame observation at 30 Hz and predicts the next 30 frames.

| File / array | Shape | Meaning |
|---|---|---|
| `observation.npz: state` | `[30, 66]` | Observed hands; object channels are filled after intention-based binding |
| `left_hand_valid`, `right_hand_valid` | `[30]` | Observed hand availability |
| `mano_betas` | `[10]` | Hand shape coefficients |
| `left_j0`, `right_j0` | `[3]` | MANO wrist-root correction |
| `scene_present` | `[33]` | Observed candidate object presence |
| `scene_transl_last`, `scene_transl_mean` | `[33, 3]` | Observed object position summaries |
| `scene_rot6d_last` | `[33, 6]` | Last observed object orientation |
| `objects.npz: state` | `[N, 30, 9]` | All observed candidate object tracks |
| `objects.npz: valid` | `[N, 30]` | Candidate availability |
| `objects.npz: slots`, `names`, `uids` | `[N]` | Global slot and object identity |

The observation archive also includes observed gaze, head motion, furniture geometry, and the anchor transform. `placement.json` stores camera calibration and support-plane information for lifting an intention's image-space placement point into 3D. Predicted object names determine the two bound objects; they are not taken from future annotations.

### State layout and coordinates

The 66 channels are `[left hand (24), right hand (24), left-bound object (9), right-bound object (9)]`.

- Hand: translation (3), wrist rotation (6), and MANO finger PCA coefficients (15).
- Object: translation (3) and rotation (6).
- Rotation: first rotation-matrix column followed by the second, decoded with Gram–Schmidt orthogonalization.

Translations are in **metres**, in a gravity-aligned anchor frame with +Z up. The origin is the last observed head position; horizontal head-forward is +X. World points transform as `anchor_rotation @ (point_world - anchor_origin)`. For a MANO hand, the translation is `R @ (t_world - anchor_origin) + (R - I) @ J0`, where `R` is `anchor_rotation` and `J0` is the corresponding wrist-root correction. Stored hand states already include this correction; do not apply it a second time.

Displayed RGB is rotated 90 degrees clockwise relative to the native linear camera. The placement routine accounts for this rotation before camera unprojection.

## Forecast outputs

For `K` requested draws, `forecast.npz` contains:

| Array | Shape | Meaning |
|---|---|---|
| `state` | `[K, 60, 66]` | Observed prefix and predicted future, in metric state coordinates |
| `hand_state` | `[K, 30, 2, 24]` | Future left/right hands |
| `bound_object_state` | `[K, 30, 2, 9]` | Future left/right-bound objects |
| `object_slots` | `[2]` | Bound object slots; a negative slot denotes no bound object |
| `hand_valid`, `object_valid` | `[60, 2]` | Observed and forecast availability masks |
| `mano_betas` | `[10]` | Hand shape used by the example |
| `draw_indices` | `[K]` | Zero-based draw indices |
| `seeds` | `[K]` | Unsigned 64-bit RNG seeds |
| `acg_*` | Field-specific | Online predicted contact, bottom/support, and landing conditions |

Future availability is derived from observed presence. Future ground-truth availability is not used to condition forecasts. Availability masks and stored visualization settings, including hidden hands and displayed object slots, determine visibility. They do not modify the generated trajectories.

`metadata.json` records bindings, parsed intention, base seed, noise profile, draws, DDIM configuration, device, versions, and intention provenance. `run.json` records the selected examples, elapsed time, and whether intentions were produced in that command or explicitly supplied.

## Comparison artifacts

`ground_truth/` and `reference_outputs/` are separate from inference inputs. The former contains annotated intentions and reference futures; the latter contains archived Gaze2HOI forecasts associated with the project page.

For full-scene GT visualization, the gallery uses `ground_truth/scene_motion.npz`. Compact `ground_truth/motion.npz` describes the two bound objects and is also accepted by the renderer. The viewer labels the newly rendered forecast separately from its optional archived website clip.

`visualization/config.json` stores the original camera, crop, rendering background, and object appearance. Mesh assets are shared across samples. The runtime applies no temporal smoothing, interpolation-based motion refinement, or optimization to the predicted trajectories.
