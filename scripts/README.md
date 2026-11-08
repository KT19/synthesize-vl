# Scripts

Run commands from the repository root using `uv run scripts/<script>.py`. The project environment is defined by `pyproject.toml`.

Training hyperparameters below are examples and may change during tuning. Read the
corresponding YAML config for the settings of a particular run.

## Current training

| Script | Purpose | Configuration |
| --- | --- | --- |
| `scripts/run_pretrain.py` | Pretrain the shared model using the teacher architecture on VM execution/traces and structured visual tasks | `configs/pretrain.yaml` |
| `scripts/run_train_sft.py` | Initialize from a checkpoint and train text/visual arithmetic responses for either role | `configs/sft_teacher.yaml` or `configs/sft_student.yaml` |
| `scripts/run_train_opd.py` | Train the student on its own sampled continuations using a frozen teacher | `configs/opd.yaml` |
| `scripts/run_train_grpo.py` | Optimize grouped student responses using exact final-answer rewards | `configs/grpo.yaml` |
| `scripts/run_preview.py` | Preview SFT questions, responses, images, and a packed batch | `configs/sft_data.yaml` |

### Commands

```bash
uv run scripts/run_pretrain.py --config configs/pretrain.yaml
uv run scripts/run_train_sft.py --config configs/sft_student.yaml
uv run scripts/run_train_sft.py --config configs/sft_teacher.yaml
uv run scripts/run_train_opd.py --config configs/opd.yaml
uv run scripts/run_train_grpo.py --config configs/grpo.yaml
uv run scripts/run_preview.py
```

### Architecture configuration

Pretraining and teacher SFT load `configs/teacher_architecture.yaml`;
student SFT loads `configs/student_architecture.yaml`. The `architecture_config`
setting selects the architecture in each pretraining/SFT YAML file, with paths
resolved from that config's directory. OPD loads both architectures through
`teacher_architecture_config` and `student_architecture_config`; its current
paths are relative to the repository root. Checkpoint configuration digests
include the loaded architecture values. GRPO uses `architecture_config` with
the same working-directory-relative path convention as OPD.

### Checkpoint flow

Pretraining writes `checkpoints/pretrain`. Student SFT loads those parameters,
uses two recurrent passes, and writes `checkpoints/student`. Teacher SFT then
loads the trained student parameters, uses four recurrent passes, and writes
`checkpoints/teacher`. The core weights are shared across passes, so both models
have matching parameter shapes. Teacher SFT starts a fresh optimizer and data
stream; its step count measures additional training after student SFT.

OPD loads both SFT checkpoints, freezes the teacher, and updates the student
with reverse KL on student-generated prefixes. It writes `checkpoints/opd`.

GRPO loads that distilled student, samples response groups, and uses the
arithmetic oracle's binary final-answer rewards to update it. It writes
`checkpoints/grpo` and does not load a teacher.

See the [training flow diagram](../README.md#training-flow) and the
[evaluation results](../README.md#evaluation-results).

### Evaluation and seeds

Student and teacher generation evaluation request worked solutions and report
final-answer accuracy, grouped by task and difficulty. The generation limit is
configurable, e.g., up to 512 tokens per response.

Both configs use `evaluation.seed` for held-out loss and generation evaluation.
Use the same value to compare both models on the same problems; `data.seed`
controls training examples independently.

OPD and GRPO also use `evaluation.seed` for evaluation and `rollout.seed` for
sampling. OPD derives sampling keys from optimizer updates; GRPO derives them
from the data-batch index so skipped batches still get distinct keys.

### GRPO steps and sampling

For example, GRPO's `data.batch_size: 4` and `grpo.group_size: 8` produce 32
responses per rollout batch. With `rollout.ppo_epochs: 2`, each batch receives
two optimizer updates on the same responses when at least one group has reward
variation. Every GRPO prompt requests a worked solution.

| Counter | Unit |
| --- | --- |
| Runtime step limit, reporting intervals, and checkpoint directory numbers | Completed GRPO steps, each including all PPO epochs |
| Optimizer schedule and saved optimizer counters | Individual optimizer updates |

An example budget is 100 GRPO steps and 200 optimizer updates with two PPO epochs.
Batches with no group-relative reward signal are skipped;
`runtime.max_uninformative_streak` controls when consecutive skipped batches stop
the run, e.g., after 10 batches.

### Saved outputs and resuming

Only the pretraining runner writes CSV metric history, together with a resolved
configuration. SFT, OPD, and GRPO print metrics and save checkpoints. The current
training runners support `runtime.resume_from` with an exact checkpoint directory.

### Data preview

`run_preview.py` generates data without loading `pretrained_checkpoint_root`
and writes visual examples to `runs/sft_preview/`.

## CLI inference

`scripts/run_demo.py` loads model parameters using `configs/demo.yaml` and runs
greedy inference. Its menu offers generated text/visual arithmetic, custom text
expressions, difficulty selection, and a worked-response toggle. Generated
examples show an oracle answer; custom expressions have no oracle.

```bash
uv run scripts/run_demo.py --config configs/demo.yaml
uv run scripts/run_demo.py --config configs/demo.yaml --once
```

`--once` generates one visual example and exits. Images are saved under
`save_dir`; `open_image` controls the external image viewer. Select the
architecture and checkpoint together, including the intended recurrent pass count.

## Smoke checks

`scripts/smoke/` contains all `run_*_tiny.py` runners and the attention,
masking, gating, generation, mHC, multimodal, primitive, recurrent, and VM
profile demonstrations. Embedding and vision demonstrations also live here.
`experiment_utils.py` supplies shared tiny batches
and experiment helpers.

```bash
uv run scripts/smoke/run_attention_masks.py
uv run scripts/smoke/run_embedding.py
uv run scripts/smoke/run_vision.py
uv run scripts/smoke/run_train_multimodal_tiny.py
uv run scripts/smoke/run_train_scaled_tiny.py --devices 1 --steps 3
uv run scripts/smoke/run_pretrain_tiny.py --help
```

Many tiny training checks repeatedly train on a fixed batch and report
teacher-forced token accuracy or first-answer-byte accuracy. These checks
verify training mechanics; their accuracy is different from complete,
generated-answer accuracy on unseen SFT examples. `run_pretrain_tiny` instead
streams fresh documents through a small model.

## Legacy runners

`scripts/legacy/` retains earlier command-line training experiments:

| Script | Data and behavior |
| --- | --- |
| `scripts/legacy/run_pretrain_stream.py` | Text-only document streaming, with legacy trace/byte generators or VM programs |
| `scripts/legacy/run_pretrain_multimodal.py` | VM full-document prediction mixed with structured visual completions |
| `scripts/legacy/run_pretrain_joint.py` | VM full/execution supervision and visual completions, using the earlier slot-based sampling configuration |

```bash
uv run scripts/legacy/run_pretrain_stream.py --help
uv run scripts/legacy/run_pretrain_multimodal.py --help
uv run scripts/legacy/run_pretrain_joint.py --help
```

These runners share some library code with current training. Their location
identifies superseded entry points, rather than unused datasets or modules.

## Data families

All data in these paths is synthetic.

| Source | Examples and role |
| --- | --- |
| `synthesize_vl/legacy/synthetic.py` | Single-operation arithmetic and predicting the printed result of short arithmetic programs; used by early smoke demonstrations |
| `synthesize_vl/legacy/vision.py` | Digit, letter, shape, and color recognition/counting in scenes of 1–4 objects; used by component checks |
| `synthesize_vl/legacy/generators.py` | Earlier register traces, stack traces, grid walks, and repeating byte patterns |
| `synthesize_vl/pretraining/task_stream.py` | Current VM and visual pretraining mixture: inventory, color counts, spatial relations, and weighted sums |
| `synthesize_vl/sft/dataset.py` | Current text arithmetic and visual arithmetic instruction/response examples, with worked solutions or direct answers |
| `synthesize_vl/opd/data.py` | Reuses SFT prompts and images without oracle responses; continuations are sampled by the student |
| `synthesize_vl/grpo/data.py` | Repeats SFT prompts into response groups and scores completed samples against the arithmetic oracle |

Current visual pretraining and visual SFT share `pretraining/world.py`.
SFT adds natural-language instructions and reasoning traces, with its own
difficulty and sampling settings.
See the [SFT difficulty tables](../README.md#sft-difficulty) for the current
operation, operand, object, and weight ranges.

Shared document and cursor types live in `synthesize_vl/data/schema.py`.
See the [library guide](../synthesize_vl/README.md) for the package layout.
