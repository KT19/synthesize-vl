# Library layout

## Package overview

| Package | Role |
| --- | --- |
| `data/` | Shared sample, document, batch, and stream cursor types, plus general sample packing |
| `models/` | Reusable model components and shared architecture configuration |
| `training/` | Optimizers, losses, gradient accumulation, distributed updates, and state checkpoints |
| `inference/` | Text and multimodal generation |
| `pretraining/` | Current VM/visual generators, task sampling, evaluation, history, and resumable pretraining |
| `sft/` | Current arithmetic instruction datasets, SFT streams, checkpoint initialization, and evaluation |
| `opd/` | Student rollout sampling and reverse-KL distillation using a frozen teacher |
| `grpo/` | Grouped rollouts, exact-answer rewards, group-relative advantages, and clipped policy updates |
| `legacy/` | Earlier toy datasets, full-document packers, and superseded streaming pipelines |

## Shared data types

`Split`, `PretrainingDocument`, and `StreamCursor` live in `data/schema.py`.
Current pretraining, SFT, OPD, GRPO, and older runners share these definitions.
`StreamCursor` tracks the position in the deterministic data stream and the
number of optimizer updates.

## Model architecture

Pretraining and SFT load the model architecture from `configs/teacher_architecture.yaml`
or `configs/student_architecture.yaml`, referenced by each training config's
`architecture_config` setting. `models/architecture.py` parses it into
`ModelArchitecture`; `build_model()` maps those values to the Flax model.

`core_layers` selects the number of distinct layers in the recurrent core;
`recurrent_steps` selects how often the entire stack is applied. Standard and
mHC residual paths both reuse each layer's parameters across iterations.

## Training stages

### Pretraining

`scripts/run_pretrain.py` loads `pretraining/config.py` and creates VM/visual
batches through `pretraining/task_stream.py`. `pretraining/tasks.py` defines the
fixed task mixture, and `pretraining/joint_data.py` packs examples with the
appropriate supervision masks.

### Supervised fine-tuning (SFT)

Student and teacher SFT share `sft/train_config.py` and `scripts/run_train_sft.py`;
`configs/sft_teacher.yaml` and `configs/sft_student.yaml` select the architecture
and source checkpoint.

The student first loads `checkpoints/pretrain` and trains with two recurrent
passes, saving to `checkpoints/student`. Teacher SFT loads those student
parameters and trains with four passes, saving to `checkpoints/teacher`.
`sft/checkpoints.py` restores parameters into a model with the same parameter
structure; additional recurrent passes reuse the core weights. Each new SFT stage
starts with a fresh optimizer and data cursor.

See the [evaluation results](../README.md#evaluation-results) for the completed runs.

### On-policy distillation (OPD)

OPD uses `opd/config.py` to load both architectures through
`teacher_architecture_config` and `student_architecture_config`.
`scripts/run_train_opd.py` initializes the student and frozen teacher from their
SFT checkpoints.

`opd/data.py` packs prompts and images without oracle responses;
`opd/core.py` samples student continuations and updates the student using the
teacher's next-token distributions at the same prefixes. Both loaded
architectures are included in the checkpoint configuration identity.

### Group relative policy optimization (GRPO)

GRPO uses `grpo/config.py` and `scripts/run_train_grpo.py` to initialize the policy
from `checkpoints/opd`.

`grpo/data.py` repeats each worked-solution prompt into a response group, then
scores EOS-terminated responses using `sft/schema.py`'s exact final-answer verifier.
It normalizes rewards within each group.

`grpo/core.py` gathers sampled-token log probabilities and applies the clipped
policy objective; all PPO epochs reuse the same rollouts and old-policy log
probabilities. Rollout packing and sampling reuse `opd/data.py` and `opd/core.py`.

See the [training flow diagram](../README.md#training-flow) for how the stages connect.

## Inference

`inference/generation.py` provides greedy text/image generation with full-prefix
forwards and a fixed padded text buffer. `scripts/run_demo.py` uses this path
after restoring only model parameters through `sft/checkpoints.py`. KV-cached
generation is available for the text-only `CausalLM`.

## Checkpoints and step counts

GRPO reports one outer step after all PPO epochs. Its saved `TrainState.step` and
`StreamCursor.updates` remain individual optimizer-update counters, matching
`pretraining/checkpoints.py`. GRPO checkpoint names use the outer GRPO step;
other training stages use the optimizer-update count.

## Evaluation and history

SFT, OPD, and GRPO use `evaluation.seed` for held-out examples independently of
the training `data.seed`. OPD and GRPO also have a separate `rollout.seed` for
sampling.

`pretraining/history.py` writes raw per-update training and per-task validation
CSV files for `run_pretrain.py`. The other current runners print their metrics.

`sft/image_dependence.py` provides an additional counterfactual-image evaluation
helper, which is not called by the current training entry points.

## Legacy code

| Module | Contents |
| --- | --- |
| `legacy/synthetic.py` | Single-operation math and short-program output prediction |
| `legacy/vision.py` | Digit, letter, shape, and color scenes for component checks |
| `legacy/generators.py`, `legacy/programs.py` | Register/stack traces, grid walks, and repeating byte documents |
| `legacy/pretraining.py`, `legacy/multimodal_pretraining.py` | Earlier document packing and byte-loss helpers |
| `legacy/stream.py`, `legacy/multimodal_stream.py`, `legacy/joint_stream.py` | Earlier streaming configurations and batch generators |
| `legacy/evaluation.py`, `legacy/vm_evaluation.py`, `legacy/visual_evaluation.py` | Evaluation-set builders for those older runners |

The common evaluation metrics and loss calculation remain in
`pretraining/evaluation.py`. Current VM profiles and the visual world renderer
remain in `pretraining/`, including `vm_curriculum.py`, `visual.py`, and
`world.py`.

## Component demonstrations

Executable demonstrations live under `scripts/smoke/`. For example:

```bash
uv run scripts/smoke/run_embedding.py
uv run scripts/smoke/run_vision.py
```

The vision demonstration writes its image and metadata under `outputs/vision/`.
