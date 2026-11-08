import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from synthesize_vl.data.schema import TrainingBatch

InputsFn = Callable[
    [TrainingBatch],
    tuple[jax.Array, ...],
]


@dataclass(frozen=True, slots=True)
class FamilyMetrics:
    family: str
    examples: int
    bytes_evaluated: int
    nats_per_target: float
    bits_per_byte: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    families: tuple[FamilyMetrics, ...]
    overall_nats_per_target: float
    overall_bits_per_byte: float


@jax.jit
def _loss_sums(
    state: TrainState, batch: TrainingBatch, inputs: tuple[jax.Array, ...]
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return numerators and denominators, not batch means."""
    logits = state.apply_fn({"params": state.params}, *inputs)

    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.astype(jnp.float32), labels=batch.target_ids
    )

    target_mask = batch.loss_mask.astype(jnp.float32)

    byte_mask = (batch.loss_mask & (batch.target_ids < 256)).astype(jnp.float32)

    return (
        jnp.sum(losses * target_mask),
        jnp.sum(target_mask),
        jnp.sum(losses * byte_mask),
        jnp.sum(byte_mask),
    )


def evaluate_families(
    state: TrainState, batches: Mapping[str, TrainingBatch], inputs_fn: InputsFn
) -> EvaluationReport:
    reports: list[FamilyMetrics] = []

    total_target_loss = 0.0
    total_targets = 0.0
    total_byte_loss = 0.0
    total_bytes = 0.0

    for family, batch in batches.items():
        target_loss, target_count, byte_loss, byte_count = _loss_sums(
            state, batch, inputs_fn(batch)
        )

        nt = float(target_count)
        nb = float(byte_count)
        lt = float(target_loss)
        lb = float(byte_loss)

        if nt <= 0 or nb <= 0:
            raise ValueError("Evaluation set is empty")

        reports.append(
            FamilyMetrics(
                family=family,
                examples=int(batch.input_ids.shape[0]),
                bytes_evaluated=int(nb),
                nats_per_target=lt / nt,
                bits_per_byte=lb / nb / math.log(2.0),
            )
        )

        total_target_loss += lt
        total_targets += nt
        total_byte_loss += lb
        total_bytes += nb

    if not reports:
        raise ValueError("No evaluation families")

    return EvaluationReport(
        families=tuple(reports),
        overall_nats_per_target=total_target_loss / total_targets,
        overall_bits_per_byte=total_byte_loss / total_bytes / math.log(2.0),
    )


def print_report(report: EvaluationReport, *, label: str) -> None:
    print(f"\n{label}")
    print(f"{'Family':<20} {'Bytes':>8} {'Nats/target':>13} {'BPB':>9}")

    for metric in report.families:
        print(
            f"{metric.family:<20} {metric.bytes_evaluated:>8,} {metric.nats_per_target:>13.4f} {metric.bits_per_byte:>9.4f}"
        )

    print("-" * 55)
    print(
        f"{'OVERALL':<20} {sum(x.bytes_evaluated for x in report.families):>8,} "
        f"{report.overall_nats_per_target:>13.4f} {report.overall_bits_per_byte:>9.4f}"
    )
