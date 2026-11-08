from collections.abc import Callable

import jax
import numpy as np
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from synthesize_vl.data.schema import TrainingBatch


def prepare_data_parallel(
    step_fn: Callable,
    state: TrainState,
    microbatches: TrainingBatch,
    *,
    num_devices: int,
) -> tuple[Callable, TrainState, TrainingBatch]:
    """Place replicated state and sharded batches on devices."""
    available = jax.local_devices()

    if not 1 <= num_devices <= len(available):
        raise ValueError("Requested device count is unavailable")

    microbatch_size = microbatches.input_ids.shape[1]

    if microbatch_size % num_devices:
        raise ValueError("Microbatch size must be divisible by device count")

    devices = available[:num_devices]

    mesh = Mesh(np.asarray(devices), ("data",))

    replicated = NamedSharding(mesh, P())

    # Dimension 0 selects the accumulation step.
    # Dimension 1 is partitioned across devices
    partitioned = NamedSharding(mesh, P(None, "data"))

    state_sharding = jax.tree_util.tree_map(lambda _: replicated, state)
    batch_sharding = jax.tree_util.tree_map(lambda _: partitioned, microbatches)

    device_state = jax.device_put(state, state_sharding)
    device_batch = jax.device_put(microbatches, batch_sharding)

    compiled_step = jax.jit(
        step_fn,
        in_shardings=(state_sharding, batch_sharding),
        out_shardings=(state_sharding, replicated),
    )

    return compiled_step, device_state, device_batch
