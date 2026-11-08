from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp
from flax.training.train_state import TrainState


def checkpoint_payload(state: TrainState) -> dict[str, Any]:
    """Exclude Python functions and optimizer definitions."""
    return {"step": state.step, "params": state.params, "opt_state": state.opt_state}


def save_checkpoint(
    path: str | Path,
    state: TrainState,
    *,
    overwrite: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(
            destination,
            checkpoint_payload(state),
            force=overwrite,
            custom_metadata=metadata,
        )

        # The save operation is asynchronous
        checkpointer.wait_until_finished()


def restore_checkpoint(path: str | Path, template: TrainState) -> TrainState:
    """Restore into an already initialized TrainState."""
    source = Path(path).resolve()

    if not source.exists():
        raise FileNotFoundError(source)

    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(source, checkpoint_payload(template))

    return template.replace(
        step=restored["step"],
        params=restored["params"],
        opt_state=restored["opt_state"],
    )
