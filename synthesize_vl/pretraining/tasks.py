from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class TaskArm:
    name: str
    modality: Literal["vm", "visual"]
    profile: str
    family: str | None = None


ARMS: tuple[TaskArm, ...] = (
    TaskArm("vm_foundation", "vm", "foundation"),
    TaskArm("vm_intermediate", "vm", "intermediate"),
    TaskArm("vm_advanced", "vm", "advanced"),
) + tuple(
    TaskArm(
        name=f"visual_{family}_{profile}",
        modality="visual",
        family=family,
        profile=profile,
    )
    for family in ("inventory", "color_counts", "spatial", "weighted_sum")
    for profile in ("foundation", "intermediate", "advanced")
)

# Equal modality shares, uniform over each modality's task/profile combinations.
TASK_PROBABILITIES: tuple[float, ...] = tuple(
    0.5 / 3 if arm.modality == "vm" else 0.5 / 12 for arm in ARMS
)


def sample_arm(*, seed: int, index: int) -> TaskArm:
    """Sample reproducibly from the fixed task mixture."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 2017]))
    chosen = int(rng.choice(len(ARMS), p=TASK_PROBABILITIES))

    return ARMS[chosen]
