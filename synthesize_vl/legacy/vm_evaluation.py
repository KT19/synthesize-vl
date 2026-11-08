from typing import Literal

from synthesize_vl.data.schema import TrainingBatch
from synthesize_vl.legacy.pretraining import pack_pretraining_documents
from synthesize_vl.pretraining.vm_curriculum import VMProfile, generate_vm_documents


def make_vm_evaluation_batches(
    *,
    seed: int,
    split: Literal["validation", "test"],
    examples_per_profile: int,
    max_length: int,
    out_of_distribution: bool = False,
) -> dict[str, TrainingBatch]:
    if examples_per_profile < 1:
        raise ValueError("examples_per_profile must be positive")

    profiles: tuple[VMProfile, ...] = (
        ("heldout_length", "heldout_values")
        if out_of_distribution
        else ("foundation", "intermediate", "advanced")
    )

    batches: dict[str, TrainingBatch] = {}

    for family_index, profile in enumerate(profiles):
        documents = [
            generate_vm_documents(
                seed=seed,
                split=split,
                start=(family_index + index * len(profiles)),
                count=1,
                profile=profile,
                max_payload_bytes=max_length - 1,
            )[0]
            for index in range(examples_per_profile)
        ]

        batches[f"vm_{profile}"] = pack_pretraining_documents(
            documents, max_length=max_length
        )

    return batches
