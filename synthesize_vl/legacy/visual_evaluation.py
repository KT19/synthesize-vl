from typing import Literal

from synthesize_vl.data.schema import TrainingBatch
from synthesize_vl.legacy.multimodal_pretraining import pack_mixed_documents
from synthesize_vl.pretraining.visual import FAMILIES, generate_visual_document


def make_visual_evaluation_batches(
    *,
    seed: int,
    split: Literal["validation", "test"],
    examples_per_family: int,
    max_length: int,
    out_of_distribution: bool = False,
) -> dict[str, TrainingBatch]:
    if examples_per_family < 1:
        raise ValueError("examples_per_family must be positive")

    batches: dict[str, TrainingBatch] = {}

    for family_index, family in enumerate(FAMILIES):
        profile = "heldout_objects" if out_of_distribution else "mixed"

        documents = [
            generate_visual_document(
                seed=seed,
                split=split,
                index=(family_index + i * len(FAMILIES)),
                profile=profile,
                family=family,
            )
            for i in range(examples_per_family)
        ]

        name = (
            f"visual_{family}_ood_objects"
            if out_of_distribution
            else f"visual_{family}"
        )

        batches[name] = pack_mixed_documents(documents, max_length=max_length)

    if out_of_distribution:
        documents = [
            generate_visual_document(
                seed=seed,
                split=split,
                index=10000 + i,
                profile="heldout_weights",
                family="weighted_sum",
            )
            for i in range(examples_per_family)
        ]

        batches["visual_weighted_sum_ood_weights"] = pack_mixed_documents(
            documents, max_length=max_length
        )

    return batches
