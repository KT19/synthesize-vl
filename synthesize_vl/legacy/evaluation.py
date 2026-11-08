from typing import Literal

from synthesize_vl.data.schema import TrainingBatch
from synthesize_vl.legacy.generators import FAMILIES, generate_document
from synthesize_vl.legacy.pretraining import pack_pretraining_documents

EvaluationSplit = Literal["validation", "test"]


def make_evaluation_batches(
    *, seed: int, split: EvaluationSplit, examples_per_family: int, max_length: int
) -> dict[str, TrainingBatch]:
    """Create a fixed evaluation set, balanced by family."""
    if examples_per_family < 1:
        raise ValueError("examples_per_family must be positive")

    result: dict[str, TrainingBatch] = {}

    for family_index, family in enumerate(FAMILIES):
        documents = [
            generate_document(
                seed=seed, split=split, index=family_index + len(FAMILIES) * i
            )
            for i in range(examples_per_family)
        ]

        if any(doc.family != family for doc in documents):
            raise AssertionError("Generator family mismatch")

        result[family] = pack_pretraining_documents(documents, max_length=max_length)

    return result
