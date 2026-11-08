from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from flax import linen as nn

from synthesize_vl.data.schema import TrainingBatch
from synthesize_vl.inference.generation import generate_greedy
from synthesize_vl.pretraining.joint_data import (
    JointDocument,
    from_visual_document,
    from_vm_document,
    pack_joint_documents,
)
from synthesize_vl.pretraining.visual import FAMILIES as VISUAL_FAMILIES
from synthesize_vl.pretraining.visual import generate_visual_document
from synthesize_vl.pretraining.vm_curriculum import generate_vm_document

Split = Literal["validation", "test"]


def make_joint_evaluation_documents(
    *,
    seed: int,
    split: Split,
    examples_per_family: int,
    max_length: int,
    out_of_distribution: bool = False,
) -> dict[str, list[JointDocument]]:
    if examples_per_family < 1:
        raise ValueError("examples_per_family must be positive")

    groups: dict[str, list[JointDocument]] = {}

    vm_profiles = (
        ("heldout_length", "heldout_values")
        if out_of_distribution
        else ("foundation", "intermediate", "advanced")
    )

    for profile_index, profile in enumerate(vm_profiles):
        for execute_only in (False, True):
            mode = "execute" if execute_only else "full"
            name = f"vm_{profile}_{mode}"

            documents: list[JointDocument] = []

            for i in range(examples_per_family):
                index = (profile_index + 1) * 100000 + 2 * i + int(execute_only)

                original = generate_vm_document(
                    seed=seed,
                    split=split,
                    index=index,
                    profile=profile,
                    max_payload_bytes=max_length - 1,
                )

                documents.append(from_vm_document(original, execute_only=execute_only))

            groups[name] = documents

    for family_index, family in enumerate(VISUAL_FAMILIES):
        profile = "heldout_objects" if out_of_distribution else "mixed"

        name = (
            f"visual_{family}_ood_pbjects"
            if out_of_distribution
            else f"visual_{family}"
        )

        groups[name] = [
            from_visual_document(
                generate_visual_document(
                    seed=seed,
                    split=split,
                    index=(500000 + family_index * 10000 + i),
                    profile=profile,
                    family=family,
                )
            )
            for i in range(examples_per_family)
        ]

    if out_of_distribution:
        groups["visual_weighted_sum_ood_weights"] = [
            from_visual_document(
                generate_visual_document(
                    seed=seed,
                    split=split,
                    index=900000 + i,
                    profile="heldout_weights",
                    family="weighted_sum",
                )
            )
            for i in range(examples_per_family)
        ]

    return groups


def pack_evaluation_groups(
    groups: Mapping[str, list[JointDocument]], *, max_length: int
) -> dict[str, TrainingBatch]:
    return {
        name: pack_joint_documents(documents, max_length=max_length)
        for name, documents in groups.items()
    }


@dataclass(frozen=True, slots=True)
class ExactTaskMetric:
    family: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)


def evaluate_exact_tasks(
    model: nn.Module,
    params: Any,
    groups: Mapping[str, list[JointDocument]],
    *,
    examples_per_family: int = 1,
) -> tuple[ExactTaskMetric, ...]:
    """Free-running exact-match evaluation including EOS"""

    if examples_per_family < 1:
        raise ValueError("examples_per_family must be positive")

    results: list[ExactTaskMetric] = []

    for family, documents in groups.items():
        if not documents or documents[0].mode == "vm_full":
            continue

        selected = documents[:examples_per_family]
        correct = 0

        for document in selected:
            prompt = document.prefix.decode("ascii")

            generation = generate_greedy(
                model,
                params,
                prompt,
                image=document.image,
                max_new_tokens=len(document.completion) + 1,
            )

            expected = tuple(document.completion)

            if generation.token_ids == expected and generation.stopped_on_eos:
                correct += 1

        results.append(
            ExactTaskMetric(family=family, correct=correct, total=len(selected))
        )

    return tuple(results)


def print_exact_report(
    metrics: tuple[ExactTaskMetric, ...],
) -> None:
    print("\nFree-running exact-answer evaluation")
    print(f"{'Task':<42} {'Correct':>8} {'Accuracy':>10}")

    for item in metrics:
        print(
            f"{item.family:<42} {item.correct:>3}/{item.total:<4} {item.accuracy:>9.1%}"
        )

    total_correct = sum(item.correct for item in metrics)
    total = sum(item.total for item in metrics)

    print("-" * 64)
    print(
        f"{'OVERALL':<42} {total_correct:>3}/{total:<4} {total_correct / max(total, 1):>9.1%}"
    )
