import jax.numpy as jnp
import numpy as np
import pytest

from synthesize_vl.data.schema import PretrainingDocument
from synthesize_vl.legacy.generators import (
    FAMILIES,
    generate_document,
    generate_documents,
)
from synthesize_vl.legacy.pretraining import bits_per_byte, pack_pretraining_documents
from synthesize_vl.legacy.programs import (
    RegisterInstruction,
    StackInstruction,
    execute_register,
    execute_stack,
)
from synthesize_vl.tokenizer import ByteTokenizer
from synthesize_vl.training.losses import masked_lm_loss


def test_register_interpreter() -> None:
    program = [
        RegisterInstruction("ADD", 3),
        RegisterInstruction("MUL", 2),
        RegisterInstruction("SUB", 4),
    ]

    assert execute_register(2, program) == (5, 10, 6)


def test_stack_interpreter_and_underflow() -> None:
    program = [
        StackInstruction("PUSH", 2),
        StackInstruction("PUSH", 4),
        StackInstruction("ADD"),
        StackInstruction("PUSH", 3),
        StackInstruction("MUL"),
    ]

    assert execute_stack(program) == (
        (2,),
        (2, 4),
        (6,),
        (6, 3),
        (18,),
    )

    with pytest.raises(ValueError, match="underflow"):
        execute_stack([StackInstruction("ADD")])


def test_generator_is_deterministic_and_stateless() -> None:
    first = generate_document(seed=42, split="train", index=101)
    second = generate_document(seed=42, split="train", index=101)

    assert first == second

    # Reaching index 101 does not depend on earlier calls
    documents = generate_documents(seed=42, split="train", start=100, count=3)

    assert documents[1] == first

    validation = generate_document(seed=42, split="validation", index=101)

    assert validation.sample_id != first.sample_id
    assert validation.payload != first.payload


def test_all_families_generate_bounded_documents() -> None:
    documents = generate_documents(seed=17, split="train", start=0, count=80)

    assert {d.family for d in documents} == set(FAMILIES)
    assert len({d.sample_id for d in documents}) == 80

    for document in documents:
        assert isinstance(document.payload, bytes)
        assert len(document.payload) > 0
        assert len(document.payload) + 1 <= 128


def raw_document(payload: bytes) -> PretrainingDocument:
    return PretrainingDocument(
        sample_id="unit:000", split="train", family="unit", payload=payload
    )


def test_full_sequence_packing_with_binary_bytes() -> None:
    batch = pack_pretraining_documents(
        [raw_document(bytes([0, 128, 255, 65]))], max_length=8
    )

    expected_input = [
        ByteTokenizer.BOS,
        0,
        128,
        255,
        65,
        ByteTokenizer.PAD,
        ByteTokenizer.PAD,
        ByteTokenizer.PAD,
    ]

    expected_target = [
        0,
        128,
        255,
        65,
        ByteTokenizer.EOS,
        ByteTokenizer.PAD,
        ByteTokenizer.PAD,
        ByteTokenizer.PAD,
    ]

    np.testing.assert_array_equal(batch.input_ids[0], expected_input)
    np.testing.assert_array_equal(batch.target_ids[0], expected_target)

    np.testing.assert_array_equal(batch.loss_mask[0], [True] * 5 + [False] * 3)

    assert not bool(batch.has_image[0])
    assert int(batch.loss_mask.sum()) == 5


def test_bits_per_byte_excludes_eos() -> None:
    batch = pack_pretraining_documents([raw_document(b"ABCD")], max_length=8)
    logits = jnp.zeros((1, 8, 260))

    reference = bits_per_byte(logits, batch)

    np.testing.assert_allclose(reference, np.log2(260.0), rtol=1e-5)
    # The EOS target occupies position 4
    changed = logits.at[0, 4, ByteTokenizer.EOS].set(30.0)

    # Improving EOS prediction should change LM loss, but NOT the byte-only evaluation metric
    np.testing.assert_allclose(bits_per_byte(changed, batch), reference, rtol=1e-5)

    assert float(masked_lm_loss(changed, batch)) < float(masked_lm_loss(logits, batch))
