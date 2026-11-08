import pytest

from synthesize_vl.pretraining.vm import (
    IfLess,
    Operation,
    Program,
    Repeat,
    VMLimits,
    execute_program,
    serialize_execution,
)
from synthesize_vl.pretraining.vm_curriculum import (
    Difficulty,
    generate_vm_document,
    generate_vm_documents,
)


def test_difficulty_rejects_invalid_control_flow_settings() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        Difficulty("bad_probability", 1, 1, 0, 1, -0.1, 0.2, 1, 2)

    with pytest.raises(ValueError, match="max_repeat"):
        Difficulty("bad_repeat", 1, 1, 0, 1, 0.1, 0.0, 1, 1)


def test_vm_executes_arithmetic_loops_and_branches() -> None:
    program = Program(
        initial=(2, 0),
        body=(
            Operation("ADD", 0, 3),
            Repeat(
                count=2,
                body=(
                    Operation("MUL", 0, 2),
                    Operation("EMIT", 0),
                ),
            ),
            IfLess(register=1, threshold=10, body=(Operation("SET", 1, 7),)),
            Operation("COPY", 1, 0),
        ),
    )

    result = execute_program(program)

    assert result.registers == (20, 20)
    assert result.output == bytes([10, 20])
    assert result.steps == 9
    assert result.states[-1] == (20, 20)

    trace = serialize_execution(program, result, format="trace")
    binary = serialize_execution(program, result, format="binary")

    assert b"O:0a14" in trace
    assert binary.endswith(bytes([10, 20]))


def test_vm_modular_arithmetic() -> None:
    program = Program(
        initial=(250, 0),
        body=(
            Operation("ADD", 0, 10),
            Operation("SUB", 0, 8),
            Operation("XOR", 0, 15),
            Operation("EMIT", 0),
        ),
    )
    result = execute_program(program, VMLimits(modulus=256))
    expected = 252 ^ 15

    assert result.registers == (expected, 0)
    assert result.output == bytes([expected])


def test_vm_rejects_resource_violations() -> None:
    repeat = Program(
        initial=(0, 0),
        body=(
            Repeat(
                5,
                (Operation("ADD", 0, 1),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="steps"):
        execute_program(repeat, VMLimits(max_steps=3))

    with pytest.raises(ValueError, match="repeat"):
        execute_program(repeat, VMLimits(max_repeat=4))

    with pytest.raises(ValueError, match="depth"):
        execute_program(repeat, VMLimits(max_depth=0))

    output_overflow = Program(
        initial=(0, 0),
        body=(
            Repeat(
                3,
                (Operation("EMIT", 0),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="output"):
        execute_program(output_overflow, VMLimits(max_output_bytes=2))

    with pytest.raises(ValueError, match="register"):
        execute_program(
            Program(
                (0, 0),
                (Operation("SET", 2, 1),),
            )
        )


def test_vm_generator_is_stateless() -> None:
    first = generate_vm_document(
        seed=27, split="train", index=123, max_payload_bytes=511
    )
    second = generate_vm_document(
        seed=27, split="train", index=123, max_payload_bytes=511
    )

    assert first == second

    documents = generate_vm_documents(
        seed=27, split="train", start=122, count=3, max_payload_bytes=511
    )

    assert documents[1] == first


def test_vm_split_and_training_families() -> None:
    train = generate_vm_document(seed=27, split="train", index=123)
    validation = generate_vm_document(seed=27, split="validation", index=123)

    assert train.sample_id != validation.sample_id

    families = {
        generate_vm_document(seed=10, split="train", index=index).family
        for index in range(80)
    }

    assert families == {"vm_foundation", "vm_intermediate", "vm_advanced"}
