from synthesize_vl.pretraining.vm_curriculum import VMProfile, generate_vm_document


def main() -> None:
    profiles: tuple[VMProfile, ...] = (
        "foundation",
        "intermediate",
        "advanced",
        "heldout_length",
        "heldout_values",
    )

    for index, profile in enumerate(profiles):
        document = generate_vm_document(
            seed=42,
            split="validation",
            index=index,
            profile=profile,
            max_payload_bytes=511,
        )

        assert document.family == f"vm_{profile}"
        assert 0 < len(document.payload) <= 511

        print(f"\n=== {profile} ===")
        print("Document ID:", document.sample_id)
        print("Bytes:", len(document.payload))
        print("Payload preview:", repr(document.payload[:160]))

    print("PASS")


if __name__ == "__main__":
    main()
