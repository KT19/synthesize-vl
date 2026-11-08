from synthesize_vl.tokenizer import ByteTokenizer

tokenizer = ByteTokenizer()


def test_ascii_roundtrip():
    text = "x = 2 + 3"
    tokens = tokenizer.encode(text)

    assert tokens[0] == tokenizer.BOS
    assert tokens[-1] == tokenizer.EOS
    assert tokenizer.decode(tokens) == text


def test_unicode_roundtrip():
    text = "π + λ = 形"

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_no_special_tokens():
    tokens = tokenizer.encode("ABC", add_bos=False, add_eos=False)

    assert tokens == [65, 66, 67]
