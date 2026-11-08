class ByteTokenizer:
    VOCAB_SIZE = 260

    BOS = 256
    EOS = 257
    PAD = 258
    IMAGE = 259

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]:
        tokens = list(text.encode("utf-8"))
        if add_bos:
            tokens = [self.BOS] + tokens
        if add_eos:
            tokens = tokens + [self.EOS]

        return tokens

    def decode(self, tokens: list[int]) -> str:
        tokens = [int(t) for t in tokens]

        for token in tokens:
            if not 0 <= token < self.VOCAB_SIZE:
                raise ValueError("Invalid token ID")

        # Ignore special tokens when decoding
        byte_values = [token for token in tokens if token < 256]

        return bytes(byte_values).decode("utf-8")

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))
