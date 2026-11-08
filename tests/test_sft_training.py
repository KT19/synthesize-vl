from types import SimpleNamespace

import numpy as np

from synthesize_vl.data.schema import StreamCursor
from synthesize_vl.sft import model_evaluation
from synthesize_vl.sft.config import SFTDataConfig
from synthesize_vl.sft.stream import next_sft_batch


def test_sft_stream_is_reproducible():
    config = SFTDataConfig(seed=17, batch_size=4, text_probability=0.0, cot_probability=1.0)

    start = StreamCursor()

    first, advanced = next_sft_batch(config, start)
    second, replay_cursor = next_sft_batch(config, start)

    assert advanced == replay_cursor

    np.testing.assert_array_equal(first.input_ids, second.input_ids)
    np.testing.assert_array_equal(first.images, second.images)

    assert int(first.has_image.sum()) == 4

    assert advanced.next_index == 4
    assert advanced.updates == 1
    assert advanced.documents_seen == 4

    assert advanced.targets_seen == int(first.loss_mask.sum())


def test_generatoin_decodes_byte_tokens(monkeypatch):
    def fake_generate(*args, **kwargs):
        return SimpleNamespace(token_ids=tuple(b"Answer: 17\n"), stopped_on_eos=True)

    monkeypatch.setattr(model_evaluation, "generate_greedy", fake_generate)

    response = model_evaluation.generate_sft_response(
        model=None,  # type: ignore
        params=None,
        prompt="Question: Compute 8 + 9.",
        image=None,
        max_new_tokens=30,
    )

    assert response == "Answer: 17\n"


def test_generation_rejects_incomplete_output(monkeypatch):
    def fake_generate(*args, **kwargs):
        return SimpleNamespace(token_ids=tuple(b"Answer: 17\n"), stopped_on_eos=False)

    monkeypatch.setattr(model_evaluation, "generate_greedy", fake_generate)

    response = model_evaluation.generate_sft_response(
        model=None,  # type: ignore
        params=None,
        prompt="Question",
        image=None,
        max_new_tokens=10,
    )

    assert response == ""
