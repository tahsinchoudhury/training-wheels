import math
from pathlib import Path

import torch

from lm_eval.api.instance import Instance

from therapml.phase2.part1.llama_eval_wrapper import LlamaEvalWrapper
from therapml.phase2.part1.models.llama import Llama, LlamaConfig
from therapml.phase2.part1.tokenizer.bpe import BPETokenizer, BPETrainingConfig


def _build_test_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    texts = [
        "Once upon a time, a model answered A.",
        "Question: Which choice is correct? Answer: A",
        "The tiny evaluator can stop at STOP or continue.",
        "A B C D answer choice jurisprudence math computer science",
    ]
    tokenizer = BPETokenizer.train_from_iterator(
        texts,
        config=BPETrainingConfig(vocab_size=128, min_frequency=1),
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    torch.manual_seed(0)
    model = Llama(
        LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=16,
            d_model=16,
            num_layers=1,
            num_heads=4,
            d_ff=32,
            rope_theta=10000.0,
        )
    )
    model_path = tmp_path / "model.pth"
    model.save(model_path)
    return model_path, tokenizer_path


def _wrapper(tmp_path: Path, *, batch_size: int = 2) -> LlamaEvalWrapper:
    model_path, tokenizer_path = _build_test_artifacts(tmp_path)
    return LlamaEvalWrapper(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device="cpu",
        batch_size=batch_size,
        max_gen_toks=4,
    )


def test_wrapper_loads_model_tokenizer_and_special_tokens(tmp_path):
    wrapper = _wrapper(tmp_path)

    assert wrapper.max_length == 16
    assert wrapper.batch_size == 2
    assert wrapper.pad_token_id == wrapper.tokenizer.token_to_id("<pad>")
    assert wrapper.bos_token_id == wrapper.tokenizer.token_to_id("<bos>")
    assert wrapper.eot_token_id == wrapper.tokenizer.token_to_id("<eos>")
    assert wrapper.model.training is False


def test_loglikelihood_returns_finite_scores_for_regular_and_empty_context(tmp_path):
    wrapper = _wrapper(tmp_path)
    requests = [
        Instance("loglikelihood", {}, ("Question: Which choice is correct? Answer:", " A"), 0),
        Instance("loglikelihood", {}, ("", "A"), 1),
    ]

    results = wrapper.loglikelihood(requests, disable_tqdm=True)

    assert len(results) == 2
    for logprob, is_greedy in results:
        assert math.isfinite(logprob)
        assert isinstance(is_greedy, bool)


def test_loglikelihood_rolling_returns_finite_score(tmp_path):
    wrapper = _wrapper(tmp_path)
    requests = [
        Instance("loglikelihood_rolling", {}, ("Once upon a time, a model answered A.",), 0),
    ]

    results = wrapper.loglikelihood_rolling(requests, disable_tqdm=True)

    assert len(results) == 1
    assert math.isfinite(results[0])


def test_generate_until_truncates_at_stop_sequence(tmp_path):
    wrapper = _wrapper(tmp_path)
    generated_tokens = wrapper.tok_encode(" answer STOP extra")

    def fake_generate(input_ids, *, device, max_new_tokens, eos_id=None):
        continuation = torch.tensor([generated_tokens[:max_new_tokens]], dtype=torch.long, device=device)
        return torch.cat((input_ids, continuation), dim=1)

    wrapper.model.generate = fake_generate
    requests = [
        Instance("generate_until", {}, ("Question:", {"until": ["STOP"], "max_gen_toks": len(generated_tokens)}), 0),
    ]

    results = wrapper.generate_until(requests, disable_tqdm=True)

    assert len(results) == 1
    assert "STOP" not in results[0]
    assert results[0] == wrapper.tok_decode(generated_tokens).split("STOP", 1)[0]
