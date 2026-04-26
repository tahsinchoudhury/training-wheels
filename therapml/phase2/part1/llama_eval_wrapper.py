from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

import lm_eval
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.utils import EnhancedJSONEncoder, make_table

from therapml.phase2.part1.models.llama import Llama
from therapml.phase2.part1.tokenizer.bpe import BPETokenizer


DEFAULT_MODEL_PATH = "models/checkpoints/checkpoint_epoch_5.pth"
DEFAULT_TOKENIZER_PATH = "data/tinystories_bpe.json"
DEFAULT_MAX_GEN_TOKS = 64


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return _repo_root() / candidate


def _coerce_batch_size(batch_size: int | str | None) -> int:
    if batch_size is None:
        return 1
    if isinstance(batch_size, str):
        if batch_size == "auto":
            raise ValueError("batch_size='auto' is not supported by LlamaEvalWrapper; pass an integer batch size.")
        batch_size = int(batch_size)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return batch_size


def _parse_limit(value: str | None) -> int | float | None:
    if value is None:
        return None
    parsed = float(value)
    if math.isclose(parsed, int(parsed)):
        return int(parsed)
    return parsed


def _as_stop_list(until: str | Iterable[str] | None) -> list[str]:
    if until is None:
        return []
    if isinstance(until, str):
        return [until]
    return [str(stop) for stop in until]


class LlamaEvalWrapper(LM):
    """lm-eval-harness adapter for the local TherapML Llama model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
        device: str | torch.device | None = None,
        batch_size: int | str | None = 1,
        max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
        **_: Any,
    ) -> None:
        super().__init__()

        self.model_path = _resolve_path(model_path)
        self.tokenizer_path = _resolve_path(tokenizer_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._batch_size = _coerce_batch_size(batch_size)
        self.max_gen_toks = int(max_gen_toks)
        if self.max_gen_toks <= 0:
            raise ValueError("max_gen_toks must be a positive integer")

        self.tokenizer = BPETokenizer.load(self.tokenizer_path)
        self.model = Llama.load(self.model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.pad_token_id = self._required_token_id("<pad>")
        self.bos_token_id = self._required_token_id("<bos>")
        self.eot_token_id = self._required_token_id("<eos>")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return int(self.model.context_length)

    @property
    def tokenizer_name(self) -> str:
        return str(self.tokenizer_path)

    def _required_token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required special token {token!r}")
        return int(token_id)

    def tok_encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def tok_decode(self, tokens: int | Sequence[int], *, skip_special_tokens: bool = True) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _context_tokens(self, context: str) -> list[int]:
        context_tokens = self.tok_encode(context)
        return context_tokens if context_tokens else [self.bos_token_id]

    def _pad_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(tensor.shape[0] for tensor in tensors)
        batch = torch.full(
            (len(tensors), max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        for idx, tensor in enumerate(tensors):
            batch[idx, : tensor.shape[0]] = tensor
        return batch

    def _score_token_requests(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        *,
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        if not requests:
            return []

        results: list[tuple[float, bool]] = []
        progress = tqdm(
            total=len(requests),
            disable=disable_tqdm or self.rank != 0,
            desc="Running loglikelihood requests",
        )

        with torch.inference_mode():
            for start in range(0, len(requests), self.batch_size):
                chunk = requests[start : start + self.batch_size]
                input_tensors: list[torch.Tensor] = []
                input_lengths: list[int] = []
                continuation_tokens: list[list[int]] = []

                for _, context_enc, continuation_enc in chunk:
                    if not context_enc:
                        context_enc = [self.bos_token_id]
                    if not continuation_enc:
                        input_tensors.append(torch.tensor(context_enc[-self.max_length :], device=self.device))
                        input_lengths.append(min(len(context_enc), self.max_length))
                        continuation_tokens.append([])
                        continue
                    if len(continuation_enc) > self.max_length:
                        raise ValueError(
                            f"Continuation length ({len(continuation_enc)}) exceeds model context length "
                            f"({self.max_length}); cannot score it in one causal window."
                        )

                    tokens = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                    input_ids = tokens[:-1]
                    input_tensors.append(torch.tensor(input_ids, dtype=torch.long, device=self.device))
                    input_lengths.append(len(input_ids))
                    continuation_tokens.append(continuation_enc)

                batched_input = self._pad_batch(input_tensors)
                log_probs = F.log_softmax(self.model(batched_input), dim=-1)

                for (cache_key, _, _), seq_log_probs, input_len, cont_tokens in zip(
                    chunk,
                    log_probs,
                    input_lengths,
                    continuation_tokens,
                    strict=True,
                ):
                    if not cont_tokens:
                        answer = (0.0, True)
                    else:
                        cont_len = len(cont_tokens)
                        scored_logits = seq_log_probs[input_len - cont_len : input_len]
                        cont_tensor = torch.tensor(cont_tokens, dtype=torch.long, device=self.device)
                        greedy_tokens = scored_logits.argmax(dim=-1)
                        selected = torch.gather(scored_logits, 1, cont_tensor.unsqueeze(-1)).squeeze(-1)
                        answer = (float(selected.sum().item()), bool(torch.equal(greedy_tokens, cont_tensor)))

                    results.append(answer)
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                    progress.update(1)

        progress.close()
        return results

    def loglikelihood(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        token_requests = []
        for request in requests:
            context, continuation = request.args
            context = str(context)
            continuation = str(continuation)
            token_requests.append(
                (
                    (context, continuation),
                    self._context_tokens(context),
                    self.tok_encode(continuation),
                )
            )

        return self._score_token_requests(token_requests, disable_tqdm=disable_tqdm)

    def loglikelihood_rolling(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[float]:
        results: list[float] = []

        for request in tqdm(
            requests,
            disable=disable_tqdm or self.rank != 0,
            desc="Preparing rolling loglikelihood requests",
        ):
            (text,) = request.args
            token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(str(text)),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            token_requests = [(None, context, continuation) for context, continuation in token_windows]
            scored = self._score_token_requests(token_requests, disable_tqdm=True)
            total = float(sum(logprob for logprob, _ in scored))
            results.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (str(text),), total)

        return results

    def generate_until(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[str]:
        outputs: list[str] = []

        for request in tqdm(
            requests,
            disable=disable_tqdm or self.rank != 0,
            desc="Running generate_until requests",
        ):
            context, gen_kwargs = request.args
            gen_kwargs = dict(gen_kwargs or {})
            until = _as_stop_list(gen_kwargs.get("until"))
            max_new_tokens = int(
                gen_kwargs.get(
                    "max_gen_toks",
                    gen_kwargs.get("max_new_tokens", gen_kwargs.get("max_tokens", self.max_gen_toks)),
                )
            )
            max_new_tokens = max(1, max_new_tokens)

            context_tokens = self._context_tokens(str(context))
            input_ids = torch.tensor([context_tokens], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids,
                    device=self.device,
                    max_new_tokens=max_new_tokens,
                    eos_id=self.eot_token_id,
                )

            generated_tokens = generated_ids[0, len(context_tokens) :].tolist()
            generated_text = self.tok_decode(generated_tokens)
            for stop in until:
                if stop and stop in generated_text:
                    generated_text = generated_text.split(stop, 1)[0]
                    break

            outputs.append(generated_text)
            self.cache_hook.add_partial("generate_until", (str(context), gen_kwargs), generated_text)

        return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the local TherapML Llama model with lm-eval-harness.")
    parser.add_argument("--tasks", nargs="+", required=True, help="lm-eval task names, e.g. mmlu_jurisprudence.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Model checkpoint path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument(
        "--tokenizer-path",
        default=DEFAULT_TOKENIZER_PATH,
        help=f"Tokenizer JSON path. Default: {DEFAULT_TOKENIZER_PATH}",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, cuda:0. Default: auto.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of loglikelihood requests per model batch.")
    parser.add_argument("--max-gen-toks", type=int, default=DEFAULT_MAX_GEN_TOKS, help="Default tokens for generate_until.")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Number of few-shot examples. Default: 0.")
    parser.add_argument("--limit", type=_parse_limit, default=None, help="Limit examples per task, as an int or fraction.")
    parser.add_argument("--output-path", default=None, help="Optional JSON file to write raw lm-eval results.")
    parser.add_argument("--log-samples", action="store_true", help="Ask lm-eval to retain per-sample outputs.")
    parser.add_argument("--bootstrap-iters", type=int, default=100000, help="Bootstrap iterations for stderr calculation.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    wrapper = LlamaEvalWrapper(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        batch_size=args.batch_size,
        max_gen_toks=args.max_gen_toks,
    )
    results = lm_eval.simple_evaluate(
        model=wrapper,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=args.log_samples,
        bootstrap_iters=args.bootstrap_iters,
    )

    if results is None:
        return 0

    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2, cls=EnhancedJSONEncoder)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
