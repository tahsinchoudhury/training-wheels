from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
HARNESS_DIR = ROOT_DIR / "lm-evaluation-harness"
for path in (ROOT_DIR, HARNESS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lm_eval
from lm_eval.utils import EnhancedJSONEncoder, make_table

from therapml.phase2.part1.llama_eval_wrapper import LlamaEvalWrapper


class EvalJSONEncoder(EnhancedJSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


DEFAULT_TASKS = (
    "mmlu_college_computer_science",
    "mmlu_elementary_mathematics",
    "mmlu_jurisprudence",
)
TASK_LABELS = {
    "mmlu_college_computer_science": "Computer Science",
    "mmlu_elementary_mathematics": "Elementary Math",
    "mmlu_jurisprudence": "Jurisprudence",
}


def _parse_limit(value: str | None) -> int | float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    if math.isclose(parsed, int(parsed)):
        return int(parsed)
    return parsed


def _device_arg(device: str) -> str | None:
    return None if device == "auto" else device


def _metric(result: dict[str, Any], task: str, metric: str) -> Any:
    task_result = result.get("results", {}).get(task, {})
    if metric in task_result:
        return task_result[metric]
    for key, value in task_result.items():
        if key.startswith(f"{metric},"):
            return value
    return None


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def _comparison_rows(
    *,
    tasks: Sequence[str],
    local_results: dict[str, Any],
    gpt2_results: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        local_acc = _metric(local_results, task, "acc")
        gpt2_acc = _metric(gpt2_results, task, "acc")
        delta = None
        if isinstance(local_acc, (float, int)) and isinstance(gpt2_acc, (float, int)):
            delta = float(local_acc) - float(gpt2_acc)
        rows.append(
            {
                "task": task,
                "subject": TASK_LABELS.get(task, task),
                "local_acc": local_acc,
                "gpt2_acc": gpt2_acc,
                "delta_local_minus_gpt2": delta,
                "local_acc_stderr": _metric(local_results, task, "acc_stderr"),
                "gpt2_acc_stderr": _metric(gpt2_results, task, "acc_stderr"),
            }
        )
    return rows


def _print_comparison(rows: Sequence[dict[str, Any]]) -> None:
    headers = ("Subject", "Local Acc", "GPT-2 Acc", "Delta")
    table_rows = [
        (
            row["subject"],
            _format_value(row["local_acc"]),
            _format_value(row["gpt2_acc"]),
            _format_value(row["delta_local_minus_gpt2"]),
        )
        for row in rows
    ]
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *table_rows, strict=False)
    ]

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(str(value).ljust(width) for value, width in zip(row, widths, strict=True))

    print("\nComparison")
    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(fmt(row))


def _evaluate_local(args: argparse.Namespace) -> dict[str, Any]:
    wrapper = LlamaEvalWrapper(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=_device_arg(args.device),
        batch_size=args.batch_size,
        max_gen_toks=args.max_gen_toks,
    )
    return lm_eval.simple_evaluate(
        model=wrapper,
        tasks=list(args.tasks),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
    )


def _evaluate_gpt2(args: argparse.Namespace) -> dict[str, Any]:
    model_args = args.gpt2_model_args or f"pretrained={args.gpt2_model},dtype={args.gpt2_dtype}"
    return lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=list(args.tasks),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.gpt2_batch_size or args.batch_size,
        device=_device_arg(args.device),
        bootstrap_iters=args.bootstrap_iters,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare local TherapML Llama and GPT-2 on selected MMLU tasks.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--model-path", default="models/checkpoints/mmlu_finetuned.pth")
    parser.add_argument("--tokenizer-path", default="data/tinystories_bpe.json")
    parser.add_argument("--gpt2-model", default="gpt2-large")
    parser.add_argument("--gpt2-model-args", default="", help="Optional raw lm-eval HF model_args override.")
    parser.add_argument("--gpt2-dtype", default="float32")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpt2-batch-size", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=_parse_limit, default=25)
    parser.add_argument("--max-gen-toks", type=int, default=64)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output-path", default="", help="Optional explicit JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Limit per task: {args.limit if args.limit is not None else 'full'}")
    print(f"Device: {args.device}")

    print("\nEvaluating local TherapML checkpoint...")
    local_results = _evaluate_local(args)
    print(make_table(local_results))

    print("\nEvaluating GPT-2 baseline...")
    gpt2_results = _evaluate_gpt2(args)
    print(make_table(gpt2_results))

    comparison = _comparison_rows(
        tasks=args.tasks,
        local_results=local_results,
        gpt2_results=gpt2_results,
    )
    _print_comparison(comparison)

    output_path = Path(args.output_path) if args.output_path else None
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.log_dir) / f"mmlu_gpt2_comparison_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "tasks": list(args.tasks),
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
            "device": args.device,
            "local_model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "gpt2_model": args.gpt2_model,
            "gpt2_model_args": args.gpt2_model_args or f"pretrained={args.gpt2_model},dtype={args.gpt2_dtype}",
        },
        "comparison": comparison,
        "local_results": local_results,
        "gpt2_results": gpt2_results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=EvalJSONEncoder)

    print(f"\nSaved comparison to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
