"""Run lm-evaluation-harness against a running vLLM server (see serve_vllm.sh).

Usage:
  python evals/run_evals.py --run-name bf16                       # default tasks, quick limits
  python evals/run_evals.py --run-name bf16 --tasks gsm8k mmlu --full
  python evals/run_evals.py --run-name bf16 --thinking            # gsm8k with Qwen3 thinking on
  python evals/run_evals.py --run-name bf16 --tasks humaneval --allow-code-exec

Each task is its own lm-eval call, in one of two groups:
  raw   loglikelihood scoring on plain text, no chat template (wikitext, mmlu, arc_challenge)
  chat  generation with the Qwen3 chat template (gsm8k, humaneval, mbpp); --thinking applies here

Results go to outputs/evals/<run-name>/, plus a summary.json with every task's metrics. Re-running tasks under
an existing run name updates only those tasks in summary.json.

gsm8k and mbpp use local fixes of the stock lm-eval tasks (evals/tasks/): gsm8k_robust adds a "math-extract"
answer filter that handles \\boxed{} answers, and mbpp_instruct_fixed fixes a code extractor that drops "def".
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import lm_eval
from lm_eval.tasks import TaskManager

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_api  # noqa: E402,F401  registers the "qwen3-vllm" model

ROOT = Path(__file__).resolve().parent.parent
# Local fixes of stock lm-eval tasks; see the comment at the top of each yaml
TASK_DIR = Path(__file__).resolve().parent / "tasks"

# name -> lm-eval task, group, few-shot count, quick-mode limit (None = all docs), and whether the
# task prefills the assistant turn (prefilled tasks always run with thinking off).
TASKS = {
    "wikitext": dict(task="wikitext", group="raw", fewshot=None, quick_limit=None),
    "mmlu": dict(task="mmlu", group="raw", fewshot=5, quick_limit=20),  # limit is per subject (57)
    "arc_challenge": dict(task="arc_challenge", group="raw", fewshot=25, quick_limit=300),
    "gsm8k": dict(task="gsm8k_robust", group="chat", fewshot=5, quick_limit=250),
    "humaneval": dict(task="humaneval_instruct", group="chat", fewshot=None, quick_limit=None, code=True,
                      prefill=True),
    "mbpp": dict(task="mbpp_instruct_fixed", group="chat", fewshot=None, quick_limit=200, code=True,
                 prefill=True),
}
DEFAULT_TASKS = ["wikitext", "mmlu", "arc_challenge", "gsm8k"]

# Context length for raw scoring is fixed so wikitext perplexity stays comparable across runs
RAW_MAX_LENGTH = 4096


def run_task(name, args, out_dir, task_manager):
    spec = TASKS[name]
    chat = spec["group"] == "chat"
    thinking = args.thinking and chat and not spec.get("prefill")
    limit = args.limit if args.limit is not None else None if args.full else spec["quick_limit"]
    model_args = dict(
        base_url=f"{args.base_url}/v1/completions",
        model=args.served_name,
        tokenizer=args.tokenizer,
        tokenizer_backend="huggingface",
        tokenized_requests=True,
        max_length=(32768 if thinking else 8192) if chat else RAW_MAX_LENGTH,
        num_concurrent=args.concurrency,
        max_retries=3,
        # Prefilled tasks can't think (the template closes the think block), so they keep the task's own
        # stop strings and token limit
        enable_thinking=thinking,
        thinking_max_gen_toks=args.thinking_budget,
    )
    print(f"\n=== {name} ({spec['task']}, {spec['group']}, limit={limit})", flush=True)
    start = time.time()
    results = lm_eval.simple_evaluate(
        model="qwen3-vllm",
        model_args=model_args,
        tasks=[spec["task"]],
        task_manager=task_manager,
        num_fewshot=spec["fewshot"],
        limit=limit,
        batch_size=args.batch_size,
        apply_chat_template=chat,
        fewshot_as_multiturn=chat,
        confirm_run_unsafe_code=args.allow_code_exec,
        log_samples=True,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    elapsed = time.time() - start
    # Per-question model outputs; mmlu stores them under each of its 57 subject tasks
    with (out_dir / f"samples_{name}.jsonl").open("w") as f:
        for subtask, samples in results.get("samples", {}).items():
            for s in samples:
                f.write(json.dumps({"subtask": subtask, **s}, default=str) + "\n")
    # lm-eval metric keys look like "acc,none" (metric,filter); everything else is bookkeeping
    metrics = {k: v for k, v in results["results"].get(spec["task"], {}).items() if "," in k}
    return {
        "task": spec["task"],
        "limit": limit,
        "num_fewshot": spec["fewshot"],
        "thinking": thinking,
        "seconds": round(elapsed, 1),
        "metrics": {k: v for k, v in metrics.items() if "stderr" not in k},
        "stderr": {k: v for k, v in metrics.items() if "stderr" in k},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True, help="label for this model variant, e.g. bf16, nvfp4, fp3-b16")
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=DEFAULT_TASKS)
    ap.add_argument("--full", action="store_true", help="run every doc instead of the quick limits")
    ap.add_argument("--limit", type=int, default=None, help="override the doc limit for all tasks")
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode for chat tasks (default off)")
    ap.add_argument("--thinking-budget", type=int, default=16384, help="max generated tokens when thinking")
    ap.add_argument("--allow-code-exec", action="store_true",
                    help="required for humaneval/mbpp: executes model-generated code on this machine")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--served-name", default="qwen3-30b-a3b")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    code_tasks = [n for n in args.tasks if TASKS[n].get("code")]
    if code_tasks and not args.allow_code_exec:
        ap.error(f"{', '.join(code_tasks)} execute generated code; pass --allow-code-exec to run them")
    if code_tasks:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    if args.thinking:
        prefilled = [n for n in args.tasks if TASKS[n].get("prefill")]
        if prefilled:
            print(f"note: {', '.join(prefilled)} prefill the answer, so they run with thinking off")

    out_dir = ROOT / "outputs" / "evals" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "summary.json"
    # Re-running some tasks under an existing run name replaces just those tasks' entries
    all_tasks = json.loads(summary_file.read_text())["tasks"] if summary_file.exists() else {}
    task_manager = TaskManager(include_path=str(TASK_DIR))
    summary = {}
    for name in args.tasks:
        summary[name] = all_tasks[name] = run_task(name, args, out_dir, task_manager)
        # Write after every task so a crash mid-run keeps what finished
        summary_file.write_text(json.dumps({"config": vars(args), "tasks": all_tasks}, indent=2))
    print(f"\n=== {args.run_name}  (thinking={'on' if args.thinking else 'off'}, "
          f"{'full' if args.full else 'quick'})")
    for name, s in summary.items():
        metrics = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in s["metrics"].items())
        print(f"{name:14} {metrics}")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
