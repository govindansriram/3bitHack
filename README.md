# 3bithack

Quantizing Qwen3-30B-A3B (a 30B-parameter MoE, 3B active) toward 3-bit (FP3) weights. Quantization
variants are evaluated by writing fake-quantized checkpoints (quantize, then dequantize back to BF16) and
serving them with vLLM, so no low-bit kernels are needed until a format is chosen.

## Layout

```
evals/
  setup_server.sh   one-time GPU server setup: venv, vLLM, lm-eval, model download
  serve_vllm.sh     start a vLLM server for a checkpoint folder (port 8000)
  run_evals.py      run lm-evaluation-harness tasks against the server
  run_variant.sh    serve one checkpoint, run evals against it, shut the server down
  qwen_api.py       lm-eval model: Qwen3 chat template + thinking on/off over /v1/completions
scripts/
  sync.sh           push code to the server / pull results back
outputs/evals/      results, one folder per run (gitignored)
```

Code is written in this repo and synced to the server; the server only runs it. GPU instances (vast.ai,
RunPod) don't persist, so pull results back after every run.

## Server setup (once per machine)

Needs a B200 (or other GPU vLLM supports) with ~200 GB of disk.

```bash
# local
SSH_KEY=~/.ssh/id_ed25519 scripts/sync.sh push <port> root@<host>

# on the server
mkdir -p /workspace/logs
tmux new -s setup 'bash /workspace/3bithack/evals/setup_server.sh 2>&1 | tee /workspace/logs/setup.log'
```

`setup_server.sh` reuses vast.ai's `/venv/main` if present and otherwise creates `/workspace/venv` (the path
is saved in `/workspace/.venv_path`). It installs the CUDA cuBLAS/cuRAND dev headers if the image lacks them;
FlashInfer needs them to compile vLLM's B200 MoE kernel on first start.

## Start the vLLM server

```bash
source "$(cat /workspace/.venv_path)/bin/activate"
tmux new -s vllm '/workspace/3bithack/evals/serve_vllm.sh 2>&1 | tee /workspace/logs/vllm.log'
# or a quantized variant:  evals/serve_vllm.sh /workspace/models/<variant-dir>
```

The first start on a new machine compiles and auto-tunes the MoE kernel (~10 minutes). The server listens on
`127.0.0.1:8000` with the fixed model name `qwen3-30b-a3b`. Check it with `curl localhost:8000/health`.

## Run evals

On the server, with the venv active and from `/workspace/3bithack`:

```bash
python evals/run_evals.py --run-name bf16                                  # quick: default tasks, subsets
python evals/run_evals.py --run-name bf16 --tasks gsm8k mmlu --full        # whole test sets
python evals/run_evals.py --run-name bf16 --thinking                       # Qwen3 thinking on
python evals/run_evals.py --run-name bf16_golden --full --thinking --allow-code-exec \
  --tasks wikitext mmlu arc_challenge gsm8k humaneval mbpp                 # everything
```

| Task | Scoring | Full set | Quick run | Default |
|---|---|---|---|---|
| wikitext | word perplexity (lower is better), 4096-token windows | 62 docs | all | yes |
| mmlu | likelihood, 5-shot | 14,042 | 20 per subject (1,140) | yes |
| arc_challenge | likelihood, 25-shot | 1,172 | 300 | yes |
| gsm8k | generation, 5-shot, chat template | 1,319 | 250 | yes |
| humaneval | generation, chat template, runs generated code | 164 | all | needs `--allow-code-exec` |
| mbpp | generation, chat template, runs generated code | 500 | 200 | needs `--allow-code-exec` |

- **Quick runs** take the first N questions of each task, so every run scores the same subset and variants
  compare fairly. Use `--full` for final numbers.
- **`--thinking`** only changes gsm8k. wikitext, mmlu and arc_challenge score fixed options and never
  generate; humaneval and mbpp prefill the start of the answer, which makes Qwen3's template turn thinking off.
- **`--allow-code-exec`** is required for humaneval/mbpp because they execute model-written code.

Two tasks are local fixes of the stock lm-eval tasks, in `evals/tasks/` (prompts and scoring otherwise unchanged):

- **`gsm8k_robust`**: stock gsm8k plus a `math-extract` answer filter that reads `\boxed{}`, then `####`, then
  the last number. The stock `strict-match` and `flexible-extract` scores are still reported.
- **`mbpp_instruct_fixed`**: stock mbpp_instruct's code extractor treats a leading `def` as a language tag and
  drops it, so every answer fails. The fixed extractor takes the code up to the closing fence.

vLLM's OpenAI API accepts at most 4 stop strings per request, so `qwen_api.py` sends the first 4 and applies the
rest to the reply.

Each run writes `outputs/evals/<run-name>/summary.json` (scores, standard errors, settings, timings) and
`samples_<task>.jsonl` (every prompt, model output and whether it was scored correct). lm-eval computes the
scores; `run_evals.py` writes the files.

To evaluate another checkpoint end to end (stop any running vLLM server first):

```bash
evals/run_variant.sh /workspace/models/Qwen3-30B-A3B-FP8 fp8_qwen \
  --tasks wikitext mmlu arc_challenge gsm8k humaneval mbpp --full --thinking --allow-code-exec
```

## Pull results back

```bash
SSH_KEY=~/.ssh/id_ed25519 scripts/sync.sh pull <port> root@<host>   # -> outputs/evals/
```

## Query the server directly

```bash
ssh -p <port> -i ~/.ssh/id_ed25519 -N -L 8000:localhost:8000 root@<host>    # tunnel, leave running

curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3-30b-a3b",
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

## Results

### BF16 quick baseline (thinking off)

Run before the gsm8k fix, so gsm8k here is the stock task (no math-extract); with thinking off the model uses
the `####` format, which the stock filters score correctly.

| Task | Score | ± | Questions |
|---|---|---|---|
| wikitext | word perplexity 11.16 | | 62 docs |
| mmlu (5-shot) | 81.1% | 1.1 | 1,140 |
| arc_challenge (25-shot) | 67.7% (acc_norm 67.0%) | 2.7 | 300 |
| gsm8k (5-shot) | 92.0% flexible / 90.4% strict | 1.7 | 250 |

### BF16 golden (full test sets, thinking on)

The reference every quantized variant is compared against. Run on a RunPod B200 with vLLM 0.34.0 and
lm-eval 0.4.13; results in `outputs/evals/bf16_golden/`.

| Task | Score | ± | Questions | Thinking | Time |
|---|---|---|---|---|---|
| wikitext | word perplexity 11.16 | | 62 docs | n/a | 32 s |
| mmlu (5-shot) | 79.5% | 0.3 | 14,042 | n/a | 19 min |
| arc_challenge (25-shot) | 65.9% (acc_norm 68.6%) | 1.4 | 1,172 | n/a | 2 min |
| gsm8k (5-shot) | 94.8% math-extract (89.8% flexible, 78.9% strict) | 0.6 | 1,319 | on | 3 min |
| humaneval | 92.1% pass@1 | 2.1 | 164 | off (prefilled) | 26 s |
| mbpp | 74.0% pass@1 | 2.0 | 500 | off (prefilled) | 71 s |

For gsm8k with thinking, read **math-extract**: the model answers in `$$\boxed{N}$$`, which the stock strict
and flexible filters mis-score (flexible extracts `"$$"`).

### Published quantized checkpoints vs BF16 (same settings as the golden)

The reference points FP3 has to match on quality (and beat on speed). Both quantize activations as well as
weights, so they are end-to-end targets rather than weight-only comparisons. Differences vs BF16 in brackets;
± is BF16's standard error.

| Task | Metric | BF16 | FP8 (Qwen) | NVFP4 (NVIDIA) |
|---|---|---|---|---|
| wikitext | perplexity ↓ | 11.16 | 11.19 (+0.02) | 11.73 (+0.56) |
| mmlu | acc | 79.5 ±0.3 | 79.5 (-0.0) | 78.1 (-1.5) |
| arc_challenge | acc_norm | 68.6 ±1.4 | 68.7 (+0.1) | 67.4 (-1.2) |
| gsm8k | math-extract | 94.8 ±0.6 | 95.1 (+0.4) | 93.9 (-0.9) |
| humaneval | pass@1 | 92.1 ±2.1 | 93.3 (+1.2) | 87.8 (-4.3) |
| mbpp | pass@1 | 74.0 ±2.0 | 74.4 (+0.4) | 73.8 (-0.2) |

| Checkpoint | Format | Weights in GPU memory | vLLM kernels | Results |
|---|---|---|---|---|
| BF16 | `Qwen/Qwen3-30B-A3B` | 57 GiB | FlashInfer TRT-LLM BF16 MoE | `outputs/evals/bf16_golden/` |
| FP8 | `Qwen/Qwen3-30B-A3B-FP8`: block FP8 weights, dynamic FP8 activations (W8A8) | 29.1 GiB | FlashInfer TRT-LLM FP8 MoE | `outputs/evals/fp8_qwen/` |
| NVFP4 | `nvidia/Qwen3-30B-A3B-NVFP4`: NVFP4 weights and activations (W4A4), FP8 KV cache | 16.9 GiB | FlashInfer NVFP4 GEMM + TRT-LLM NVFP4 MoE | `outputs/evals/nvfp4_nvidia/` |

- **FP8** is indistinguishable from BF16: every difference is within noise.
- **NVFP4** loses a little everywhere: +5% perplexity, -1.5 MMLU (well outside MMLU's ±0.3), -4.3 HumanEval
  (about 2 standard errors). Its checkpoint has no KV-cache scales, so vLLM uses a scale of 1.0 for the FP8 KV
  cache, which may account for part of the loss.
- Eval wall-clock times are not a speed measurement (lm-eval's request pattern dominates them); speed needs
  a dedicated throughput/latency benchmark.

## Notes: what to measure for 3-bit work

- Distribution shape: absmax, std, kurtosis, and the fraction of weights beyond 6σ, per tensor type and per layer.
- Spread within groups: how much the range varies across output channels and across 128-weight groups.
- Quantization error: relative error after simple 3-bit rounding (group size 128, symmetric vs asymmetric), which shows which layers and projections suffer most.
- Differences between experts: do some experts have more extreme outliers than others?
