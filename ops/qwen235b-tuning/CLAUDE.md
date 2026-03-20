# Qwen 235B Parameter Tuning Agent

You are an autonomous parameter tuning agent. Your goal: find optimal sampling parameters for the **Qwen3-235B-A22B-Thinking-2507-AWQ** model running on SGLang. The target is clean output (no repetition) at `max_tokens=16384`.

## Your Loop

1. Read `ops/qwen235b-tuning/prd.json` — find the highest-priority story with `"passes": false`
2. Read `ops/qwen235b-tuning/progress.txt` — load prior learnings and context
3. Execute the story using `test_params.py` (see Tool section below)
4. Update `ops/qwen235b-tuning/prd.json` — set `"passes": true` for completed stories
5. Append findings to `ops/qwen235b-tuning/progress.txt`
6. If changes were made to Ansible files (QT-009), commit with message: `ops: [QT-XXX] - <description>`
7. If ALL stories have `"passes": true`, output `<promise>COMPLETE</promise>` and stop

## Tool: test_params.py

Non-interactive test harness. Sends streaming requests to SGLang, detects repetition in real-time (guard) and post-hoc (analysis), outputs JSON to stdout.

### CLI

```bash
python ops/qwen235b-tuning/test_params.py \
  --prompt analytical --max-tokens 4096 \
  -t 0.6 --top-p 0.95 --top-k 40 --min-p 0.1 \
  --presence-penalty 1.0 --frequency-penalty 0.0 --repetition-penalty 0.0

# Run all 3 prompts:
python ops/qwen235b-tuning/test_params.py --prompt all --max-tokens 4096 -t 0.6

# Save full output text for manual inspection:
python ops/qwen235b-tuning/test_params.py --prompt creative --save-output --max-tokens 8192
```

### Available prompts

- **analytical**: Microservices vs monolith trade-off analysis — tests sustained analytical reasoning
- **creative**: Cyberpunk short story — tests narrative coherence, prone to loop patterns
- **technical**: Full HTTP request lifecycle deep dive — tests structured technical explanation

### Output schema

```json
{
  "prompt": "analytical",
  "params": {"max_tokens": 4096, "temperature": 0.6, "top_p": 0.95, ...},
  "elapsed_s": 245.3,
  "completion_tokens": 4096,
  "reasoning_tokens_est": 256,
  "finish_reason": "length",
  "guard_triggered": false,
  "guard_reason": null,
  "content_chars": 18234,
  "repetition": {
    "overall": 0.023,
    "severity": "none",
    "ngram": 0.031,
    "sentence": 0.018,
    "loop": 0.0
  },
  "summary": "[NONE] score=0.02 — no issues"
}
```

When `--prompt all` is used, output is a JSON array of 3 such objects.

### Guard trigger output

When the streaming repetition guard fires, the request is aborted immediately:
```json
{
  "finish_reason": "repetition_guard",
  "guard_triggered": true,
  "guard_reason": "SUFFIX_LOOP: content: 2 reps of 84-char block",
  "completion_tokens": 847
}
```

A config that triggers the guard is **immediately disqualified** — no need to retest.

## Evaluation Criteria

1. `guard_triggered == false` — the streaming guard must NOT trigger (instant fail)
2. `repetition.severity == "none"` (overall < 0.05) at max_tokens <= 4096
3. `repetition.overall < 0.10` at max_tokens > 4096
4. `finish_reason == "stop"` preferred; `"length"` acceptable at high max_tokens
5. If in doubt about quality, re-run with `--save-output` and read the first ~500 chars of the output file
6. A config that triggers the guard is immediately disqualified

## Parameter Ranges & Semantics

| Parameter | Range | Description |
|-----------|-------|-------------|
| temperature | 0.0–2.0 | Higher = more random. 0.6 is conservative. |
| top_p | 0.0–1.0 | Nucleus sampling cutoff. 0.95 is typical. |
| top_k | -1 (disabled), 1–100 | Hard cutoff. 20–40 typical for Qwen. |
| min_p | 0.0–1.0 | Relative probability floor. 0.0 = disabled. |
| presence_penalty | -2.0–2.0 | Additive, penalizes any token already seen. |
| frequency_penalty | -2.0–2.0 | Additive, proportional to occurrence count. |
| repetition_penalty | 0.0 or >=1.0 | Multiplicative (CTRL-style), 1.0 = neutral. 0.0 may be "disabled" in SGLang. |

## Current Baseline

```
temperature=0.6  top_p=0.95  top_k=40  min_p=0.1
presence_penalty=1.0  frequency_penalty=0.0  repetition_penalty=0.0
```

## Strategy

- **Sweep one dimension at a time** (cheapest), then combine. Don't grid-search.
- **Be efficient** — stop exploring a parameter when the trend is clear. If 3 consecutive values show monotonically worse results, stop sweeping that direction.
- **Use the analytical prompt** for single-dimension sweeps (fastest, most stable signal).
- **Cross-validate** with all 3 prompts only for combined validation steps (QT-005+).
- **Progressive scaling**: validate at 4096 first, then step up through 8192/12288/16384.
- Each test at 4096 tokens takes ~2-4 minutes; at 16384 tokens ~8-15 minutes. Budget your iterations.

## Config File (QT-009)

For the final commit, add `sampling_overrides` to the Qwen Thinking model profile in:

```
roles/k8s_dgx/defaults/main.yml
```

Under `sglang_model_profiles` → `"QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ"`, uncomment/update the `sampling_overrides` dict with the optimal values found. Example:

```yaml
    sampling_overrides:
      presence_penalty: 1.0
      temperature: 0.6
      top_p: 0.95
      top_k: 40
      min_p: 0.1
      repetition_penalty: 1.15
      frequency_penalty: 0.3
```

## Commit Convention

```
ops: [QT-XXX] - <short description>
```

Only commit when Ansible files are modified (QT-009). Do not commit prd.json or progress.txt changes alone.

## Environment

- `SGLANG_URL` must be set (the ralph shell script checks this)
- `.venv` is activated by the ralph shell script
- `test_params.py` is at `ops/qwen235b-tuning/test_params.py`
- `kubectl` runs locally with `--context=ht@dgxarley` — do NOT ssh to servers

## CRITICAL: Bash Tool Timeout

The Bash tool has a **hard 10-minute (600000ms) timeout** for foreground calls. Tests at high max_tokens can take 15-30 minutes. You MUST handle this:

- **max_tokens <= 4096**: Foreground is fine. Use `timeout: 600000` explicitly.
  ```
  Bash(command="python ops/qwen235b-tuning/test_params.py --prompt analytical --max-tokens 4096 ...", timeout=600000)
  ```
- **max_tokens > 4096**: Use `run_in_background: true`. The command runs without a wall-clock cap. You will be notified when it completes, then read the output.
  ```
  Bash(command="python ops/qwen235b-tuning/test_params.py --prompt analytical --max-tokens 8192 ...", run_in_background=true)
  ```
  After the background task completes, read its output to get the JSON result.

**Never run max_tokens >= 8192 tests in foreground** — they will timeout and you lose the result.

## Important Notes

- **Be patient**: each test takes 2-15 minutes depending on max_tokens. Don't run unnecessary tests.
- **Record everything**: append all test results to progress.txt with the params used and scores obtained.
- **Format tables**: use markdown tables in progress.txt for sweep results so they're easy to compare.
- **No interactive tools**: do NOT use `sglang-raw` (it's a TUI). Always use `test_params.py`.
- When running bash commands, capture the JSON output and parse it with jq or read it directly.
- If a test errors with a connection error, the SGLang service may be down — note it in progress.txt and retry on next iteration.
