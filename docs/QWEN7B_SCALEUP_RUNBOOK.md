# Qwen 7B Scale-Up Runbook

## Goal

Use `Qwen/Qwen2.5-7B-Instruct` as the **only** next-stage scale-up target and reproduce the minimum set of conclusions from the 3B line:

1. `baseline -> interference` still produces a late-layer collapse pattern.
2. `runtime_output_only_safe` remains the strongest main runtime-safe baseline.
3. `single-head augmentation` is still treated as a secondary check, not a primary method line.

## What To Reuse

Use the existing pipeline as-is:

- [scripts/run_local_probe.py](/Users/shiqi/code/graduation-project/scripts/run_local_probe.py)
- [scripts/run_local_probe_mechanistic_analysis.py](/Users/shiqi/code/graduation-project/scripts/run_local_probe_mechanistic_analysis.py)
- [src/open_model_probe/model_runner.py](/Users/shiqi/code/graduation-project/src/open_model_probe/model_runner.py)
- [src/open_model_probe/mechanistic.py](/Users/shiqi/code/graduation-project/src/open_model_probe/mechanistic.py)
- [src/open_model_probe/internal_signal_predictor.py](/Users/shiqi/code/graduation-project/src/open_model_probe/internal_signal_predictor.py)

## Important Constraints

- Do **not** copy 3B late layers such as `31~35` directly into the 7B interpretation.
- Do **not** reuse 3B fixed heads such as `L34H11` as the 7B single-head target.
- Do **not** start with large patching or multi-model expansion.

## Minimal Execution Order

### 1. Smoke

Run:

```bash
zsh scripts/run_qwen7b_local_probe_smoke.sh
```

This verifies:

- local model loading
- MPS execution
- baseline / interference / recheck outputs
- logits / hidden states / attentions

### 2. Small Probe

Run:

```bash
zsh scripts/run_qwen7b_small_repro.sh
```

This does:

- build a 24-sample subset from the existing 100-sample probe set
- run three-scenario local probe
- run mechanistic summary

### 3. Interpret the 7B Late-Layer Band

After the small mechanistic run finishes:

- inspect the layer summary
- find the strongest late-layer band by relative depth
- identify one 7B-specific candidate head for secondary checking

Only after that should the sparse runtime monitor and single-head validation be adapted for the 7B run.

## New Small Helper

To avoid building a brand-new sample pipeline, use:

- [scripts/build_local_probe_subset.py](/Users/shiqi/code/graduation-project/scripts/build_local_probe_subset.py)

It takes an existing sample set and creates a smaller balanced subset by `sample_type`.

## Expected Output Roots

- smoke:
  - `outputs/experiments/local_probe_qwen7b_smoke/...`
- small probe:
  - `outputs/experiments/local_probe_qwen7b/...`
- mechanistic summary:
  - `outputs/experiments/local_probe_qwen7b_mechanistic/...`

## Recommended First Parameters

- `device=mps`
- `dtype=float32`
- `max_length=512`
- `top_k=8`
- `hidden_state_layers=-1,-2,-3,-4`

For `Qwen/Qwen2.5-7B-Instruct` on Apple Silicon, prefer `dtype=float32` on MPS.
The 3B run was stable with `float16`, but the 7B run produced intermittent `NaN` logits and
final-layer hidden summaries under `mps + float16`. `mps + float32` stayed numerically stable
while preserving the same qualitative smoke behavior.

## What Not To Claim Too Early

Even if 7B reproduces the 3B pattern:

- do not immediately claim stronger trigger utility
- do not treat single-head gain as a primary result
- do not generalize from one 7B run to all larger models

The first goal is only:

**confirm that the 3B main qualitative conclusions are not obviously small-model artifacts.**
