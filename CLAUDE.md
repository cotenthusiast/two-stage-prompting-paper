# two-prompt-research

Research codebase for measuring positional bias in LLM multiple-choice answering and testing whether prompting/debiasing interventions reduce that bias. Based on Zheng et al. (ICLR 2024), "Large Language Models Are Not Robust Multiple Choice Selectors".

**Current paper framing:**

- Primary: robustness / positional-bias robustness
- Secondary: faithfulness of two-stage decomposition
- Accuracy: tertiary/supporting
- Main empirical claim: two-stage prompting does not reliably mitigate MCQ positional bias and reduces end-to-end accuracy across evaluated models/benchmarks.

**Methods:** Direct MCQ baseline, two-stage prompting (v1/v2/v3 prompt ablations), text extraction, semantic match (v1/v2), cyclic permutation, PriDe logprob debiasing, independent hypothesis evaluation.

**Benchmarks:** MMLU, ARC-Challenge.

**Models/providers:**

- OpenAI: `gpt-4.1-mini`
- Google Gemini: `gemini-2.5-flash`
- Groq: `llama-3.1-8b-instant`
- Together AI: `Qwen/Qwen2.5-7B-Instruct-Turbo`
- Local (Kelvin2): `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`

PriDe runs on Together AI and local models (logprob access required). Not available for OpenAI/Gemini/Groq.

---

## How Claude should work in this repository

Claude should behave conservatively. The current priority is paper correctness, reproducibility, and clean extension toward model-generalization experiments.

**Prefer:**

- Small, reviewable diffs
- Preserving existing file structure
- Explaining the intended change before editing
- Adding tests or sanity checks when modifying logic
- Avoiding unnecessary new dependencies and premature abstraction

**Do not:**

- Run expensive API experiments unless explicitly instructed
- Modify `.env` or expose API keys
- Delete cache/checkpoints unless explicitly instructed
- Change benchmark schemas casually or silently rename method/model keys
- Edit `src/twoprompt/config/` for normal run configuration
- Rewrite working modules for style or convert this into a generic framework before the paper is finished
- Prioritize framework refactoring until after the paper is reviewed/submitted

**For important changes, prefer this workflow:**

1. State what the file/change should do.
2. Identify inputs, outputs, and edge cases.
3. Make the smallest change.
4. Explain the change.
5. Run or suggest tests.
6. Avoid unrelated cleanup.

**Rule:** If AI vanished tomorrow, the user should still be able to explain, modify, and continue the project slowly.

---

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env
```

`.env` requires: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`

---

## Running experiments

All run configuration lives in `config/default.yaml` (model list, job matrix, rate limits, temperature/seed, cache settings, prompt version). Do not edit `src/twoprompt/config/` for ordinary experiment configuration.

```bash
# Dry run only, no API calls
python scripts/run_experiment.py --dry-run

# Full run, asks for confirmation
python scripts/run_experiment.py

# Force no cache
python scripts/run_experiment.py --no-cache
```

---

## Evaluating runs

All paper eval data lives in `paper_results/eval_ready/`. Use `--config paper_results/eval_config.yaml` so reports go to `paper_results/reports/` instead of the default `reports/`.

```bash
# Main eval folders (have baseline rows — no --baseline-dir needed)
python scripts/evaluate_run.py paper_api_main --benchmark mmlu --config paper_results/eval_config.yaml
python scripts/evaluate_run.py paper_combined_main --benchmark mmlu --config paper_results/eval_config.yaml

# Ablation folders (no baseline rows — supply --baseline-dir for overlap/choice-shift analysis)
python scripts/evaluate_run.py paper_api_ablation_v2 --benchmark mmlu \
  --baseline-dir paper_api_main --config paper_results/eval_config.yaml
python scripts/evaluate_run.py paper_local_ablation_v3 --benchmark mmlu \
  --baseline-dir paper_local_main --config paper_results/eval_config.yaml
```

Reports written to `paper_results/reports/<run_folder>/<benchmark>/`.

**`--baseline-dir`** — loads baseline rows from another eval_ready folder and merges them in before all computations. Required for ablation folders which contain no baseline rows of their own.

---

## Fallback analysis

Separate diagnostic script — never overwrites main reports.

```bash
# Folder with baseline rows already present
python scripts/fallback_analysis.py --run-dir paper_results/eval_ready/paper_api_main --benchmark mmlu

# Ablation folder — point to a main folder for baseline rows
python scripts/fallback_analysis.py \
  --run-dir paper_results/eval_ready/paper_api_ablation_v2 \
  --baseline-dir paper_results/eval_ready/paper_api_main \
  --benchmark mmlu
```

Output: `paper_results/reports_fallback/<folder>/<benchmark>/fallback_summary.csv` and `fallback_per_question.csv`.

---

## Aggregating paper tables

```bash
python scripts/aggregate_results.py <run_folder> --benchmark mmlu --config paper_results/eval_config.yaml
python scripts/aggregate_results.py <run_folder> --cross-benchmark --config paper_results/eval_config.yaml
```

---

## Project structure

```text
config/
  default.yaml             main run config
  text_extraction.yaml     ablation run config for text_extraction method

scripts/
  run_experiment.py        run API experiments
  evaluate_run.py          compute eval metrics from run CSVs
  aggregate_results.py     build paper tables from eval reports
  visualise_results.py     personal matplotlib plots (not for LaTeX)
  fallback_analysis.py     diagnostic: what if unscorable rows fell back to baseline?
  semantic_matching.py     derive twostage_semantic_match from existing two_prompt runs
  prepare_data.py          data preparation

src/twoprompt/
  clients/
    base.py           openai_client.py   gemini_client.py
    groq_client.py    together_client.py types.py
  runners/
    base.py           direct_mcq.py      permutation.py
    two_stage.py      two_stage_permutation.py
    pride.py          pride_debias.py
  infra/
    cache.py          checkpoint.py
  config/
    experiment.py     models.py          paths.py
  benchmarks/         parsing/           scoring/
  io/                 pipeline/

prompts/v1/   prompts/v2/   prompts/v3/
  direct_mcq.txt    free_text.txt    option_matching.txt

paper_results/
  eval_config.yaml         points runs_dir and reports_dir at paper_results/
  eval_ready/              eval-ready CSVs (gitignored); method_name renamed for ablations
    paper_api_main/        API models, v1 methods
    paper_local_main/      local models, v1 methods
    paper_combined_main/   all 6 models, v1 methods
    paper_api_ablation_v2/ API models, two_prompt_v2 + twostage_semantic_match_v2
    paper_local_ablation_v2/
    paper_api_ablation_v3/ API models, two_prompt_v3
    paper_local_ablation_v3/
  inventories/             coverage notes and run inventory (committed)
  reports/                 generated eval reports (gitignored)
  reports_fallback/        generated fallback analysis (gitignored)

runs/   checkpoints/   .cache/responses/
data/processed/    data/splits/
```

---

## Experiment methods

| Key | Class | Description |
| --- | ----- | ----------- |
| `baseline` | `DirectMCQRunner` | Single prompt, parse first answer letter |
| `two_prompt` | `TwoStageRunner` | Stage 1 free-text answer, Stage 2 letter-extraction matching (v1 prompts) |
| `two_prompt_v2` | `TwoStageRunner` | Two-stage with v2 Stage-1 prompt (ablation; method_name renamed in eval_ready) |
| `two_prompt_v3` | `TwoStageRunner` | Two-stage with v3 Stage-2 prompt (ablation; method_name renamed in eval_ready) |
| `cyclic` | `PermutationRunner` | Four cyclic option rotations with majority vote |
| `two_prompt_cyclic` | `TwoStagePermutationRunner` | Free-text answer plus cyclic option matching (not in current paper runs) |
| `pride` | `PriDeRunner` | Logprob-based positional-prior debiasing |
| `text_extraction` | `TextExtractionRunner` | Stage 1 free-text, Stage 2 deterministic text matching (no second LLM call) |
| `independent_hypothesis` | `IndependentHypothesisRunner` | Four independent calls per question, one per option framed as a hypothesis; argmax of regex-parsed confidence scores, seeded random tie-break |
| `twostage_semantic_match` | (post-hoc) | Derived from two_prompt Stage 1 outputs via sentence-transformer cosine similarity |
| `twostage_semantic_match_v2` | (post-hoc) | Same derivation from two_prompt_v2 Stage 1 outputs |

`EXTERNALLY_SCORED_METHODS` in `experiment.py` — pride, text_extraction, twostage_semantic_match, twostage_semantic_match_v2. These must never be reparsed by `evaluate_run.py`.

---

## Evaluation outputs

Per benchmark, written to `paper_results/reports/<run_folder>/<benchmark>/` (with `--config`):

- `accuracy.csv` — end-to-end and conditional accuracy with 95% CIs
- `positional_bias.csv` — MAD from ground-truth answer distribution
- `overlap.csv` — question-level overlap vs baseline (requires baseline rows present)
- `choice_shifts.csv` — per-question answer change direction vs baseline
- `subject_accuracy.csv` — per-subject breakdown
- `two_stage_metrics.csv` — free-text availability and latency (all free-text-stage methods)
- `free_text_decomposition.csv` — 2×2 stage decomposition; has `method` column (per-method rows)

Fallback outputs in `paper_results/reports_fallback/<folder>/<benchmark>/`:

- `fallback_summary.csv` — per (model, method) fallback impact
- `fallback_per_question.csv` — per question detail

**Interpretation rules:**

- End-to-end accuracy counts unscorable outputs as incorrect; conditional uses only scorable outputs.
- Conditional accuracy alone can hide matching-stage failures.
- MAD lower means less positional bias.
- Do not overclaim statistical significance unless explicitly computed.

---

## Current main empirical findings

- On MMLU and ARC-Challenge, two-stage prompting reduces end-to-end accuracy for every model.
- On MMLU, two-stage prompting increases MAD point estimate for every model.
- On ARC-Challenge, MAD results are mixed; two-stage does not show consistent bias reduction.
- Gemini 2.5 Flash suffers substantial parse failures under two-stage prompting, especially on MMLU.
- GPT-4.1 mini shows two-stage can harm accuracy even without parse failures.
- Cyclic permutation is the strongest model-agnostic intervention overall.
- PriDe gives limited positive results for Qwen/Together but requires logprob access so is not broadly model-agnostic.

The accurate claim is: **two-stage prompting fails to reliably mitigate MCQ positional bias and reduces end-to-end accuracy across all evaluated model/benchmark settings.** Do not rewrite this as "two-stage always increases bias everywhere."

---

## PriDe implementation notes

- Together AI returns logprobs in a non-standard parallel-array format; `together_client.py` handles both formats.
- Assistant prefill `"The answer is "` is injected when `request_logprobs=True` (practical deviation from the paper).
- `top_logprobs=20` is requested; missing letters get `_LOGPROB_FLOOR = -30.0`.
- Calibration sidecars saved to `runs/<run_id>/pride_calibration__<model>__<benchmark>.json`.
- If stale cache entries with empty logprobs exist from broken runs: `grep -rl '"logprobs": \[\]' .cache/responses/ | xargs rm`
- Do not delete cache broadly unless explicitly asked.

---

## Gotchas

- **Exact model names matter.** `MODEL_ORDER` in `evaluate_run.py` and `aggregate_results.py` must match exact strings in run CSVs. `Qwen/Qwen2.5-7B-Instruct-Turbo` ≠ `Qwen/Qwen2.5-7B-Instruct` — a mismatch silently drops model rows.
- **ARC alias:** `--benchmark arc` aliases `arc_challenge`.
- **Cache keys include** `request_logprobs`, so logprob and non-logprob requests are cached separately.
- **PriDe reruns:** if the run CSV is deleted, delete the matching calibration sidecar too — they cannot be rerun independently.
- **Prompt snapshots** are versioned under `prompts/<version>/` and may be copied into run folders; preserve for reproducibility.

---

## Coding style

Use simple, explicit Python. **Prefer:** dataclasses/typed dicts, clear function boundaries, readable loops, explicit error handling, deterministic seeds, stable CSV schemas, small helpers, tests for parsing/scoring changes. **Avoid:** hidden global state, broad exception swallowing, changing CSV columns or method/model keys without migration, heavy dependencies, complex framework machinery.

---

## Testing and sanity checks

```bash
pytest
python scripts/run_experiment.py --dry-run
python scripts/evaluate_run.py paper_api_main --benchmark mmlu --config paper_results/eval_config.yaml
python scripts/aggregate_results.py paper_api_main --benchmark mmlu --config paper_results/eval_config.yaml
```

When modifying **parsing/scoring**: test A/B/C/D extraction, unscorable outputs, lowercase/verbose outputs, outputs with explanations, deterministic scoring.

When modifying **PriDe**: test logprob extraction, missing-letter floor, calibration state load/save, Eq.(7)/Eq.(8) separately.

---

## Future direction: model generalization

The next research direction is likely: do MCQ bias-mitigation methods validated in limited model settings preserve their effects across larger open-source models?

Do not implement this inside the current codebase until the design is explicitly chosen. Reversible prep is allowed (literature table, candidate-method table, Kelvin2/SLURM learning, small toy HPC jobs). Do not prematurely restructure this repository into a generic MCQ framework.

---

## Kelvin2 / HPC caution

- Do not run heavy jobs on the login node — use SLURM batch jobs.
- Start with tiny test jobs; log stdout/stderr clearly.
- Do not assume local laptop paths work on HPC.
- Keep environment setup documented.
- Mental model: login node = prepare/submit; compute node = run workload; SLURM = scheduler.

---

## Immediate priorities

- Run the full paper eval suite on `paper_results/eval_ready/` and verify all reports.
- Run fallback analysis on main and ablation folders for the diagnostic table.
- Overhaul the paper itself based on the full 6-model, 9-method results.
- Support supervisor-review changes.
- Model-generalization direction: do not implement until paper draft is stable.
