# Locomo-Plus

**Beyond-factual cognitive memory evaluation for LLM agents.**  
This repository contains the code and data pipeline for the paper:

> **Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents**  
> ARR 2026 (January)

Locomo-Plus extends the [LoCoMo](https://github.com/snap-research/LoCoMo) dialogue benchmark with a sixth task category **Cognitive**, which evaluates long-context memory and implicit recall: whether a model can connect a later *trigger query* to an earlier *cue dialogue* in multi-session conversations.

---

## What’s in this repo

- **Data pipeline** — Generate cue dialogues and trigger queries (with optional human filtering and similarity-based ranking), then build unified inputs by stitching cue/query into LoCoMo conversations.
- **Unified evaluation** — Six categories in one format: LoCoMo’s original five (multi-hop, temporal, common-sense, single-hop, adversarial) plus Cognitive. Run model predictions and score with an LLM-as-judge (correct=1, partial=0.5, wrong=0).

## Repository layout

| Directory | Contents |
|-----------|----------|
| `data/` | `build_conv.py`, `unified_input.py`; LoCoMo-Plus samples (`locomo_plus.json`) and LoCoMo conversations (`locomo10.json`). See [data/README.md](data/README.md). |
| `generation_pipeline/` | Cue dialogue generation, trigger query generation, and similarity ranking. Steps 2 and 5 are manual. See [generation_pipeline/README.md](generation_pipeline/README.md). |
| `evaluation_framework/` | Scripts and code to run models on the unified dataset and to run the LLM-as-judge. |

## Requirements

- **Generation**: `openai`, `tqdm`; for ranking, `rank_bm25`, `numpy`, `sentence-transformers`.
- **Evaluation**: Python 3; for API-backed evaluation, `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`), `GOOGLE_API_KEY` for Gemini, `ANTHROPIC_API_KEY` for Claude.

All API keys and paths are configured via environment variables or local config files (no secrets in the repo).

## Results — Cognitive Memory (401 samples, `gemini-2.5-flash` judge)

| Configuration | Total Score | Average Score | Absolute Delta | Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| Gemini-2.5-flash (Baseline) | 278.0 / 401 | **69.33%** | — | — |
| Prism-MCP (Gemini-2.5-flash + Memory) | 361.0 / 401 | **90.02%** | +20.69pp | 67.5% |
| Gemini-3.1-pro-preview (Baseline) | 272.0 / 401 | **67.83%** | — | — |
| Prism-MCP (Gemini-3.1-pro + Memory) | 382.0 / 401 | **95.26%** | +27.43pp | 85.3% |
| Gemini-3.5-flash (Baseline) | 237.0 / 401 | **59.10%** | — | — |
| Prism-MCP (Gemini-3.5-flash + Memory) | 388.0 / 401 | **96.76%** | +37.66pp | 92.1% |
| Claude Sonnet 4.6 (Baseline) | 290.0 / 401 | **72.32%** | — | — |
| Prism-MCP (Claude Sonnet 4.6 + Memory) | 357.0 / 401 | **89.03%** | +16.71pp | 60.4% |

## Quick start

**1. Build unified input**  
Put `locomo10.json` and `locomo_plus.json` in `data/`, then:

```bash
cd data && python unified_input.py
```

This writes a unified JSON (e.g. `unified_input_samples_v2.json`). Large generated JSONs are gitignored; run the script locally to produce them.

**2. Run evaluation**  
Set `DATA_FILE_PATH` in `evaluation_framework/scripts/env.sh` to your unified JSON. Copy `evaluation_framework/scripts/env.local.sh.example` to `env.local.sh` and set your API keys. Then:

```bash
./evaluation_framework/scripts/evaluate.sh gpt-4o-mini call_llm 0.3 4
./evaluation_framework/scripts/judge.sh output/unified_predictions.json output/judged.json gpt-4o-mini 4 output/judge_summary.json
```

**3. (Optional) Reproduce data generation**  
See [generation_pipeline/README.md](generation_pipeline/README.md) for cue dialogue generation, human filtering, trigger query generation, ranking, and final validation to produce `locomo_plus.json`.

## Configuration

| What | How |
|------|-----|
| Generation API | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` |
| Gemini API | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Claude API | `ANTHROPIC_API_KEY` |
| Filtered cues for trigger generation | `CUE_QUERY_INPUT` (default: `generation_pipeline/selected_cue_query.json`) |
| Full cue–query JSON for ranking | `RANK_INPUT` (default: `complete_data_all_models.json` in script dir) |
| Embedding models for ranking | `SENTENCE_TRANSFORMER_MPNET`, `SENTENCE_TRANSFORMER_BGE` (defaults: HuggingFace IDs) |
| Unified input for evaluation | `DATA_FILE_PATH` in `evaluation_framework/scripts/env.sh` |
| Judge / model API | `env.local.sh`: `OPENAI_BASE_URL`, `OPENAI_API_KEY` |

## Citation

If you use Locomo-Plus in your work, please cite:

```bibtex
@misc{li2026locomoplusbeyondfactualcognitivememory,
      title={Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents}, 
      author={Yifei Li and Weidong Guo and Lingling Zhang and Rongman Xu and Muye Huang and Hui Liu and Lijiao Xu and Yu Xu and Jun Liu},
      year={2026},
      eprint={2602.10715},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.10715}, 
}

```


## License

See the repository for license information. API keys and paths are configured locally (e.g. via `env.local.sh` or environment variables); the repo ships no credentials.
