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

## Results — Cognitive Memory

### What this benchmark measures

Each of the 401 test samples contains a **multi-day conversation** (~65,000 characters) between two people. Buried days earlier in the conversation is a specific detail — the **cue** (e.g., "I learned to say 'no' and it reduced my stress"). Later, a **trigger** appears (e.g., "I volunteered for a project and now I'm overwhelmed"). The model must respond in a way that **naturally connects back** to the earlier cue without being told to.

This tests *cognitive memory* — can an AI notice that something said days ago is relevant to what's happening now? Humans do this effortlessly; LLMs often miss the connection when it's buried in tens of thousands of tokens of conversation.

### How scoring works

- An independent judge model (`gemini-2.5-flash`, temperature=0) reads each prediction and decides: does the response connect to the evidence? **correct = 1 point**, **wrong = 0 points**.
- **Baseline** = the model receives the full conversation and responds directly (no help).
- **Prism-MCP** = the same model, but with [Prism](https://github.com/dcostenco/prism-coder)'s semantic memory system — it stores conversation fragments as embeddings in a local database and retrieves the relevant memory when the trigger arrives.

### Results (401 samples)

| Configuration | Score | Accuracy | vs Baseline | Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| Gemini-2.5-flash (Baseline) | 278 / 401 | **69.33%** | — | — |
| Prism-MCP (Gemini-2.5-flash + Memory) | 361 / 401 | **90.02%** | +20.69pp | 67.5% |
| Gemini-3.1-pro-preview (Baseline) | 272 / 401 | **67.83%** | — | — |
| Prism-MCP (Gemini-3.1-pro + Memory) | 382 / 401 | **95.26%** | +27.43pp | 85.3% |
| Gemini-3.5-flash (Baseline) | 237 / 401 | **59.10%** | — | — |
| Prism-MCP (Gemini-3.5-flash + Memory) | 388 / 401 | **96.76%** | +37.66pp | 92.1% |
| Claude Sonnet 4.6 (Baseline) | 290 / 401 | **72.32%** | — | — |
| Prism-MCP (Claude Sonnet 4.6 + Memory) | 357 / 401 | **89.03%** | +16.71pp | 60.4% |

> **How to read the columns:** *Score* is correct answers out of 401. *vs Baseline* is the percentage-point gain from adding Prism memory to that model. *Error Reduction* is how much of the baseline's failure rate was eliminated (e.g., 92.1% means Prism fixed 92 out of every 100 mistakes Gemini 3.5 Flash made alone).

### What the results mean

- **No model aces this without help.** The best raw model (Claude Sonnet 4.6) still misses 28% of memory cues. Cheaper models miss 30–40%.
- **Structured memory changes the game.** Adding Prism-MCP's semantic retrieval lifts every model by 17–38 percentage points. The cheapest model + Prism (96.76%) outperforms the most expensive model alone (72.32%).
- **Bigger models need less help, but still benefit.** Claude starts higher (72%) so Prism's absolute gain is smaller (+17pp) than for Gemini Flash (+38pp), but it still eliminates 60% of Claude's errors.

<details>
<summary>Concrete example</summary>

**Evidence** (buried in a conversation from two weeks ago):
> Caroline: After learning to say 'no', I've felt a lot less stressed overall.
> Melanie: That's a great skill to develop; protecting your time is important.

**Trigger** (in today's conversation):
> Caroline: I ended up volunteering for that project, and now I'm totally overwhelmed.

**Baseline model response** (misses the connection):
> "That sounds tough. Maybe try breaking the project into smaller pieces?"

**Prism-MCP response** (recalls the earlier conversation):
> "Remember how you learned to say 'no' and it reduced your stress? This might be a good time to practice that again — protecting your time is important!"

The baseline model gives generic advice. Prism retrieves the specific earlier conversation about boundaries and weaves it into the response.
</details>

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
