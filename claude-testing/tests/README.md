# OpenMDAO Skills A/B Testing Framework

This framework lets you objectively measure whether your Claude skills improve answers compared to base Claude.

## What It Does

For each test prompt, it:
1. Asks the prompt to base Claude (no skills loaded)
2. Asks the same prompt to Claude with the OpenMDAO skills loaded
3. Scores both responses against a rubric
4. Reports per-category lift to show where skills help most

## Setup

Install the Anthropic Python SDK:

```bash
pip install anthropic
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Files

| File | Purpose |
|------|---------|
| `ab_test_prompts.json` | The test prompts and scoring rubrics |
| `run_ab_test.py` | Runs each prompt against base Claude and skills-loaded Claude |
| `score_responses.py` | Scores responses (LLM-as-judge or manual) |
| `analyze_results.py` | Aggregates scores and produces a report |
| `results/` | Output folder for raw results and scores |

## Workflow

### Step 1: Run the test

```bash
python tests/run_ab_test.py
```

This produces `tests/results/raw_results_TIMESTAMP.json` with both responses for each prompt.

### Step 2: Score the responses

**Option A: Automated scoring with LLM-as-judge (fast)**

```bash
python tests/score_responses.py tests/results/raw_results_TIMESTAMP.json --mode llm
```

**Option B: Manual scoring by an expert (rigorous)**

```bash
python tests/score_responses.py tests/results/raw_results_TIMESTAMP.json --mode manual
```

This generates a blinded markdown scoring sheet. Have an OpenMDAO expert fill in the blanks, then process the results.

### Step 3: Analyze

```bash
python tests/analyze_results.py tests/results/scored_raw_results_TIMESTAMP.json
```

Output looks like:

```
Category                            No Skills    With Skills      Lift
--------------------------------------------------------------------------------
anti_pattern_detection                  35.0%         91.0%    +160.0%
basic_concepts                          85.0%         88.0%      +3.5%
code_generation_constrained             58.0%         85.0%     +46.6%
code_generation_simple                  92.0%         94.0%      +2.2%
convergence_debugging                   55.0%         85.0%     +54.5%
project_convention                      40.0%         88.0%    +120.0%
recent_api                              60.0%         80.0%     +33.3%
--------------------------------------------------------------------------------
OVERALL                                 61.2%         87.7%     +43.3%
```

## Interpreting Results

- **High lift (>30%)** in a category means skills add real value there. Invest more.
- **Low lift (<10%)** means Claude already knows this well. Skills add little.
- **Negative lift** is a red flag — investigate skill content for confusion.

## Recommended Cadence

Re-run this test:
- After significant skill content changes
- After OpenMDAO releases a new version (catch API drift)
- After Claude releases a new model
- Before deciding whether to expand the skill set further

## Adding More Prompts

Edit `ab_test_prompts.json` and add new entries. Aim for:
- 30-50 prompts total for statistical meaningfulness
- Even coverage across categories
- Mix of basic, intermediate, and edge-case prompts
- Anti-pattern prompts where skills should clearly outperform
