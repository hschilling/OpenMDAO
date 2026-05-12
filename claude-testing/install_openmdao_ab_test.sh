#!/bin/bash
# install_openmdao_ab_test.sh
# Creates a complete A/B testing framework for OpenMDAO Claude skills.
# Usage: bash install_openmdao_ab_test.sh

set -e

TEST_DIR="tests"

echo "Creating directory: $TEST_DIR"
mkdir -p "$TEST_DIR/results"

# ============================================================================
# tests/ab_test_prompts.json
# ============================================================================
cat << 'TEST_EOF' > "$TEST_DIR/ab_test_prompts.json"
{
  "prompts": [
    {
      "id": "basic-001",
      "category": "basic_concepts",
      "prompt": "What is OpenMDAO?",
      "rubric": {
        "mentions_dataflow_graph_or_similar": 1,
        "lists_5_building_blocks": 2,
        "distinguishes_from_procedural": 1,
        "uses_circuit_board_or_similar_metaphor": 1,
        "max_score": 5
      }
    },
    {
      "id": "basic-002",
      "category": "basic_concepts",
      "prompt": "What is the difference between an ExplicitComponent and an ImplicitComponent?",
      "rubric": {
        "explicit_outputs_directly_computed": 2,
        "implicit_outputs_satisfy_residual": 2,
        "mentions_when_to_use_each": 2,
        "max_score": 6
      }
    },
    {
      "id": "antipattern-001",
      "category": "anti_pattern_detection",
      "prompt": "I have two components. Component A has output 'temperature' and Component B has input 'temperature'. Both use promotes=['*']. Will they connect correctly, and is this a good practice?",
      "rubric": {
        "warns_about_silent_collision": 3,
        "recommends_explicit_promotes_lists": 2,
        "suggests_list_connections_verification": 2,
        "explains_why_dangerous": 2,
        "max_score": 9
      }
    },
    {
      "id": "antipattern-002",
      "category": "anti_pattern_detection",
      "prompt": "Should I put my entire aerodynamics analysis in one ExplicitComponent? It is about 500 lines of code.",
      "rubric": {
        "recommends_decomposition": 3,
        "explains_benefits_of_smaller_components": 2,
        "suggests_grouping_related_components": 2,
        "max_score": 7
      }
    },
    {
      "id": "antipattern-003",
      "category": "anti_pattern_detection",
      "prompt": "I do not need analytic derivatives because I am only running the model once, not optimizing. Can I skip declaring partials entirely?",
      "rubric": {
        "advises_against_skipping_setup_partials": 2,
        "recommends_method_fd_as_starting_point": 2,
        "mentions_future_optimization_compatibility": 2,
        "max_score": 6
      }
    },
    {
      "id": "convergence-001",
      "category": "convergence_debugging",
      "prompt": "My solver says 'NL: NLBGS Failed to Converge in 100 iterations'. What do I do?",
      "rubric": {
        "suggests_better_initial_values": 1,
        "suggests_increase_maxiter": 1,
        "suggests_relax_tolerance": 1,
        "suggests_try_newton_solver": 2,
        "suggests_check_model_correctness": 1,
        "presents_in_order_of_complexity": 1,
        "max_score": 7
      }
    },
    {
      "id": "convergence-002",
      "category": "convergence_debugging",
      "prompt": "My optimizer is taking 500 iterations and not converging. My design variable ranges from 0 to 100,000.",
      "rubric": {
        "identifies_scaling_as_likely_cause": 3,
        "recommends_ref_value_on_design_var": 2,
        "explains_order_1_rule": 2,
        "shows_correct_add_design_var_with_ref": 1,
        "max_score": 8
      }
    },
    {
      "id": "code-gen-001",
      "category": "code_generation_simple",
      "prompt": "Create a paraboloid component.",
      "rubric": {
        "uses_om_alias": 1,
        "inherits_explicit_component": 1,
        "has_setup_method": 1,
        "has_setup_partials_method": 1,
        "has_compute_method": 1,
        "correct_paraboloid_formula": 1,
        "max_score": 6
      }
    },
    {
      "id": "code-gen-002",
      "category": "code_generation_constrained",
      "prompt": "Create a component for lift force using L = 0.5 * rho * V^2 * S * Cl. Make sure I can use this in optimization later.",
      "rubric": {
        "includes_units_on_physical_quantities": 2,
        "uses_descriptive_variable_names": 1,
        "includes_setup_partials": 2,
        "mentions_analytic_derivatives_for_optimization": 2,
        "warns_about_fd_for_optimization": 1,
        "max_score": 8
      }
    },
    {
      "id": "code-gen-003",
      "category": "code_generation_constrained",
      "prompt": "Write an ImplicitComponent that solves x^2 = a for x.",
      "rubric": {
        "uses_apply_nonlinear_for_residual": 2,
        "residual_writes_to_residuals_not_outputs": 2,
        "includes_jacobian_x_x_partial": 3,
        "uses_linearize_method": 1,
        "mentions_solver_needed_at_group_level": 2,
        "max_score": 10
      }
    },
    {
      "id": "verification-001",
      "category": "project_convention",
      "prompt": "I added analytic derivatives to all my components. check_partials passed. Should I optimize now?",
      "rubric": {
        "recommends_check_totals_first": 3,
        "explains_partials_vs_totals_difference": 2,
        "shows_verification_chain": 2,
        "max_score": 7
      }
    },
    {
      "id": "verification-002",
      "category": "project_convention",
      "prompt": "What is the recommended workflow for adding analytic derivatives to a component?",
      "rubric": {
        "recommends_starting_with_method_fd": 2,
        "shows_compute_partials_replacement": 2,
        "recommends_check_partials_with_cs_method": 2,
        "mentions_force_alloc_complex": 1,
        "max_score": 7
      }
    },
    {
      "id": "api-modern-001",
      "category": "recent_api",
      "prompt": "How do I declare a design variable for optimization in OpenMDAO?",
      "rubric": {
        "uses_modern_set_val_approach": 2,
        "does_not_unnecessarily_use_indepvarcomp": 2,
        "shows_add_design_var_directly": 1,
        "max_score": 5
      }
    },
    {
      "id": "api-modern-002",
      "category": "recent_api",
      "prompt": "Show me how to set input values before running my model.",
      "rubric": {
        "uses_prob_set_val": 2,
        "shows_units_argument_optional": 1,
        "no_unnecessary_indepvarcomp": 1,
        "max_score": 4
      }
    },
    {
      "id": "debug-001",
      "category": "convergence_debugging",
      "prompt": "I am getting wrong results from my model. The component math looks correct. How do I debug?",
      "rubric": {
        "recommends_n2_diagram": 2,
        "recommends_list_connections": 2,
        "recommends_list_inputs_outputs": 1,
        "suggests_checking_unconnected_inputs": 2,
        "max_score": 7
      }
    }
  ]
}
TEST_EOF

# ============================================================================
# tests/run_ab_test.py
# ============================================================================
cat << 'TEST_EOF' > "$TEST_DIR/run_ab_test.py"
"""
A/B test runner: Compare base Claude vs Claude with OpenMDAO skills.

Usage:
    python tests/run_ab_test.py

Requires:
    pip install anthropic
    Environment variable: ANTHROPIC_API_KEY
"""
import anthropic
import json
import os
import time
from pathlib import Path
from datetime import datetime

# Configuration
MODEL = "claude-sonnet-4-5"
SKILLS_DIRS = [
    '.claude/skills/01-beginner-tasks',
    '.claude/skills/02-intermediate-tasks',
]
PROMPTS_FILE = "tests/ab_test_prompts.json"
RESULTS_DIR = "tests/results"
SLEEP_BETWEEN_CALLS = 1.0  # Rate limit safety

client = anthropic.Anthropic()


def load_skills():
    """Concatenate all skill markdown files into a single context block."""
    skills_text = ""
    for skills_dir in SKILLS_DIRS:
        path = Path(skills_dir)
        if not path.exists():
            print(f"WARNING: skills directory not found: {skills_dir}")
            continue
        for f in sorted(path.glob('*.md')):
            skills_text += f"\n\n--- {f.name} ---\n\n" + f.read_text()
    if not skills_text:
        raise RuntimeError("No skill files found. Check SKILLS_DIRS paths.")
    return skills_text


def query_claude(prompt, system_prompt):
    """Send a prompt to Claude with the given system prompt."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def run_test():
    print(f"Loading skills from {SKILLS_DIRS}")
    skills = load_skills()
    print(f"Loaded {len(skills)} characters of skill content")

    print(f"Loading prompts from {PROMPTS_FILE}")
    prompts = json.loads(Path(PROMPTS_FILE).read_text())['prompts']
    print(f"Found {len(prompts)} test prompts")

    base_system = "You are an OpenMDAO assistant. Help users with OpenMDAO questions."
    skills_system = base_system + "\n\nUse the following knowledge to help users:\n\n" + skills

    results = []
    for i, item in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {item['id']}: {item['prompt'][:60]}...")

        try:
            print("  Querying base Claude (no skills)...")
            response_a = query_claude(item['prompt'], system_prompt=base_system)
            time.sleep(SLEEP_BETWEEN_CALLS)

            print("  Querying Claude with skills...")
            response_b = query_claude(item['prompt'], system_prompt=skills_system)
            time.sleep(SLEEP_BETWEEN_CALLS)

            results.append({
                "id": item['id'],
                "category": item['category'],
                "prompt": item['prompt'],
                "rubric": item['rubric'],
                "response_a_no_skills": response_a,
                "response_b_with_skills": response_b
            })
        except Exception as e:
            print(f"  ERROR on {item['id']}: {e}")
            results.append({
                "id": item['id'],
                "category": item['category'],
                "prompt": item['prompt'],
                "rubric": item['rubric'],
                "error": str(e)
            })

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(RESULTS_DIR) / f"raw_results_{timestamp}.json"
    output_file.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Raw results saved to: {output_file}")
    print(f"Next step: python tests/score_responses.py {output_file}")


if __name__ == "__main__":
    run_test()
TEST_EOF

# ============================================================================
# tests/score_responses.py
# ============================================================================
cat << 'TEST_EOF' > "$TEST_DIR/score_responses.py"
"""
Score A/B test responses using either LLM-as-judge or manual scoring.

Usage:
    # LLM-as-judge (automated):
    python tests/score_responses.py tests/results/raw_results_TIMESTAMP.json --mode llm

    # Generate markdown for manual scoring:
    python tests/score_responses.py tests/results/raw_results_TIMESTAMP.json --mode manual
"""
import anthropic
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

MODEL = "claude-sonnet-4-5"
SLEEP_BETWEEN_CALLS = 1.0

client = anthropic.Anthropic()


def llm_score(response, rubric, prompt):
    """Use Claude to score a response against the rubric."""
    rubric_no_max = {k: v for k, v in rubric.items() if k != 'max_score'}

    judge_prompt = f"""You are evaluating an OpenMDAO assistant's response against a strict rubric.

USER PROMPT:
{prompt}

RUBRIC CRITERIA (each criterion is worth the points shown):
{json.dumps(rubric_no_max, indent=2)}

ASSISTANT RESPONSE:
{response}

For each rubric criterion, decide if the response meets that criterion.
- If the criterion is fully met: award the full points shown.
- If the criterion is not met or only partially addressed: award 0 points.

Return your scoring as JSON only, no other text. Use exactly the criterion names from the rubric.

Example format:
{{"criterion_name_1": 2, "criterion_name_2": 0, ...}}
"""
    result = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    text = result.content[0].text.strip()
    # Strip markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text)


def score_with_llm(results_file):
    print(f"Loading results from {results_file}")
    results = json.loads(Path(results_file).read_text())

    scored = []
    for i, item in enumerate(results):
        print(f"\n[{i+1}/{len(results)}] Scoring {item['id']}...")

        if 'error' in item:
            print(f"  Skipping (had error): {item['error']}")
            continue

        try:
            print("  Scoring response A (no skills)...")
            scores_a = llm_score(item['response_a_no_skills'], item['rubric'], item['prompt'])
            time.sleep(SLEEP_BETWEEN_CALLS)

            print("  Scoring response B (with skills)...")
            scores_b = llm_score(item['response_b_with_skills'], item['rubric'], item['prompt'])
            time.sleep(SLEEP_BETWEEN_CALLS)

            total_a = sum(scores_a.values())
            total_b = sum(scores_b.values())

            scored.append({
                "id": item['id'],
                "category": item['category'],
                "prompt": item['prompt'],
                "rubric": item['rubric'],
                "scores_a": scores_a,
                "scores_b": scores_b,
                "score_a": total_a,
                "score_b": total_b,
                "max_score": item['rubric']['max_score']
            })
            print(f"  Score A: {total_a}/{item['rubric']['max_score']}, Score B: {total_b}/{item['rubric']['max_score']}")
        except Exception as e:
            print(f"  ERROR scoring {item['id']}: {e}")

    output_file = Path(results_file).parent / f"scored_{Path(results_file).stem}.json"
    output_file.write_text(json.dumps(scored, indent=2))
    print(f"\nScored results saved to: {output_file}")
    print(f"Next step: python tests/analyze_results.py {output_file}")


def generate_manual_scoring_sheet(results_file):
    """Generate a blinded markdown file for manual scoring."""
    results = json.loads(Path(results_file).read_text())

    rows = []
    for item in results:
        if 'error' in item:
            continue
        # Randomize order
        if random.random() < 0.5:
            order = ('a', 'b')
            first, second = item['response_a_no_skills'], item['response_b_with_skills']
        else:
            order = ('b', 'a')
            first, second = item['response_b_with_skills'], item['response_a_no_skills']

        rows.append({
            "id": item['id'],
            "category": item['category'],
            "prompt": item['prompt'],
            "rubric": item['rubric'],
            "response_1": first,
            "response_2": second,
            "true_order": order
        })

    md = "# A/B Test Manual Scoring Sheet\n\n"
    md += "Score each response against the rubric. Award full points or 0 for each criterion.\n\n"
    md += "**DO NOT scroll to the bottom until you have finished scoring** (the answer key is there).\n\n"
    md += "---\n\n"

    for r in rows:
        md += f"## {r['id']} (category: {r['category']})\n\n"
        md += f"**Prompt:** {r['prompt']}\n\n"
        md += f"**Rubric (max {r['rubric']['max_score']} points):**\n"
        for k, v in r['rubric'].items():
            if k != 'max_score':
                md += f"- [ ] {k} ({v} pts)\n"
        md += f"\n### Response 1\n\n{r['response_1']}\n\n"
        md += f"**Score Response 1: ___ / {r['rubric']['max_score']}**\n\n"
        md += f"### Response 2\n\n{r['response_2']}\n\n"
        md += f"**Score Response 2: ___ / {r['rubric']['max_score']}**\n\n"
        md += "---\n\n"

    md += "\n\n# ANSWER KEY (do not look until done scoring!)\n\n"
    md += "| ID | Response 1 was | Response 2 was |\n"
    md += "|----|---------------|----------------|\n"
    for r in rows:
        labels = {'a': 'No Skills', 'b': 'With Skills'}
        md += f"| {r['id']} | {labels[r['true_order'][0]]} | {labels[r['true_order'][1]]} |\n"

    output_file = Path(results_file).parent / f"manual_scoring_{Path(results_file).stem}.md"
    output_file.write_text(md)

    # Save the order key separately for later automated analysis
    key_file = Path(results_file).parent / f"manual_key_{Path(results_file).stem}.json"
    key_file.write_text(json.dumps([{"id": r['id'], "true_order": r['true_order']} for r in rows], indent=2))

    print(f"Manual scoring sheet: {output_file}")
    print(f"Order key (private): {key_file}")
    print("\nProcess:")
    print("1. Open the markdown file and score each response (fill in the blanks)")
    print("2. Save the scored markdown file")
    print("3. Use the order key to map your scores back to A vs B")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Path to raw_results_*.json file")
    parser.add_argument("--mode", choices=['llm', 'manual'], default='llm',
                        help="Scoring mode: llm (automated) or manual (generate sheet)")
    args = parser.parse_args()

    if args.mode == 'llm':
        score_with_llm(args.results_file)
    else:
        generate_manual_scoring_sheet(args.results_file)
TEST_EOF

# ============================================================================
# tests/analyze_results.py
# ============================================================================
cat << 'TEST_EOF' > "$TEST_DIR/analyze_results.py"
"""
Analyze scored A/B test results.

Usage:
    python tests/analyze_results.py tests/results/scored_raw_results_TIMESTAMP.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def analyze(scored_file):
    data = json.loads(Path(scored_file).read_text())

    if not data:
        print("No scored results found.")
        return

    by_category = defaultdict(lambda: {'a': [], 'b': []})

    for item in data:
        cat = item['category']
        max_s = item['max_score']
        a_pct = item['score_a'] / max_s
        b_pct = item['score_b'] / max_s
        by_category[cat]['a'].append(a_pct)
        by_category[cat]['b'].append(b_pct)

    print("\n" + "=" * 80)
    print("OPENMDAO SKILLS A/B TEST RESULTS")
    print("=" * 80 + "\n")

    print(f"{'Category':<35} {'No Skills':>12} {'With Skills':>14} {'Lift':>10}")
    print("-" * 80)

    for cat in sorted(by_category.keys()):
        scores = by_category[cat]
        a_avg = mean(scores['a'])
        b_avg = mean(scores['b'])
        lift = (b_avg - a_avg) / a_avg * 100 if a_avg > 0 else float('inf')
        n = len(scores['a'])
        print(f"{cat:<35} {a_avg:>11.1%} {b_avg:>13.1%} {lift:>+9.1f}% (n={n})")

    all_a = [s for cat in by_category.values() for s in cat['a']]
    all_b = [s for cat in by_category.values() for s in cat['b']]
    overall_lift = (mean(all_b) - mean(all_a)) / mean(all_a) * 100 if mean(all_a) > 0 else float('inf')
    print("-" * 80)
    print(f"{'OVERALL':<35} {mean(all_a):>11.1%} {mean(all_b):>13.1%} {overall_lift:>+9.1f}% (n={len(all_a)})")
    print()

    # Per-prompt detail
    print("=" * 80)
    print("PER-PROMPT DETAIL")
    print("=" * 80 + "\n")
    print(f"{'ID':<25} {'Category':<30} {'A':>6} {'B':>6} {'Diff':>6}")
    print("-" * 80)
    for item in data:
        diff = item['score_b'] - item['score_a']
        print(f"{item['id']:<25} {item['category']:<30} {item['score_a']:>5}/{item['max_score']} {item['score_b']:>5}/{item['max_score']} {diff:>+6}")

    # Highlight biggest wins and losses
    print("\n" + "=" * 80)
    print("BIGGEST IMPACT PROMPTS")
    print("=" * 80 + "\n")
    sorted_by_diff = sorted(data, key=lambda x: x['score_b'] - x['score_a'], reverse=True)

    print("Top 3 wins for skills:")
    for item in sorted_by_diff[:3]:
        diff = item['score_b'] - item['score_a']
        print(f"  +{diff} points: {item['id']} - {item['prompt'][:70]}...")

    print("\nTop 3 losses (or non-wins) for skills:")
    for item in sorted_by_diff[-3:]:
        diff = item['score_b'] - item['score_a']
        print(f"  {diff:+} points: {item['id']} - {item['prompt'][:70]}...")

    # Recommendations
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80 + "\n")

    if overall_lift > 30:
        print(f"Skills provide STRONG overall lift ({overall_lift:+.1f}%). Continue investing in them.")
    elif overall_lift > 10:
        print(f"Skills provide MODERATE overall lift ({overall_lift:+.1f}%). Worth maintaining.")
    elif overall_lift > 0:
        print(f"Skills provide MINIMAL overall lift ({overall_lift:+.1f}%). Consider focusing on high-impact areas only.")
    else:
        print(f"Skills do NOT improve responses overall ({overall_lift:+.1f}%). Investigate whether skills are being loaded correctly.")

    # Per-category recommendations
    print("\nCategory-by-category guidance:")
    for cat, scores in sorted(by_category.items()):
        cat_lift = (mean(scores['b']) - mean(scores['a'])) / mean(scores['a']) * 100 if mean(scores['a']) > 0 else float('inf')
        if cat_lift > 30:
            print(f"  {cat}: HIGH value from skills ({cat_lift:+.1f}%) - prioritize this area")
        elif cat_lift > 10:
            print(f"  {cat}: Moderate value ({cat_lift:+.1f}%)")
        elif cat_lift > 0:
            print(f"  {cat}: Marginal value ({cat_lift:+.1f}%)")
        else:
            print(f"  {cat}: Skills don't help here ({cat_lift:+.1f}%) - Claude already knows this well")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scored_file", help="Path to scored_*.json file")
    args = parser.parse_args()
    analyze(args.scored_file)
TEST_EOF

# ============================================================================
# tests/README.md
# ============================================================================
cat << 'TEST_EOF' > "$TEST_DIR/README.md"
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
TEST_EOF

echo ""
echo "Done! A/B testing framework installed in: $TEST_DIR"
echo ""
echo "Files created:"
ls -1 "$TEST_DIR"
echo ""
echo "Next steps:"
echo "  1. Install dependencies: pip install anthropic"
echo "  2. Set your API key: export ANTHROPIC_API_KEY='your-key-here'"
echo "  3. Run the test: python tests/run_ab_test.py"
echo "  4. Score responses: python tests/score_responses.py tests/results/raw_results_TIMESTAMP.json --mode llm"
echo "  5. Analyze: python tests/analyze_results.py tests/results/scored_raw_results_TIMESTAMP.json"
echo ""
echo "See tests/README.md for full documentation."

