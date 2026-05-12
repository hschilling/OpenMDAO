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
