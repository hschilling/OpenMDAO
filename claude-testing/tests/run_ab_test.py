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
    '../.claude/skills/01-beginner-tasks',
    '../.claude/skills/02-intermediate-tasks',
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
