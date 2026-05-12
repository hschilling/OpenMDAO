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
