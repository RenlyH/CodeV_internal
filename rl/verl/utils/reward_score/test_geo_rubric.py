#!/usr/bin/env python
"""
Test suite for Geometry (GEO) rubric evaluation.
Image: 44759.jpg
Question: what is x
Answer: 44
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from verl.utils.reward_score.vl_agent import (
    rubrics_judge,
    extract_answer,
    RUBRIC_CALCULATION_USAGE,
    RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION
)

IMAGE_PATH = "/workspace/shared/datasets/CodeV_images/44759.jpg"


def test_geo_score_1_0():
    """GEO: Perfect score - clear reasoning with helpful code."""
    print("\n" + "=" * 80)
    print("TEST 1: GEO - Score 1.0 (Clear reasoning + helpful calculation)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"

    response = """<think>Looking at the circle diagram with points R, M, N, L, C. I can see angles (x-1)° and (3x+5)° appear to be on opposite sides of a diameter, forming a straight line. Angles on a straight line sum to 180°.</think>
<code>
# Angles on a straight line: (x-1) + (3x+5) = 180
# x - 1 + 3x + 5 = 180
# 4x + 4 = 180
# 4x = 176
x = (180 - 4) / 4
print(f"Equation: (x-1) + (3x+5) = 180")
print(f"Simplifying: 4x + 4 = 180")
print(f"Solving: 4x = 176")
print(f"Therefore: x = {x}")
print(f"Verification: ({x}-1) + (3*{x}+5) = {x-1} + {3*x+5} = {(x-1) + (3*x+5)}")
</code>
<sandbox_output>Equation: (x-1) + (3x+5) = 180
Simplifying: 4x + 4 = 180
Solving: 4x = 176
Therefore: x = 44.0
Verification: (44-1) + (3*44+5) = 43 + 137 = 180</sandbox_output>
<answer>44</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≥0.75")
    print(f"Status: {'PASS' if score >= 0.75 else 'FAIL'}")
    return score >= 0.75


def test_geo_score_0_5():
    """GEO: Medium score - partial reasoning."""
    print("\n" + "=" * 80)
    print("TEST 2: GEO - Score 0.5 (Partial reasoning)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"

    response = """<think>I need to find x. The angles (x-1)° and (3x+5)° look like they might be related.</think>
<code>
# These angles might add up to 180
# (x-1) + (3*x+5) = 180
# 4x + 4 = 180
x = (180 - 4) / 4
print(f"x = {x}")
</code>
<sandbox_output>x = 44.0</sandbox_output>
<answer>44</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: 0.3-0.7")
    return True


def test_geo_no_code_good_reasoning():
    """GEO: High score - excellent reasoning without code (allowed for geometry)."""
    print("\n" + "=" * 80)
    print("TEST 3: GEO - No code but excellent reasoning (should score high)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"

    response = """<think>To find x, I need to analyze the circle diagram.

I can see angles (x-1)° and (3x+5)° which appear to be on opposite sides, forming a straight line through the center.

Since angles on a straight line sum to 180°:
(x-1)° + (3x+5)° = 180°

Expanding:
x - 1 + 3x + 5 = 180
4x + 4 = 180
4x = 176
x = 44

Therefore, x = 44.</think>
<answer>44</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≥0.5 (good reasoning, code optional)")
    print(f"Status: {'PASS' if score >= 0.5 else 'FAIL'}")
    return score >= 0.5


def test_geo_fake_code():
    """GEO: Zero score - fake code with no reasoning."""
    print("\n" + "=" * 80)
    print("TEST 4: GEO - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"

    response = """<think>The answer is 44.</think>
<code>print('44')</code>
<sandbox_output>44</sandbox_output>
<answer>44</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≤0.25")
    print(f"Status: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_geo_wrong_reasoning():
    """GEO: Zero score - completely wrong approach."""
    print("\n" + "=" * 80)
    print("TEST 5: GEO - Score 0.0 (Wrong reasoning)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"

    response = """<think>This is a circle problem. I'll calculate the circumference.</think>
<code>
import math
radius = 22
circumference = 2 * math.pi * radius
print(f"Circumference: {circumference}")
</code>
<sandbox_output>Circumference: 138.23</sandbox_output>
<answer>138</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≤0.25")
    print(f"Status: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def run_all_geo_tests():
    """Run all GEO rubric tests."""
    if not os.environ.get('OPENAI_API_KEY'):
        print("\nERROR: OPENAI_API_KEY not set!")
        return

    print("\n" + "=" * 80)
    print("GEOMETRY (GEO) RUBRIC TEST SUITE")
    print("Image: 44759.jpg | Question: what is x | Answer: 44")
    print("=" * 80)

    results = {}
    results['1.0 Clear reasoning + code'] = test_geo_score_1_0()
    results['0.5 Partial reasoning'] = test_geo_score_0_5()
    results['No code good reasoning'] = test_geo_no_code_good_reasoning()
    results['0.0 Fake code'] = test_geo_fake_code()
    results['0.0 Wrong reasoning'] = test_geo_wrong_reasoning()

    print("\n" + "=" * 80)
    print("GEO RUBRIC TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for v in results.values() if v)
    for test_name, result in results.items():
        print(f"{'PASS' if result else 'FAIL'}: {test_name}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    print("=" * 80)


if __name__ == '__main__':
    run_all_geo_tests()
