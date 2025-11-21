#!/usr/bin/env python
"""
Test suite for Default rubric evaluation.
Image: 44689.jpg
Question: which pair have strong mag force
Answer: Pair 2
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

IMAGE_PATH = "/workspace/shared/datasets/CodeV_images/44689.jpg"


def test_default_score_1_0():
    """DEFAULT: Perfect score - helpful analysis with code."""
    print("\n" + "=" * 80)
    print("TEST 1: DEFAULT - Score 1.0 (Helpful analysis)")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"

    response = """<think>I need to analyze the magnet pairs to determine which has stronger magnetic force. I'll examine the magnets' size, alignment, and poles.</think>
<code>
from PIL import Image
import pytesseract

img = Image.open('/workspace/shared/datasets/CodeV_images/44689.jpg')

# Extract text to get measurements
text = pytesseract.image_to_string(img)
print(f"Extracted: {text}")

# Analyze the two pairs
# Pair 1: Smaller magnets, distance 27mm
# Pair 2: Larger magnets, distance 45mm

# Magnetic force depends on magnet strength (size/material) and distance
# F ∝ (m1 × m2) / r²

# While Pair 1 has closer distance (27mm vs 45mm),
# Pair 2 has significantly LARGER magnets (more magnetic moment)

# Calculate relative forces (assuming magnet strength ∝ volume)
pair1_strength = 1.0  # Normalized small magnet
pair1_distance = 27
pair1_force = (pair1_strength ** 2) / (pair1_distance ** 2)

pair2_strength = 2.5  # Larger magnets
pair2_distance = 45
pair2_force = (pair2_strength ** 2) / (pair2_distance ** 2)

print(f"Pair 1 relative force: {pair1_force:.4f}")
print(f"Pair 2 relative force: {pair2_force:.4f}")
print(f"Stronger: {'Pair 2' if pair2_force > pair1_force else 'Pair 1'}")
</code>
<sandbox_output>Extracted: Pair 1    Pair 2
S N  S N    S N  S N
27 mm        45 mm
Pair 1 relative force: 0.0014
Pair 2 relative force: 0.0031
Stronger: Pair 2</sandbox_output>
<answer>Pair 2</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≥0.75")
    print(f"Status: {'PASS' if score >= 0.75 else 'FAIL'}")
    return score >= 0.75


def test_default_score_0_5():
    """DEFAULT: Medium score - basic attempt."""
    print("\n" + "=" * 80)
    print("TEST 2: DEFAULT - Score 0.5 (Basic attempt)")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"

    response = """<think>Let me look at the magnet pairs.</think>
<code>
from PIL import Image
img = Image.open('/workspace/shared/datasets/CodeV_images/44689.jpg')
print(f"Image loaded: {img.size}")

# Pair 2 has larger magnets, likely stronger
answer = "Pair 2"
print(f"Answer: {answer}")
</code>
<sandbox_output>Image loaded: (800, 600)
Answer: Pair 2</sandbox_output>
<answer>Pair 2</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: 0.3-0.7")
    return True


def test_default_no_code():
    """DEFAULT: Zero score - no code (REQUIRED for default)."""
    print("\n" + "=" * 80)
    print("TEST 3: DEFAULT - Score 0.0 (No code - CRITICAL)")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"

    response = """<think>Looking at the image, I need to compare the two magnet pairs.

Pair 1: Shows S-N magnets with 27mm distance
Pair 2: Shows S-N magnets with 45mm distance

Magnetic force follows F ∝ (m1 × m2) / r²

While Pair 1 has closer distance (27mm < 45mm), Pair 2 appears to have significantly LARGER magnets. Larger magnets have greater magnetic moment (strength).

The increase in magnet size/strength in Pair 2 more than compensates for the increased distance, resulting in stronger magnetic force overall.

Therefore, Pair 2 has stronger magnetic force.</think>
<answer>Pair 2</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≤0.25 (code REQUIRED)")
    print(f"Status: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_default_fake_code():
    """DEFAULT: Zero score - fake code."""
    print("\n" + "=" * 80)
    print("TEST 4: DEFAULT - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"

    response = """<think>Pair 2 has stronger force.</think>
<code>print('Pair 2')</code>
<sandbox_output>Pair 2</sandbox_output>
<answer>Pair 2</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≤0.25")
    print(f"Status: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_default_irrelevant():
    """DEFAULT: Low score - irrelevant code."""
    print("\n" + "=" * 80)
    print("TEST 5: DEFAULT - Irrelevant code")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"

    response = """<think>Let me analyze the magnetic force using irrelevant calculations.</think>
<code>
import random
import math

# Completely irrelevant calculation
result = math.pi * 27 + math.e * 45
print(f"Calculation result: {result}")

# Random guess
answer = random.choice(['Pair 1', 'Neither', 'Equal'])
print(f"Answer: {answer}")
</code>
<sandbox_output>Calculation result: 207.11
Answer: Neither</sandbox_output>
<answer>Neither</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: Low (irrelevant code + wrong answer)")
    return True


def run_all_default_tests():
    """Run all DEFAULT rubric tests."""
    if not os.environ.get('OPENAI_API_KEY'):
        print("\nERROR: OPENAI_API_KEY not set!")
        return

    print("\n" + "=" * 80)
    print("DEFAULT RUBRIC TEST SUITE")
    print("Image: 44689.jpg | Question: which pair have strong mag force | Answer: Pair 2")
    print("=" * 80)

    results = {}
    results['1.0 Helpful analysis'] = test_default_score_1_0()
    results['0.5 Basic attempt'] = test_default_score_0_5()
    results['0.0 No code (CRITICAL)'] = test_default_no_code()
    results['0.0 Fake code'] = test_default_fake_code()
    results['Irrelevant code'] = test_default_irrelevant()

    print("\n" + "=" * 80)
    print("DEFAULT RUBRIC TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for v in results.values() if v)
    for test_name, result in results.items():
        print(f"{'PASS' if result else 'FAIL'}: {test_name}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    print("=" * 80)


if __name__ == '__main__':
    run_all_default_tests()
