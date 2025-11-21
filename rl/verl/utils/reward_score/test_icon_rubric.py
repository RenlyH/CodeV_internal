#!/usr/bin/env python
"""
Test suite for Counting/Icon (ICON) rubric evaluation.
Image: 44750.jpg
Question: how many in total
Answer: 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from verl.utils.reward_score.vl_agent import (
    rubrics_judge,
    extract_answer,
    RUBRIC_ICON_USAGE,
    RUBRICS_JUDGE_SYSTEM_PROMPT_ICON
)

IMAGE_PATH = "/workspace/shared/datasets/CodeV_images/44750.jpg"


def test_icon_score_1_0():
    """ICON: Perfect score - systematic counting with image processing."""
    print("\n" + "=" * 80)
    print("TEST 1: ICON - Score 1.0 (Systematic counting)")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"

    response = """<think>I need to count the total number of dots on the dice systematically using image processing.</think>
<code>
from PIL import Image
import numpy as np
from scipy import ndimage
import cv2

img = Image.open('/workspace/shared/datasets/CodeV_images/44750.jpg')
img_array = np.array(img)

# Convert to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# Use thresholding to detect white dots on red dice
# Dots are lighter colored circular regions
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# Find connected components (dots)
labeled, num_dots = ndimage.label(binary)

print(f"Image processed: {img.size}")
print(f"Binary threshold applied")
print(f"Detected {num_dots} white dots across both dice")
print(f"Die 1 appears to show 5 dots")
print(f"Die 2 appears to show 5 dots")
print(f"Total count: {num_dots}")
</code>
<sandbox_output>Image processed: (400, 300)
Binary threshold applied
Detected 10 white dots across both dice
Die 1 appears to show 5 dots
Die 2 appears to show 5 dots
Total count: 10</sandbox_output>
<answer>10</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_ICON_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≥0.75")
    print(f"Status: {'PASS' if score >= 0.75 else 'FAIL'}")
    return score >= 0.75


def test_icon_score_0_5():
    """ICON: Medium score - basic counting attempt."""
    print("\n" + "=" * 80)
    print("TEST 2: ICON - Score 0.5 (Basic counting)")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"

    response = """<think>Let me count the dots on the dice.</think>
<code>
from PIL import Image
import numpy as np

img = Image.open('/workspace/shared/datasets/CodeV_images/44750.jpg')
img_array = np.array(img)

# Basic approach: manually count from observation
# I see two dice with dots
die1_dots = 5
die2_dots = 5
total = die1_dots + die2_dots

print(f"Die 1: {die1_dots} dots")
print(f"Die 2: {die2_dots} dots")
print(f"Total: {total}")
</code>
<sandbox_output>Die 1: 5 dots
Die 2: 5 dots
Total: 10</sandbox_output>
<answer>10</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_ICON_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: 0.3-0.7")
    return True


def test_icon_no_code():
    """ICON: Zero score - no code (REQUIRED for counting)."""
    print("\n" + "=" * 80)
    print("TEST 3: ICON - Score 0.0 (No code - CRITICAL)")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"

    response = """<think>Counting the objects in the image: I see 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 objects total.</think>
<answer>10</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_ICON_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≤0.25 (code REQUIRED for counting)")
    print(f"Status: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_icon_fake_code():
    """ICON: Zero score - fake code."""
    print("\n" + "=" * 80)
    print("TEST 4: ICON - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"

    response = """<think>There are 10 objects.</think>
<code>print('10')</code>
<sandbox_output>10</sandbox_output>
<answer>10</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_ICON_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≤0.25")
    print(f"Status: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_icon_wrong_count():
    """ICON: Low score - wrong count."""
    print("\n" + "=" * 80)
    print("TEST 5: ICON - Wrong count")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"

    response = """<think>Let me count the dots using image processing.</think>
<code>
from PIL import Image
import numpy as np
from scipy import ndimage

img = Image.open('/workspace/shared/datasets/CodeV_images/44750.jpg')
img_array = np.array(img)

# Wrong threshold - detecting too many regions
gray = np.mean(img_array, axis=2)
threshold = gray > 50  # TOO LOW threshold
labeled, count = ndimage.label(threshold)

print(f"Detected regions: {count}")
print(f"Total count: {count}")
</code>
<sandbox_output>Detected regions: 25
Total count: 25</sandbox_output>
<answer>25</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_ICON_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: Low (wrong answer)")
    return True


def run_all_icon_tests():
    """Run all ICON rubric tests."""
    if not os.environ.get('OPENAI_API_KEY'):
        print("\nERROR: OPENAI_API_KEY not set!")
        return

    print("\n" + "=" * 80)
    print("COUNTING (ICON) RUBRIC TEST SUITE")
    print("Image: 44750.jpg | Question: how many in total | Answer: 10")
    print("=" * 80)

    results = {}
    results['1.0 Systematic counting'] = test_icon_score_1_0()
    results['0.5 Basic counting'] = test_icon_score_0_5()
    results['0.0 No code (CRITICAL)'] = test_icon_no_code()
    results['0.0 Fake code'] = test_icon_fake_code()
    results['Wrong count'] = test_icon_wrong_count()

    print("\n" + "=" * 80)
    print("ICON RUBRIC TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for v in results.values() if v)
    for test_name, result in results.items():
        print(f"{'PASS' if result else 'FAIL'}: {test_name}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    print("=" * 80)


if __name__ == '__main__':
    run_all_icon_tests()
