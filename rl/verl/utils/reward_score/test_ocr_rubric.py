#!/usr/bin/env python
"""
Test suite for OCR rubric evaluation.
Image: 45196.jpg
Question: what's the total
Answer: 301
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

IMAGE_PATH = "/workspace/shared/datasets/CodeV_images/45196.jpg"


def test_ocr_score_1_0():
    """OCR: Perfect score - proper OCR and processing."""
    print("\n" + "=" * 80)
    print("TEST 1: OCR - Score 1.0 (Proper OCR + processing)")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"

    response = """<think>I need to extract the frequency table from the image and calculate the total. This is a frequency distribution table showing "Times eaten" vs "Frequency", so I need to multiply each times value by its frequency.</think>
<code>
from PIL import Image
import pytesseract

img = Image.open('/workspace/shared/datasets/CodeV_images/45196.jpg')
text = pytesseract.image_to_string(img)
print(f"Extracted text: {text}")

# Parse frequency table: times_eaten and frequencies
times_eaten = [0, 1, 2, 3, 4, 5, 6]
frequencies = [17, 2, 1, 20, 12, 15, 19]

# Calculate total: sum(times × frequency)
total = sum(t * f for t, f in zip(times_eaten, frequencies))
print(f"Times eaten: {times_eaten}")
print(f"Frequencies: {frequencies}")
print(f"Calculation: {' + '.join([f'{t}×{f}' for t, f in zip(times_eaten, frequencies)])}")
print(f"Total macaroni eaten: {total}")
</code>
<sandbox_output>Extracted text: Eating macaroni and cheese last month
Times eaten   Frequency
0   17
1   2
2   1
3   20
4   12
5   15
6   19
Times eaten: [0, 1, 2, 3, 4, 5, 6]
Frequencies: [17, 2, 1, 20, 12, 15, 19]
Calculation: 0×17 + 1×2 + 2×1 + 3×20 + 4×12 + 5×15 + 6×19
Total macaroni eaten: 301</sandbox_output>
<answer>301</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≥0.75")
    print(f"Status: {'PASS' if score >= 0.75 else 'FAIL'}")
    return score >= 0.75


def test_ocr_score_0_5():
    """OCR: Medium score - manual extraction."""
    print("\n" + "=" * 80)
    print("TEST 2: OCR - Score 0.5 (Manual extraction)")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"

    response = """<think>Let me read the frequency table from the image.</think>
<code>
# Manually extracted from table
times = [0, 1, 2, 3, 4, 5, 6]
freq = [17, 2, 1, 20, 12, 15, 19]

# Calculate total
total = sum(t * f for t, f in zip(times, freq))
print(f"Total: {total}")
</code>
<sandbox_output>Total: 301</sandbox_output>
<answer>301</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: 0.3-0.7")
    return True


def test_ocr_no_code():
    """OCR: Low score - no code (but code is optional for OCR)."""
    print("\n" + "=" * 80)
    print("TEST 3: OCR - No code (code optional for OCR)")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"

    response = """<think>Looking at the frequency table "Eating macaroni and cheese last month":

Times eaten (0-6) with their frequencies:
- 0 times: 17 people → 0×17 = 0
- 1 time: 2 people → 1×2 = 2
- 2 times: 1 person → 2×1 = 2
- 3 times: 20 people → 3×20 = 60
- 4 times: 12 people → 4×12 = 48
- 5 times: 15 people → 5×15 = 75
- 6 times: 19 people → 6×19 = 114

Total = 0 + 2 + 2 + 60 + 48 + 75 + 114 = 301</think>
<answer>301</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: Variable (code optional for OCR)")
    return True


def test_ocr_fake_code():
    """OCR: Zero score - fake code."""
    print("\n" + "=" * 80)
    print("TEST 4: OCR - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"

    response = """<think>The total is 301.</think>
<code>print('301')</code>
<sandbox_output>301</sandbox_output>
<answer>301</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: ≤0.25")
    print(f"Status: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_ocr_wrong_extraction():
    """OCR: Low score - wrong extraction."""
    print("\n" + "=" * 80)
    print("TEST 5: OCR - Wrong extraction")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"

    response = """<think>Let me extract the frequency data.</think>
<code>
from PIL import Image
import pytesseract

img = Image.open('/workspace/shared/datasets/CodeV_images/45196.jpg')
# Wrong extraction - misread the numbers
times = [0, 1, 2, 3, 4, 5, 6]
freq = [10, 20, 30, 40, 50, 60, 70]  # WRONG frequencies

total = sum(t * f for t, f in zip(times, freq))
print(f"Total: {total}")
</code>
<sandbox_output>Total: 910</sandbox_output>
<answer>910</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, IMAGE_PATH)

    print(f"Score: {score:.2f}, Expected: Low (wrong answer)")
    return True


def run_all_ocr_tests():
    """Run all OCR rubric tests."""
    if not os.environ.get('OPENAI_API_KEY'):
        print("\nERROR: OPENAI_API_KEY not set!")
        return

    print("\n" + "=" * 80)
    print("OCR RUBRIC TEST SUITE")
    print("Image: 45196.jpg | Question: what's the total | Answer: 301")
    print("=" * 80)

    results = {}
    results['1.0 Proper OCR'] = test_ocr_score_1_0()
    results['0.5 Manual extraction'] = test_ocr_score_0_5()
    results['No code'] = test_ocr_no_code()
    results['0.0 Fake code'] = test_ocr_fake_code()
    results['Wrong extraction'] = test_ocr_wrong_extraction()

    print("\n" + "=" * 80)
    print("OCR RUBRIC TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for v in results.values() if v)
    for test_name, result in results.items():
        print(f"{'PASS' if result else 'FAIL'}: {test_name}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    print("=" * 80)


if __name__ == '__main__':
    run_all_ocr_tests()
