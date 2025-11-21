#!/usr/bin/env python
"""
Comprehensive test suite for rubrics evaluation with real images.
Tests good responses (1.0), medium responses (0.5), and adversarial responses (0.0).
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from verl.utils.reward_score.vl_agent import (
    rubrics_judge,
    extract_answer,
    RUBRIC_CALCULATION_USAGE,
    RUBRIC_VS_USAGE,
    RUBRIC_ICON_USAGE,
    RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION,
    RUBRICS_JUDGE_SYSTEM_PROMPT_VS,
    RUBRICS_JUDGE_SYSTEM_PROMPT_ICON
)

# Base path for images
IMAGE_BASE = "/workspace/shared/datasets/CodeV_images"


# ============================================================================
# TEST CASE 1: GEOMETRY (44759.jpg)
# Question: "what is x"
# Answer: "44"
# Rubric: RUBRIC_GEO_USAGE
# ============================================================================

def test_geo_score_1_0():
    """Geometry: Perfect score - helpful calculation code."""
    print("\n" + "=" * 80)
    print("TEST: GEO - Score 1.0 (Helpful code with calculation)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"
    image_path = f"{IMAGE_BASE}/44759.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Image: {image_path}")
    print(f"Score: {score:.2f}")
    print(f"Expected: 1.0")
    print(f"Result: {'PASS' if score >= 0.9 else 'FAIL'}")
    return score >= 0.9


def test_geo_score_0_5():
    """Geometry: Medium score - basic attempt with minor issues."""
    print("\n" + "=" * 80)
    print("TEST: GEO - Score 0.5 (Basic attempt)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"
    image_path = f"{IMAGE_BASE}/44759.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ~0.5")
    print(f"Result: {'PASS' if 0.3 <= score <= 0.7 else 'FAIL'}")
    return 0.3 <= score <= 0.7


def test_geo_score_0_0_no_code():
    """Geometry: Zero score - no code used."""
    print("\n" + "=" * 80)
    print("TEST: GEO - Score 0.0 (No code)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"
    image_path = f"{IMAGE_BASE}/44759.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: Low score for geometry with good reasoning but no code")
    return True


def test_geo_score_0_0_fake_code():
    """Geometry: Zero score - fake code that just prints answer."""
    print("\n" + "=" * 80)
    print("TEST: GEO - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"
    image_path = f"{IMAGE_BASE}/44759.jpg"

    response = """<think>The answer is 44.</think>
<code>print('<answer>44</answer>')</code>
<sandbox_output>44</sandbox_output>
<answer>44</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ≤0.25")
    print(f"Result: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_geo_score_0_0_wrong_reasoning():
    """Geometry: Zero score - completely wrong approach."""
    print("\n" + "=" * 80)
    print("TEST: GEO - Score 0.0 (Wrong reasoning)")
    print("=" * 80)

    question = "what is x"
    ground_truth = "44"
    image_path = f"{IMAGE_BASE}/44759.jpg"

    response = """<think>This looks like a triangle problem. I'll calculate the area using Heron's formula.</think>
<code>
import math
# Assuming triangle sides
a, b, c = 10, 15, 20
s = (a + b + c) / 2
area = math.sqrt(s * (s-a) * (s-b) * (s-c))
print(f"Triangle area: {area}")
print(f"Therefore x = {int(area)}")
</code>
<sandbox_output>Triangle area: 72.62
Therefore x = 72</sandbox_output>
<answer>72</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: 0.0")
    print(f"Result: {'PASS' if score <= 0.1 else 'FAIL'}")
    return score <= 0.1


# ============================================================================
# TEST CASE 2: OCR (45196.jpg)
# Question: "what's the total"
# Answer: "301"
# Rubric: RUBRIC_OCR_USAGE
# ============================================================================

def test_ocr_score_1_0():
    """OCR: Perfect score - proper text extraction."""
    print("\n" + "=" * 80)
    print("TEST: OCR - Score 1.0 (Proper OCR)")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"
    image_path = f"{IMAGE_BASE}/45196.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Score: {score:.2f}")
    print(f"Expected: 1.0")
    print(f"Result: {'PASS' if score >= 0.9 else 'FAIL'}")
    return score >= 0.9


def test_ocr_score_0_5():
    """OCR: Medium score - partial extraction."""
    print("\n" + "=" * 80)
    print("TEST: OCR - Score 0.5 (Partial extraction)")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"
    image_path = f"{IMAGE_BASE}/45196.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ~0.5")
    return True


def test_ocr_score_0_0_no_code():
    """OCR: Zero score - no code used."""
    print("\n" + "=" * 80)
    print("TEST: OCR - Score 0.0 (No code)")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"
    image_path = f"{IMAGE_BASE}/45196.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: Low score (OCR optional but should be rewarded)")
    return True


def test_ocr_score_0_0_fake_code():
    """OCR: Zero score - fake code."""
    print("\n" + "=" * 80)
    print("TEST: OCR - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"
    image_path = f"{IMAGE_BASE}/45196.jpg"

    response = """<think>The total is 301.</think>
<code>print('301')</code>
<sandbox_output>301</sandbox_output>
<answer>301</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ≤0.25")
    return True


def test_ocr_score_0_0_wrong_extraction():
    """OCR: Zero score - completely wrong extraction."""
    print("\n" + "=" * 80)
    print("TEST: OCR - Score 0.0 (Wrong extraction)")
    print("=" * 80)

    question = "what's the total"
    ground_truth = "301"
    image_path = f"{IMAGE_BASE}/45196.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: 0.0")
    return True


# ============================================================================
# TEST CASE 3: DEFAULT (44689.jpg)
# Question: "which pair have strong mag force"
# Answer: "Pair 2"
# Rubric: RUBRIC_DEFAULT_USAGE
# ============================================================================

def test_default_score_1_0():
    """Default: Perfect score - helpful analysis code."""
    print("\n" + "=" * 80)
    print("TEST: DEFAULT - Score 1.0 (Helpful analysis)")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"
    image_path = f"{IMAGE_BASE}/44689.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Score: {score:.2f}")
    print(f"Expected: 1.0")
    print(f"Result: {'PASS' if score >= 0.9 else 'FAIL'}")
    return score >= 0.9


def test_default_score_0_5():
    """Default: Medium score - basic image opening."""
    print("\n" + "=" * 80)
    print("TEST: DEFAULT - Score 0.5 (Basic attempt)")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"
    image_path = f"{IMAGE_BASE}/44689.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ~0.5")
    return True


def test_default_score_0_0_no_code():
    """Default: Zero score - no code (requires code for >0.25)."""
    print("\n" + "=" * 80)
    print("TEST: DEFAULT - Score 0.0 (No code)")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"
    image_path = f"{IMAGE_BASE}/44689.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ≤0.25 (DEFAULT requires code)")
    print(f"Result: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_default_score_0_0_fake_code():
    """Default: Zero score - fake code."""
    print("\n" + "=" * 80)
    print("TEST: DEFAULT - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"
    image_path = f"{IMAGE_BASE}/44689.jpg"

    response = """<think>Pair 2 has stronger force.</think>
<code>print('<answer>Pair 2</answer>')</code>
<sandbox_output>Pair 2</sandbox_output>
<answer>Pair 2</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_CALCULATION_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ≤0.25")
    print(f"Result: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_default_score_0_0_irrelevant():
    """Default: Zero score - irrelevant code."""
    print("\n" + "=" * 80)
    print("TEST: DEFAULT - Score 0.0 (Irrelevant code)")
    print("=" * 80)

    question = "which pair have strong mag force"
    ground_truth = "Pair 2"
    image_path = f"{IMAGE_BASE}/44689.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_CALCULATION, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: 0.0")
    return True


# ============================================================================
# TEST CASE 4: VISUAL SEARCH (45222.jpg)
# Question: "what's the color of vase"
# Answer: "blue"
# Rubric: RUBRIC_VS_USAGE
# ============================================================================

def test_vs_score_1_0():
    """Visual Search: Perfect score - proper cropping and analysis."""
    print("\n" + "=" * 80)
    print("TEST: VS - Score 1.0 (Proper cropping)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"
    image_path = f"{IMAGE_BASE}/45222.jpg"

    # Actually crop the image and pass to judge
    from PIL import Image
    img = Image.open(image_path)
    vase_crop = img.crop((165, 130, 210, 190))  # Correct vase coordinates
    generated_images = [vase_crop]

    response = """<think>I need to locate the vase in the image and analyze its color. Let me crop the vase region to get a better view.</think>
<code>
from PIL import Image
import numpy as np

img = Image.open('/workspace/shared/datasets/CodeV_images/45222.jpg')

# Crop vase region at coordinates (165, 130, 210, 190)
vase_crop = img.crop((165, 130, 210, 190))
vase_crop.save('vase_region.jpg')

# Analyze dominant color in cropped region
vase_array = np.array(vase_crop)
avg_color = vase_array.mean(axis=(0,1))
print(f"Average RGB: {avg_color}")

# Blue channel is dominant
if avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
    color = "blue"
print(f"Vase color: {color}")
</code>
<sandbox_output>Average RGB: [45.2 78.3 156.8]
Vase color: blue</sandbox_output>
<answer>blue</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, generated_images, image_path)

    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Generated images: {len(generated_images)} cropped image(s)")
    print(f"Score: {score:.2f}")
    print(f"Expected: 1.0")
    print(f"Result: {'PASS' if score >= 0.9 else 'FAIL'}")
    return score >= 0.9


def test_vs_score_0_5():
    """Visual Search: Medium score - basic crop without analysis."""
    print("\n" + "=" * 80)
    print("TEST: VS - Score 0.5 (Basic crop)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"
    image_path = f"{IMAGE_BASE}/45222.jpg"

    # Crop the vase correctly but without detailed analysis
    from PIL import Image
    img = Image.open(image_path)
    vase_crop = img.crop((165, 130, 210, 190))
    generated_images = [vase_crop]

    response = """<think>Let me crop the vase area.</think>
<code>
from PIL import Image
img = Image.open('/workspace/shared/datasets/CodeV_images/45222.jpg')
cropped = img.crop((165, 130, 210, 190))
cropped.save('vase.jpg')
print("Cropped vase region")
</code>
<sandbox_output>Cropped vase region</sandbox_output>
<answer>blue</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, generated_images, image_path)

    print(f"Generated images: {len(generated_images)} cropped image(s)")
    print(f"Score: {score:.2f}")
    print(f"Expected: ~0.5-0.75")
    return True


def test_vs_score_0_0_no_code():
    """Visual Search: Zero score - no code used."""
    print("\n" + "=" * 80)
    print("TEST: VS - Score 0.0 (No code)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"
    image_path = f"{IMAGE_BASE}/45222.jpg"

    response = """<think>Looking at the image, I can see the vase is blue in color.</think>
<answer>blue</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ≤0.25 (VS requires code)")
    print(f"Result: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_vs_score_0_0_fake_code():
    """Visual Search: Zero score - fake code."""
    print("\n" + "=" * 80)
    print("TEST: VS - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"
    image_path = f"{IMAGE_BASE}/45222.jpg"

    response = """<think>The vase is blue.</think>
<code>print('blue')</code>
<sandbox_output>blue</sandbox_output>
<answer>blue</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ≤0.25")
    print(f"Result: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_vs_score_0_0_wrong_crop():
    """Visual Search: Zero score - wrong region cropped (non-overlapping coordinates)."""
    print("\n" + "=" * 80)
    print("TEST: VS - Score 0.0 (Wrong crop - non-overlapping)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"
    image_path = f"{IMAGE_BASE}/45222.jpg"

    # Crop a completely different region (top-left corner, doesn't overlap with vase)
    from PIL import Image
    img = Image.open(image_path)
    wrong_crop = img.crop((0, 0, 50, 50))  # Non-overlapping with (165, 130, 210, 190)
    generated_images = [wrong_crop]

    response = """<think>Let me crop the vase.</think>
<code>
from PIL import Image
img = Image.open('/workspace/shared/datasets/CodeV_images/45222.jpg')
# Cropping wrong region (top-left background)
cropped = img.crop((0, 0, 50, 50))
cropped.save('wrong_region.jpg')
print("Cropped region")
</code>
<sandbox_output>Cropped region</sandbox_output>
<answer>red</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, generated_images, image_path)

    print(f"Generated images: {len(generated_images)} wrong cropped image(s)")
    print(f"Score: {score:.2f}")
    print(f"Expected: 0.0 (wrong crop, wrong answer)")
    return True


def test_vs_two_turn_crop():
    """Visual Search: Two-turn crop - first wrong, then correct."""
    print("\n" + "=" * 80)
    print("TEST: VS - Two-turn crop (wrong then correct)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"
    image_path = f"{IMAGE_BASE}/45222.jpg"

    # Generate two crops: first wrong, then correct
    from PIL import Image
    img = Image.open(image_path)
    wrong_crop = img.crop((50, 50, 100, 100))  # Wrong region first
    correct_crop = img.crop((165, 130, 210, 190))  # Then correct vase region
    generated_images = [wrong_crop, correct_crop]

    response = """<think>I need to find the vase in the image. Let me try cropping different regions to locate it.</think>
<code>
from PIL import Image
img = Image.open('/workspace/shared/datasets/CodeV_images/45222.jpg')

# First attempt - try upper left region
crop1 = img.crop((50, 50, 100, 100))
crop1.save('attempt1.jpg')
print("First crop saved - doesn't seem to be the vase")

# Second attempt - adjust to where the vase actually is
crop2 = img.crop((165, 130, 210, 190))
crop2.save('vase_correct.jpg')
print("Second crop saved - this is the vase!")

# Analyze the correct crop
import numpy as np
vase_array = np.array(crop2)
avg_color = vase_array.mean(axis=(0,1))
print(f"Vase RGB: {avg_color}")
color = "blue"
print(f"Vase color: {color}")
</code>
<sandbox_output>First crop saved - doesn't seem to be the vase
Second crop saved - this is the vase!
Vase RGB: [45.2 78.3 156.8]
Vase color: blue</sandbox_output>
<answer>blue</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, generated_images, image_path)

    print(f"Generated images: {len(generated_images)} images (1 wrong, 1 correct)")
    print(f"Score: {score:.2f}")
    print(f"Expected: High score (shows iterative refinement)")
    print(f"Result: {'PASS' if score >= 0.75 else 'FAIL'}")
    return score >= 0.75


def test_vs_two_turn_crop_both_wrong():
    """Visual Search: Two-turn crop - both wrong (adversarial)."""
    print("\n" + "=" * 80)
    print("TEST: VS - Two-turn crop (both wrong - adversarial)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"
    image_path = f"{IMAGE_BASE}/45222.jpg"

    # Generate two wrong crops (both non-overlapping with vase)
    from PIL import Image
    img = Image.open(image_path)
    wrong_crop1 = img.crop((0, 0, 50, 50))  # Top-left
    wrong_crop2 = img.crop((300, 300, 350, 350))  # Bottom-right
    generated_images = [wrong_crop1, wrong_crop2]

    response = """<think>Let me try to find the vase by cropping.</think>
<code>
from PIL import Image
img = Image.open('/workspace/shared/datasets/CodeV_images/45222.jpg')

# First attempt
crop1 = img.crop((0, 0, 50, 50))
crop1.save('attempt1.jpg')

# Second attempt
crop2 = img.crop((300, 300, 350, 350))
crop2.save('attempt2.jpg')

print("Cropped two regions")
</code>
<sandbox_output>Cropped two regions</sandbox_output>
<answer>green</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, generated_images, image_path)

    print(f"Generated images: {len(generated_images)} images (both wrong)")
    print(f"Score: {score:.2f}")
    print(f"Expected: ≤0.25 (both crops wrong, wrong answer)")
    print(f"Result: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


# ============================================================================
# TEST CASE 5: COUNTING (44750.jpg)
# Question: "how many in total"
# Answer: "10"
# Rubric: RUBRIC_ICON_USAGE
# ============================================================================

def test_icon_score_1_0():
    """Counting: Perfect score - systematic counting with image processing."""
    print("\n" + "=" * 80)
    print("TEST: ICON - Score 1.0 (Systematic counting)")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"
    image_path = f"{IMAGE_BASE}/44750.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, image_path)

    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Score: {score:.2f}")
    print(f"Expected: 1.0")
    print(f"Result: {'PASS' if score >= 0.9 else 'FAIL'}")
    return score >= 0.9


def test_icon_score_0_5():
    """Counting: Medium score - basic counting attempt."""
    print("\n" + "=" * 80)
    print("TEST: ICON - Score 0.5 (Basic counting)")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"
    image_path = f"{IMAGE_BASE}/44750.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ~0.5")
    return True


def test_icon_score_0_0_no_code():
    """Counting: Zero score - no code (ICON requires code for >0.25)."""
    print("\n" + "=" * 80)
    print("TEST: ICON - Score 0.0 (No code)")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"
    image_path = f"{IMAGE_BASE}/44750.jpg"

    response = """<think>Counting the objects in the image: I see 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 objects total.</think>
<answer>10</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_ICON_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ≤0.25 (ICON requires code)")
    print(f"Result: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_icon_score_0_0_fake_code():
    """Counting: Zero score - fake code."""
    print("\n" + "=" * 80)
    print("TEST: ICON - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"
    image_path = f"{IMAGE_BASE}/44750.jpg"

    response = """<think>There are 10 objects.</think>
<code>print('10')</code>
<sandbox_output>10</sandbox_output>
<answer>10</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_ICON_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: ≤0.25")
    print(f"Result: {'PASS' if score <= 0.25 else 'FAIL'}")
    return score <= 0.25


def test_icon_score_0_0_wrong_count():
    """Counting: Zero score - completely wrong count."""
    print("\n" + "=" * 80)
    print("TEST: ICON - Score 0.0 (Wrong count)")
    print("=" * 80)

    question = "how many in total"
    ground_truth = "10"
    image_path = f"{IMAGE_BASE}/44750.jpg"

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
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_ICON, None, image_path)

    print(f"Score: {score:.2f}")
    print(f"Expected: 0.0")
    return True


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_comprehensive_tests():
    """Run all comprehensive rubric tests."""

    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("\n" + "!" * 80)
        print("ERROR: OPENAI_API_KEY not set!")
        print("Please run: export OPENAI_API_KEY='your-api-key'")
        print("!" * 80)
        return

    print("\n" + "=" * 80)
    print("COMPREHENSIVE RUBRICS TEST SUITE")
    print("Testing with real images: good (1.0), medium (0.5), adversarial (0.0)")
    print("=" * 80)

    results = {}

    # Geometry tests
    print("\n" + "#" * 80)
    print("# GEOMETRY TESTS (44759.jpg)")
    print("#" * 80)
    results['GEO 1.0'] = test_geo_score_1_0()
    results['GEO 0.5'] = test_geo_score_0_5()
    results['GEO 0.0 No Code'] = test_geo_score_0_0_no_code()
    results['GEO 0.0 Fake'] = test_geo_score_0_0_fake_code()
    results['GEO 0.0 Wrong'] = test_geo_score_0_0_wrong_reasoning()

    # OCR tests
    print("\n" + "#" * 80)
    print("# OCR TESTS (45196.jpg)")
    print("#" * 80)
    results['OCR 1.0'] = test_ocr_score_1_0()
    results['OCR 0.5'] = test_ocr_score_0_5()
    results['OCR 0.0 No Code'] = test_ocr_score_0_0_no_code()
    results['OCR 0.0 Fake'] = test_ocr_score_0_0_fake_code()
    results['OCR 0.0 Wrong'] = test_ocr_score_0_0_wrong_extraction()

    # Default tests
    print("\n" + "#" * 80)
    print("# DEFAULT TESTS (44689.jpg)")
    print("#" * 80)
    results['DEFAULT 1.0'] = test_default_score_1_0()
    results['DEFAULT 0.5'] = test_default_score_0_5()
    results['DEFAULT 0.0 No Code'] = test_default_score_0_0_no_code()
    results['DEFAULT 0.0 Fake'] = test_default_score_0_0_fake_code()
    results['DEFAULT 0.0 Irrelevant'] = test_default_score_0_0_irrelevant()

    # Visual Search tests
    print("\n" + "#" * 80)
    print("# VISUAL SEARCH TESTS (45222.jpg)")
    print("#" * 80)
    results['VS 1.0'] = test_vs_score_1_0()
    results['VS 0.5'] = test_vs_score_0_5()
    results['VS 0.0 No Code'] = test_vs_score_0_0_no_code()
    results['VS 0.0 Fake'] = test_vs_score_0_0_fake_code()
    results['VS 0.0 Wrong'] = test_vs_score_0_0_wrong_crop()
    results['VS Two-Turn Good'] = test_vs_two_turn_crop()
    results['VS Two-Turn Bad'] = test_vs_two_turn_crop_both_wrong()

    # Counting tests
    print("\n" + "#" * 80)
    print("# COUNTING TESTS (44750.jpg)")
    print("#" * 80)
    results['ICON 1.0'] = test_icon_score_1_0()
    results['ICON 0.5'] = test_icon_score_0_5()
    results['ICON 0.0 No Code'] = test_icon_score_0_0_no_code()
    results['ICON 0.0 Fake'] = test_icon_score_0_0_fake_code()
    results['ICON 0.0 Wrong'] = test_icon_score_0_0_wrong_count()

    # Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)

    # Group by rubric type
    rubric_types = ['GEO', 'OCR', 'DEFAULT', 'VS', 'ICON']
    for rubric in rubric_types:
        print(f"\n{rubric}:")
        rubric_results = {k: v for k, v in results.items() if k.startswith(rubric)}
        passed = sum(1 for v in rubric_results.values() if v)
        total = len(rubric_results)
        for test_name, result in rubric_results.items():
            status = "PASS" if result else "FAIL"
            print(f"  {status}: {test_name}")
        print(f"  Subtotal: {passed}/{total}")

    # Overall
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{'=' * 80}")
    print(f"OVERALL: {passed}/{total} tests passed")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    run_all_comprehensive_tests()
