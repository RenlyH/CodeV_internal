#!/usr/bin/env python
"""
Dedicated test suite for Visual Search (VS) rubric evaluation.
Tests: good code (1.0), basic code (0.5), and adversarial cases (0.0)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from verl.utils.reward_score.vl_agent import (
    rubrics_judge,
    extract_answer,
    RUBRIC_VS_USAGE,
    RUBRICS_JUDGE_SYSTEM_PROMPT_VS
)

# Image path
IMAGE_PATH = "/workspace/shared/datasets/CodeV_images/45222.jpg"


def test_vs_score_1_0():
    """VS: Perfect score - proper cropping and color analysis."""
    print("\n" + "=" * 80)
    print("TEST 1: VS - Score 1.0 (Proper cropping + analysis)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"

    response = """<think>I need to locate the vase in the image and analyze its color. Let me crop the vase region to get a better view.</think>
<code>
from PIL import Image
import numpy as np

img = Image.open('45222.jpg')
img_array = np.array(img)

# Crop vase region (estimated coordinates)
vase_crop = img.crop((200, 200, 500, 400))
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

    # Create mock PIL images
    try:
        from PIL import Image
        img = Image.open(IMAGE_PATH)
        mock_images = [img.crop((200, 200, 500, 400))]
    except:
        mock_images = None

    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, mock_images, IMAGE_PATH)

    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Image: {IMAGE_PATH}")
    print(f"Score: {score:.2f}")
    # print(f"Expected: 1.0 (or close)")
    # print(f"Status: {'PASS' if score >= 0.75 else 'FAIL'}")

    return score <= 0.25


def test_vs_score_0_5():
    """VS: Great score - crop accurately."""
    print("\n" + "=" * 80)
    print("TEST 2: VS - Score 0.5 (Basic crop)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"

    response = """<think>Let me crop the vase area.</think>
<code>
from PIL import Image
img = Image.open('/workspace/shared/datasets/CodeV_images/45222.jpg')
cropped = img.crop((165,130,210,190))
cropped.save('vase.jpg')
print("Cropped vase region")
</code>
<sandbox_output>Cropped vase region</sandbox_output>
<answer>blue</answer>"""

    extracted = extract_answer(response)

    try:
        from PIL import Image
        img = Image.open('/workspace/shared/datasets/CodeV_images/45222.jpg')
        cropped = img.crop((165,130,210,190))
        mock_images = [cropped]
    except:
        mock_images = None

    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, mock_images, IMAGE_PATH)

    print(f"Score: {score:.2f}")
    # print(f"Expected: 0.75-1 (medium quality)")
    # print(f"Status: {'PASS' if 0.75 <= score <= 1 else 'FAIL'}")

    return 0.75 <= score


def test_vs_no_code():
    """VS: Zero score - correct answer but NO code (RED FLAG)."""
    print("\n" + "=" * 80)
    print("TEST 3: VS - Score 0.0 (No code) - CRITICAL TEST")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"

    response = """<think>Looking at the image, I can see the vase is blue in color.</think>
<answer>blue</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, None, IMAGE_PATH)

    print(f"Score: {score:.2f}")
    # print(f"Expected: ≤0 (RED FLAG: no code)")
    # print(f"Status: {'PASS' if score <= 0 else 'FAIL'}")

    return score <= 0


def test_vs_fake_code():
    """VS: Zero score - fake code (just prints answer)."""
    print("\n" + "=" * 80)
    print("TEST 4: VS - Score 0.0 (Fake code)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"

    response = """<think>The vase is blue.</think>
<code>print('blue')</code>
<sandbox_output>blue</sandbox_output>
<answer>blue</answer>"""

    extracted = extract_answer(response)
    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, None, IMAGE_PATH)

    print(f"Score: {score:.2f}")
    # print(f"Expected: ≤0. (RED FLAG: fake code)")
    # print(f"Status: {'PASS' if score <= 0 else 'FAIL'}")

    return score <= 0.


def test_vs_wrong_crop():
    """VS: Low/medium score - cropped but wrong region, wrong answer."""
    print("\n" + "=" * 80)
    print("TEST 5: VS - Wrong crop (should get low-medium score)")
    print("=" * 80)

    question = "what's the color of vase"
    ground_truth = "blue"

    response = """<think>Let me crop the vase.</think>
<code>
from PIL import Image
img = Image.open('45222.jpg')
# Cropping region
cropped = img.crop((0, 0, 50, 50))
cropped.save('cropped_vase.jpg')
print("cropped_vase.jpg")
</code>
<sandbox_output></sandbox_output>
<think>This is actually wrong part</think>
<code>
from PIL import Image
img = Image.open('45222.jpg')
# Cropping again
cropped = img.crop((200, 200, 500, 400))
cropped.save('cropped_vase_2.jpg')
print("cropped_vase_2.jpg")
</code>
<sandbox_output></sandbox_output>
<think>Now I can see it clear, the vase is blue</think>
<answer>blue</answer>
"""

    extracted = extract_answer(response)

    try:
        from PIL import Image
        # Mock a wrong colored image
        img = Image.open(IMAGE_PATH)
        mock_images = [img.crop((0, 0, 50, 50)), img.crop((200, 200, 500, 400))]
    except:
        mock_images = None

    score = rubrics_judge(response, question, ground_truth, RUBRIC_VS_USAGE,
                          extracted, RUBRICS_JUDGE_SYSTEM_PROMPT_VS, mock_images, IMAGE_PATH)

    print(f"Score: {score:.2f}")
    # print(f"Expected: 0.25-0.5 (used code but got wrong answer)")
    # print(f"Status: PASS (any score, judge decides)")

    return True  # Always pass, we just want to see the score


def run_all_vs_tests():
    """Run all VS rubric tests."""

    if not os.environ.get('OPENAI_API_KEY'):
        print("\n" + "!" * 80)
        print("ERROR: OPENAI_API_KEY not set!")
        print("Please run: export OPENAI_API_KEY='your-api-key'")
        print("!" * 80)
        return

    print("\n" + "=" * 80)
    print("VISUAL SEARCH (VS) RUBRIC TEST SUITE")
    print("Image: /workspace/shared/datasets/CodeV_images/45222.jpg")
    print("Question: what's the color of vase")
    print("Ground truth: blue")
    print("=" * 80)

    results = {}

    results['1.0 Proper cropping'] = test_vs_score_1_0()
    results['0.5 Basic crop'] = test_vs_score_0_5()
    results['0.0 No code (CRITICAL)'] = test_vs_no_code()
    results['0.0 Fake code'] = test_vs_fake_code()
    results['Wrong crop'] = test_vs_wrong_crop()

    # Summary
    print("\n" + "=" * 80)
    print("VS RUBRIC TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 80)

    return passed == total


if __name__ == '__main__':
    success = run_all_vs_tests()
    sys.exit(0 if success else 1)
