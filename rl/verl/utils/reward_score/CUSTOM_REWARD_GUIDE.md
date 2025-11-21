# Custom Reward Function Guide

This guide shows how to add custom process reward functions for different data sources.

## Architecture Overview

The refactored code separates concerns into:

1. **Shared utilities** (single function calls, no duplication):
   - `check_format_errors()` - Format validation
   - `extract_and_verify_answer()` - Answer extraction and verification
   - `rule_math_verify()` / `generative_verify()` - Answer judging

2. **Customizable process rewards** (per data source):
   - `compute_process_reward_default()` - Default for most datasets
   - `compute_process_reward_vstar()` - Example custom for vstar
   - `compute_process_reward_<your_dataset>()` - Add your own!

## How to Add Custom Reward for Your Dataset

### Step 1: Add Process Reward Function in `vl_agent.py`

```python
def compute_process_reward_mydataset(predict_str: str, question: str, ground_truth: str,
                                      is_format_error: bool, code_count: int) -> float:
    """
    Custom process reward for 'mydataset'.

    Args:
        predict_str: Full model response
        question: The question being asked
        ground_truth: Ground truth answer
        is_format_error: Whether there's a format error
        code_count: Number of <code> blocks in response

    Returns:
        float: Process reward score (typically 0.0 to 1.0)
    """
    # Don't give process reward if format error or no code
    if is_format_error or code_count < 1:
        return 0.0

    # Example: Use different combination of judges
    tool_score = judge_tool_usage(predict_str, question, ground_truth)
    reasoning_score = judge_reasoning_quality(predict_str, question, ground_truth)
    decomp_score = judge_problem_decomposition(predict_str, question, ground_truth)

    # Custom weighting for your dataset
    return 0.5 * tool_score + 0.3 * reasoning_score + 0.2 * decomp_score
```

### Step 2: Add Dispatcher in `compute_score_generative()`

In the `compute_score_generative()` function, add your data source to the if-elif chain:

```python
# Step 3: Compute process reward (customizable per data source)
if extra_info and 'question' in extra_info:
    # Dispatch to different process reward functions based on data_source
    if data_source == 'vstar':
        process_reward = compute_process_reward_vstar(...)
    elif data_source == 'mydataset':  # <-- ADD THIS
        process_reward = compute_process_reward_mydataset(
            predict_str, extra_info['question'], ground_truth, is_format_error, code_count
        )
    else:
        # Default for all other data sources
        process_reward = compute_process_reward_default(...)
```

### Step 3: (Optional) Customize Reward Weights

You can also customize the final reward combination per dataset:

```python
# Step 4: Combine rewards (can be customized per data source)
if data_source == 'mydataset':
    # Custom weights for your dataset
    total_reward = 0.8 * acc_reward + 0.1 * format_reward + 0.5 * process_reward
else:
    # Default weights
    total_reward = 1.0 * acc_reward + 0.2 * format_reward + 0.2 * process_reward
```

## Example: Different Reward Functions for Different Datasets

```python
# vstar: Focus on tool usage
def compute_process_reward_vstar(...):
    if is_format_error or code_count < 1:
        return 0.0
    return judge_tool_usage(predict_str, question, ground_truth)

# chart: Focus on reasoning quality
def compute_process_reward_chart(...):
    if is_format_error or code_count < 1:
        return 0.0
    return judge_reasoning_quality(predict_str, question, ground_truth)

# TabMWP: Balanced combination
def compute_process_reward_tabmwp(...):
    if is_format_error or code_count < 1:
        return 0.0
    tool = judge_tool_usage(predict_str, question, ground_truth)
    reasoning = judge_reasoning_quality(predict_str, question, ground_truth)
    return 0.6 * tool + 0.4 * reasoning
```

## Benefits of This Architecture

✅ **No code duplication**: Format checking and answer verification are single function calls
✅ **Easy to customize**: Just add a new `compute_process_reward_<dataset>()` function
✅ **Simple dispatcher**: Just `if data_source == 'xxx'` - no complex class hierarchies
✅ **Backwards compatible**: Default behavior maintained for existing datasets
✅ **Parallel execution**: Batched version automatically uses custom rewards per data source
