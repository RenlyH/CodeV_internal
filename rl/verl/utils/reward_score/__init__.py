# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code
import torch

def _batched_compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """
    Batched version of _default_compute_score with parallel execution for LLM-as-a-judge.
    This function is used by BatchRewardManager.

    Args:
        data_sources: List of data source identifiers
        solution_strs: List of solution strings to score
        ground_truths: List of ground truth answers
        extra_infos: List of extra information dicts
        **kwargs: Additional arguments (e.g., num_workers for parallelization)

    Returns:
        List of scores (can be floats or dicts)
    """
    # Check if this is a data source that needs batched parallel processing
    if len(data_sources) > 0 and data_sources[0] in ['high_res', 'vstar', 'vl_agent', 'chart', 'thinklite_eureka',
                                              'FigureQA', 'GeoQA', 'Geometry3K', 'IconQA',
                                              'ScienceQA', 'TabMWP']:
        # Use batched parallel version for LLM-as-a-judge
        from . import vl_agent
        return vl_agent.compute_score_generative_batched(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)

    # For other data sources, fall back to sequential processing
    results = []
    for i in range(len(solution_strs)):
        data_source = data_sources[i] if i < len(data_sources) else data_sources[0]
        extra_info = extra_infos[i] if extra_infos and i < len(extra_infos) else None
        score = _default_compute_score(data_source, solution_strs[i], ground_truths[i], extra_info)
        results.append(score)
    return results


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)

    elif data_source in ['rag_v2-train']:
        from . import agent
        res = agent.compute_score(solution_str, ground_truth)
    elif data_source in ['rag_v2-test']:
        from . import agent
        res = agent.compute_score_eval(solution_str, ground_truth)

    elif data_source in ['vstar', 'vl_agent', 'chart', 'high_res']:
        from . import vl_agent
        res = vl_agent.compute_score_generative(solution_str, ground_truth, extra_info, data_source=data_source)

    elif data_source in ['geoguessr']:
        from . import vl_agent
        res = vl_agent.compute_common_reasoning(solution_str, ground_truth, extra_info)

    elif data_source in ['thinklite_eureka', 'FigureQA', 'GeoQA', 'Geometry3K', 'IconQA', 'ScienceQA', 'TabMWP']:
        from . import vl_agent
        res = vl_agent.compute_score_generative(solution_str, ground_truth, extra_info, data_source=data_source)

    elif data_source in ['TabMWP']:
        from . import vl_agent
        res = vl_agent.compute_score_generative(solution_str, ground_truth, extra_info, data_source=data_source)

    elif data_source in ["frozenlake"]:
        res = 0.0

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
