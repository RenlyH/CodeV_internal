"""
Reward function for Python code execution in vision tasks
"""

import re
import ast
import sys
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PythonCodeRewardScorer:
    """
    Reward scorer for evaluating Python code generation and execution quality
    """
    
    def __init__(self):
        self.syntax_weight = 0.3
        self.execution_weight = 0.4
        self.task_completion_weight = 0.3
        
    def evaluate_syntax_quality(self, code: str) -> float:
        """
        Evaluate the syntactic quality of generated Python code
        
        Returns:
            float: Score between 0.0 and 1.0
        """
        if not code.strip():
            return 0.0
            
        try:
            # Parse the code to check syntax
            ast.parse(code)
            syntax_score = 1.0
        except SyntaxError:
            syntax_score = 0.0
        except Exception:
            syntax_score = 0.2  # Some other parsing issue
            
        # Additional quality checks
        quality_bonuses = 0.0
        
        # Check for proper imports
        if re.search(r'from PIL import Image|import PIL', code):
            quality_bonuses += 0.1
            
        # Check for proper image handling patterns
        if re.search(r'\.open\(|\.crop\(|\.rotate\(|\.show\(', code):
            quality_bonuses += 0.1
            
        # Check for proper variable naming
        if re.search(r'\b(img|image|result|output)\b', code):
            quality_bonuses += 0.05
            
        # Penalize overly complex single-line expressions
        lines = code.split('\\n')
        if any(len(line) > 120 for line in lines):
            quality_bonuses -= 0.05
            
        return min(1.0, syntax_score + quality_bonuses)
    
    def evaluate_execution_success(self, execution_info: Dict) -> float:
        """
        Evaluate whether the code executed successfully
        
        Args:
            execution_info: Dict containing execution results from tool
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        if execution_info.get("status") == "success":
            return 1.0
        elif execution_info.get("status") == "failed":
            error_msg = execution_info.get("error", "").lower()
            
            # Partial credit for certain types of errors
            if "timeout" in error_msg:
                return 0.1  # Code ran but was slow
            elif "nameerror" in error_msg or "undefined" in error_msg:
                return 0.2  # Variable issues
            elif "importerror" in error_msg or "modulenotfound" in error_msg:
                return 0.3  # Import issues
            else:
                return 0.0  # Other execution failures
        else:
            return 0.0
    
    def evaluate_task_completion(self, code: str, task_context: str) -> float:
        """
        Evaluate how well the code addresses the given task
        
        Args:
            code: The generated Python code
            task_context: Original task description/prompt
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        task_lower = task_context.lower()
        code_lower = code.lower()
        
        completion_score = 0.0
        
        # Task-specific pattern matching
        task_patterns = {
            'crop': [r'\.crop\(', r'crop\w*\s*='],
            'zoom': [r'\.crop\(', r'crop\w*\s*='],  # Zoom is typically done via crop
            'rotate': [r'\.rotate\(', r'rotate\w*\s*='],
            'resize': [r'\.resize\(', r'resize\w*\s*='],
            'flip': [r'\.transpose\(', r'\.flip\('],
            'combine': [r'\.paste\(', r'Image\.new\(', r'composite'],
            'filter': [r'ImageFilter', r'\.filter\('],
        }
        
        # Check if code addresses the mentioned tasks
        for task_keyword, patterns in task_patterns.items():
            if task_keyword in task_lower:
                for pattern in patterns:
                    if re.search(pattern, code_lower):
                        completion_score += 0.3
                        break
        
        # Check for output generation (show, save, return)
        if re.search(r'\.show\(|\.save\(|return\s+\w+', code_lower):
            completion_score += 0.2
            
        # Check for proper image loading
        if re.search(r'input_images\[|Image\.open\(', code_lower):
            completion_score += 0.1
            
        # Bonus for compositional operations (loops, multiple operations)
        if re.search(r'for\s+\w+\s+in|while\s+\w+', code_lower):
            completion_score += 0.1
            
        if len(re.findall(r'\.\w+\(', code_lower)) >= 3:  # Multiple method calls
            completion_score += 0.1
            
        return min(1.0, completion_score)
    
    def compute_reward(self, 
                      generated_code: str, 
                      execution_info: Dict, 
                      task_context: str,
                      **kwargs) -> float:
        """
        Compute overall reward for Python code generation
        
        Args:
            generated_code: The Python code generated by the model
            execution_info: Results from code execution
            task_context: Original task prompt
            
        Returns:
            float: Overall reward score
        """
        
        syntax_score = self.evaluate_syntax_quality(generated_code)
        execution_score = self.evaluate_execution_success(execution_info)
        task_score = self.evaluate_task_completion(generated_code, task_context)
        
        # Weighted combination
        total_reward = (
            self.syntax_weight * syntax_score +
            self.execution_weight * execution_score +
            self.task_completion_weight * task_score
        )
        
        logger.info(f"Code reward breakdown: syntax={syntax_score:.3f}, "
                   f"execution={execution_score:.3f}, task={task_score:.3f}, "
                   f"total={total_reward:.3f}")
        
        return total_reward


def reward_fn(batch_data: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Main reward function interface for training pipeline
    
    Args:
        batch_data: List of samples with generated responses and metadata
        
    Returns:
        List[float]: Reward scores for each sample
    """
    scorer = PythonCodeRewardScorer()
    rewards = []
    
    for sample in batch_data:
        try:
            # Extract information from sample
            response = sample.get('response', '')
            task_prompt = sample.get('prompt', '')
            
            # Extract code from response
            code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
            generated_code = code_match.group(1).strip() if code_match else ''
            
            # Extract execution info (this would come from the tool)
            execution_info = sample.get('execution_info', {'status': 'unknown'})
            
            if not generated_code:
                # No code generated - low reward
                reward = 0.1
            else:
                reward = scorer.compute_reward(
                    generated_code=generated_code,
                    execution_info=execution_info,
                    task_context=task_prompt
                )
            
            rewards.append(reward)
            
        except Exception as e:
            logger.error(f"Error computing reward for sample: {e}")
            rewards.append(0.0)
    
    return rewards


# For backward compatibility
def compute_reward(batch_data: List[Dict[str, Any]], **kwargs) -> List[float]:
    """Legacy interface"""
    return reward_fn(batch_data, **kwargs)