"""
Math and reasoning reward functions for SafeTune.
"""

from typing import List, Optional
import logging
import re
import math
from .base import RewardFunction
from safetune.utils.math_grading import (
    grade_math_answer,
    grade_gsm8k_answer,
    extract_math_answer,
    extract_gsm8k_answer,
)

logger = logging.getLogger(__name__)

class MathCorrectnessReward(RewardFunction):
    """Reward for mathematical correctness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        math_expressions = self._extract_math_expressions(text)
        if not math_expressions: return 0.0
        correct_count = 0
        for expr in math_expressions:
            if self._evaluate_math_expression(expr):
                correct_count += 1
        return correct_count / len(math_expressions)
    
    def _extract_math_expressions(self, text: str) -> List[str]:
        pattern = r'\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*'
        expressions = re.findall(pattern, text)
        equation_pattern = r'\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*\s*=\s*\d+\.?\d*'
        equations = re.findall(equation_pattern, text)
        return expressions + equations
    
    def _evaluate_math_expression(self, expr: str) -> bool:
        try:
            if '=' in expr:
                left, right = expr.split('=')
                return math.isclose(eval(left.strip(), {"__builtins__": None}, {}), 
                                   eval(right.strip(), {"__builtins__": None}, {}), rel_tol=1e-6)
            else:
                eval(expr, {"__builtins__": None}, {})
                return True
        except Exception:
            return False


class MathReasoningReward(RewardFunction):
    """Reward function for mathematical reasoning quality."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        total_score = 0.0
        components = 0
        params = self.config.params
        
        if params.get('check_correctness', True) and reference:
            total_score += self._evaluate_correctness(text, reference)
            components += 1
        
        if params.get('check_reasoning', True):
            total_score += self._evaluate_reasoning(text)
            components += 1
        
        if params.get('check_format', True):
            total_score += self._evaluate_format(text)
            components += 1
        
        return (total_score / components) if components > 0 else 0.0
    
    def _evaluate_correctness(self, response: str, reference: str) -> float:
        params = self.config.params
        dataset_type = params.get('dataset_type', 'auto')
        if dataset_type == 'auto':
            dataset_type = 'math' if '\\boxed' in reference or '\\frac' in reference else 'gsm8k'

        if dataset_type == 'math':
            pred = extract_math_answer(response)
            gold = extract_math_answer(reference) if '\\boxed' in reference else reference
            return 1.0 if grade_math_answer(pred, gold) else 0.0
        else:
            pred = extract_gsm8k_answer(response)
            gold = extract_gsm8k_answer(reference) if '####' in reference else reference
            return 1.0 if grade_gsm8k_answer(pred, gold) else 0.0
    
    def _evaluate_reasoning(self, response: str) -> float:
        step_indicators = ['step 1', 'first', 'then', 'therefore', 'thus', 'so', 'because']
        step_count = sum(1 for indicator in step_indicators if indicator in response.lower())
        score = 0.0
        if step_count >= 2: score += 0.4
        elif step_count >= 1: score += 0.2
        if bool(re.search(r'[+\-*/=]|\d+', response)): score += 0.3
        if len(response.split('.')) >= 3: score += 0.2
        if any(ind in response.lower() for ind in ['answer:', 'final answer']): score += 0.1
        return min(score, 1.0)
    
    def _evaluate_format(self, response: str) -> float:
        score = 0.0
        if re.search(r'\$[^$]+\$|\\\[.*?\\\]', response): score += 0.2
        if re.search(r'\d+\s*[+\-*/÷×]\s*\d+', response): score += 0.3
        if re.search(r'[=]', response): score += 0.2
        if len(response.split()) > 0 and len(set(response.lower().split())) / len(response.split()) > 0.5:
            score += 0.3
        return min(score, 1.0)


class LogicalConsistencyReward(RewardFunction):
    """Reward for logical consistency."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2: return 1.0
        score = 1.0
        text_lower = text.lower()
        contradictions = [("always", "never"), ("all", "none"), ("true", "false")]
        for pos, neg in contradictions:
            if pos in text_lower and neg in text_lower: score -= 0.2
        logical_connectors = ["therefore", "thus", "hence", "because"]
        score += min(0.3, sum(1 for c in logical_connectors if c in text_lower) * 0.1)
        return max(0.0, min(1.0, score))


class CommonsenseReward(RewardFunction):
    """Reward for commonsense reasoning."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        indicators = ["common sense", "obvious", "logical", "reasonable", "naturally"]
        score = min(1.0, sum(1 for i in indicators if i in text_lower) * 0.2)
        absurds = ["impossible", "absurd", "ridiculous"]
        return max(0.0, score - sum(1 for i in absurds if i in text_lower) * 0.3)


class CausalReasoningReward(RewardFunction):
    """Reward for causal reasoning."""
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        causal_indicators = ["because", "causes", "leads to", "result of", "consequently"]
        return min(1.0, sum(1 for i in causal_indicators if i in text_lower) * 0.2)


class CounterfactualReasoningReward(RewardFunction):
    """Reward for counterfactual reasoning markers ("if X were Y, then ...").

    A counterfactual response should both (a) acknowledge the hypothetical
    premise and (b) reason about its consequences. We score on the joint
    presence of antecedent markers ("if", "suppose", "had", "were") and
    consequent markers ("then", "would", "could", "might"). A response with
    neither lands at zero; one with both rich antecedent and consequent
    markers approaches 1.0.
    """

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        antecedent = ["if ", "suppose", "imagine", " were ", " had ", "instead of"]
        consequent = ["then ", "would ", "could ", "might ", "consequently", "result"]
        a = sum(1 for w in antecedent if w in text_lower)
        c = sum(1 for w in consequent if w in text_lower)
        if a == 0 or c == 0:
            return 0.0
        return min(1.0, 0.2 * a + 0.2 * c)


class CounterfactualMathReward(RewardFunction):
    """Reward for counterfactual math reasoning.

    Counterfactual math problems pose hypotheticals like "if 2 + 2 were 5,
    what would 4 + 4 equal?" A good response should (a) keep the original
    counterfactual axiom internally consistent and (b) arrive at a numeric
    answer. We check for a numeric answer in the response, presence of
    counterfactual markers, and optional agreement with a ``reference``
    answer string if supplied.
    """

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        score = 0.0
        text_lower = text.lower()
        markers = ["if ", "suppose", "assume", " were ", " instead "]
        if any(m in text_lower for m in markers):
            score += 0.4
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if nums:
            score += 0.3
            if reference is not None:
                ref_nums = re.findall(r"-?\d+(?:\.\d+)?", reference)
                if ref_nums and nums[-1] == ref_nums[-1]:
                    score += 0.3
                else:
                    score += 0.0
            else:
                score += 0.3
        return min(1.0, score)
