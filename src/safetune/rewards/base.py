"""
Base classes and types for SafeTune reward functions.
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import torch

logger = logging.getLogger(__name__)

class RewardType(Enum):
    """Types of reward functions."""
    # Basic rewards
    LENGTH = "length"
    COHERENCE = "coherence"
    
    # Task-specific rewards
    SENTIMENT = "sentiment"
    SAFETY = "safety"
    FACTUALITY = "factuality"
    BIAS = "bias"
    
    # Generation quality
    BLEU = "bleu"
    ROUGE = "rouge"
    METEOR = "meteor"
    BERTSCORE = "bertscore"
    
    # Code quality
    CODE_SYNTAX = "code_syntax"
    CODE_EXECUTION = "code_execution"
    CODE_COMPLETENESS = "code_completeness"
    
    # Math and reasoning
    MATH_CORRECTNESS = "math_correctness"
    LOGICAL_CONSISTENCY = "logical_consistency"
    COMMONSENSE = "commonsense"
    
    # Specialized
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    POLITENESS = "politeness"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    
    # NEW: Math and reasoning
    MATH_REASONING = "math_reasoning"
    
    # NEW: Code rewards
    CODE_QUALITY = "code_quality"
    CODE_CORRECTNESS = "code_correctness"
    
    # NEW: Enhanced quality rewards
    DIVERSITY = "diversity"
    FLUENCY = "fluency"
    RELEVANCE = "relevance"
    BREVITY = "brevity"
    
    # NEW: Instruction following & alignment
    INSTRUCTION_FOLLOWING = "instruction_following"
    HARMLESSNESS = "harmlessness"
    CONCISENESS = "conciseness"
    
    # NEW: Context & temporal awareness
    CONTEXT_RELEVANCE = "context_relevance"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    
    # NEW: Advanced quality metrics
    SEMANTIC_SIMILARITY = "semantic_similarity"
    READABILITY = "readability"
    ENGAGEMENT = "engagement"
    
    # NEW: Domain-specific rewards
    MEDICAL_ACCURACY = "medical_accuracy"
    LEGAL_COMPLIANCE = "legal_compliance"
    FINANCIAL_ACCURACY = "financial_accuracy"
    
    # NEW: Advanced reasoning
    CAUSAL_REASONING = "causal_reasoning"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"
    
    # Multi-modal (for future)
    IMAGE_RELEVANCE = "image_relevance"
    AUDIO_QUALITY = "audio_quality"
    
    # New Rewards 
    COUNTERFACTUAL_MATH = "counterfactual_math"
    MBPP_REWARD = "mbpp_reward"


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    reward_type: RewardType
    weight: float = 1.0
    params: Dict[str, Any] = None
    model_name: Optional[str] = None
    device: str = "auto"
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class RewardFunction:
    """Base class for reward functions."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.device = self._setup_device()
        self._model = None
        self._tokenizer = None
    
    def _setup_device(self) -> str:
        """Setup device for reward computation."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Compute reward for given text."""
        raise NotImplementedError
    
    def batch_compute(self, texts: List[str], references: Optional[List[str]] = None, **kwargs) -> List[float]:
        """Compute rewards for a batch of texts."""
        if references is None:
            references = [None] * len(texts)
        
        # Check if this reward supports batch processing via pipeline
        if hasattr(self, '_batch_compute_pipeline'):
            return self._batch_compute_pipeline(texts, references, **kwargs)
        
        return [self.compute(text, ref, **kwargs) for text, ref in zip(texts, references)]


class CompositeReward(RewardFunction):
    """A reward function that combines multiple reward functions with weights."""
    
    def __init__(self, reward_functions: List[RewardFunction], weights: Optional[List[float]] = None):
        self.reward_functions = reward_functions
        if weights is None:
            self.weights = [rf.config.weight for rf in reward_functions]
        else:
            self.weights = weights
        
        # Create a dummy config for the composite reward
        config = RewardConfig(reward_type=RewardType.LENGTH, weight=sum(self.weights))
        super().__init__(config)
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        total_reward = 0.0
        for rf, weight in zip(self.reward_functions, self.weights):
            total_reward += weight * rf.compute(text, reference, **kwargs)
        return total_reward
    
    def batch_compute(self, texts: List[str], references: Optional[List[str]] = None, **kwargs) -> List[float]:
        batch_size = len(texts)
        total_rewards = [0.0] * batch_size
        
        for rf, weight in zip(self.reward_functions, self.weights):
            rewards = rf.batch_compute(texts, references, **kwargs)
            for i in range(batch_size):
                total_rewards[i] += weight * rewards[i]
                
        return total_rewards
