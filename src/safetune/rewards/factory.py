"""
Factory for creating SafeTune reward functions.
"""

from typing import List, Dict, Type
from .base import RewardType, RewardConfig, RewardFunction
from .text import (
    LengthReward, CoherenceReward, DiversityReward, 
    FluencyReward, ReadabilityReward, ConcisenessReward, BrevityReward
)
from .nlp import (
    SentimentReward, BLEUReward, ROUGEReward, 
    METEORReward, BERTScoreReward, SemanticSimilarityReward
)
from .safety import (
    SafetyReward, ToxicityReward, BiasReward, 
    HarmlessnessReward, HallucinationReward
)
from .code import (
    CodeSyntaxReward, CodeExecutionReward, 
    CodeCompletenessReward, CodeQualityReward, CodeCorrectnessReward,
    MBPPReward
)
from .math import (
    MathCorrectnessReward, MathReasoningReward, 
    LogicalConsistencyReward, CommonsenseReward, 
    CausalReasoningReward, CounterfactualReasoningReward,
    CounterfactualMathReward
)
from .specialized import (
    MedicalAccuracyReward, LegalComplianceReward, FinancialAccuracyReward,
    PolitenessReward, HelpfulnessReward, HonestyReward,
    InstructionFollowingReward, ContextRelevanceReward, 
    TemporalConsistencyReward, EngagementReward,
    ImageRelevanceReward, AudioQualityReward
)

class RewardFunctionFactory:
    """Factory for creating reward functions."""
    
    _reward_classes: Dict[RewardType, Type[RewardFunction]] = {
        RewardType.LENGTH: LengthReward,
        RewardType.COHERENCE: CoherenceReward,
        RewardType.FLUENCY: FluencyReward,
        RewardType.SENTIMENT: SentimentReward,
        RewardType.SAFETY: SafetyReward,
        # NOTE: FACTUALITY intentionally has no mapping — MedicalAccuracyReward
        # (medical-terminology heuristics) was the wrong signal for a
        # general-domain "factuality" name. Register a real implementation
        # before re-adding.
        RewardType.BIAS: BiasReward,
        RewardType.BLEU: BLEUReward,
        RewardType.ROUGE: ROUGEReward,
        RewardType.METEOR: METEORReward,
        RewardType.BERTSCORE: BERTScoreReward,
        RewardType.MATH_CORRECTNESS: MathCorrectnessReward,
        RewardType.CODE_SYNTAX: CodeSyntaxReward,
        RewardType.CODE_EXECUTION: CodeExecutionReward,
        RewardType.CODE_COMPLETENESS: CodeCompletenessReward,
        RewardType.LOGICAL_CONSISTENCY: LogicalConsistencyReward,
        RewardType.COMMONSENSE: CommonsenseReward,
        RewardType.HALLUCINATION: HallucinationReward,
        RewardType.TOXICITY: ToxicityReward,
        RewardType.POLITENESS: PolitenessReward,
        RewardType.HELPFULNESS: HelpfulnessReward,
        RewardType.HONESTY: HonestyReward,
        RewardType.MATH_REASONING: MathReasoningReward,
        RewardType.CODE_QUALITY: CodeQualityReward,
        RewardType.CODE_CORRECTNESS: CodeCorrectnessReward,
        RewardType.DIVERSITY: DiversityReward,
        # REVELANCE and others
        RewardType.BREVITY: BrevityReward,
        RewardType.INSTRUCTION_FOLLOWING: InstructionFollowingReward,
        RewardType.HARMLESSNESS: HarmlessnessReward,
        RewardType.CONCISENESS: ConcisenessReward,
        RewardType.CONTEXT_RELEVANCE: ContextRelevanceReward,
        RewardType.TEMPORAL_CONSISTENCY: TemporalConsistencyReward,
        RewardType.SEMANTIC_SIMILARITY: SemanticSimilarityReward,
        RewardType.READABILITY: ReadabilityReward,
        RewardType.ENGAGEMENT: EngagementReward,
        RewardType.MEDICAL_ACCURACY: MedicalAccuracyReward,
        RewardType.LEGAL_COMPLIANCE: LegalComplianceReward,
        RewardType.FINANCIAL_ACCURACY: FinancialAccuracyReward,
        RewardType.CAUSAL_REASONING: CausalReasoningReward,
        RewardType.COUNTERFACTUAL_REASONING: CounterfactualReasoningReward,
        RewardType.IMAGE_RELEVANCE: ImageRelevanceReward,
        RewardType.AUDIO_QUALITY: AudioQualityReward,
        RewardType.COUNTERFACTUAL_MATH: CounterfactualMathReward,
        RewardType.MBPP_REWARD: MBPPReward,
    }
    
    @classmethod
    def create_reward(cls, config: RewardConfig) -> RewardFunction:
        """Create a reward function from config."""
        if config.reward_type not in cls._reward_classes:
            raise ValueError(f"Unknown reward type: {config.reward_type}")
        
        reward_class = cls._reward_classes[config.reward_type]
        return reward_class(config)
    
    @classmethod
    def create_rewards(cls, configs: List[RewardConfig]) -> List[RewardFunction]:
        """Create multiple reward functions."""
        return [cls.create_reward(config) for config in configs]
    
    @classmethod
    def register_reward(cls, reward_type: RewardType, reward_class: Type[RewardFunction]):
        """Register a custom reward function."""
        cls._reward_classes[reward_type] = reward_class
