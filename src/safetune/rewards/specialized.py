"""
Specialized domain-specific reward functions for SafeTune.
"""

from typing import Optional
import logging
import re
from .base import RewardFunction

logger = logging.getLogger(__name__)

class MedicalAccuracyReward(RewardFunction):
    """Reward for medical information accuracy."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        accuracy_indicators = ['peer-reviewed', 'clinical study', 'medical research', 'evidence-based', 'physician']
        accuracy_count = sum(1 for indicator in accuracy_indicators if indicator in text_lower)
        
        misinformation_indicators = ['miracle cure', 'guaranteed to work', 'secret remedy', 'doctors hate this']
        misinformation_count = sum(1 for indicator in misinformation_indicators if indicator in text_lower)
        
        disclaimer_indicators = ['consult a doctor', 'not medical advice', 'medical disclaimer']
        disclaimer_count = sum(1 for indicator in disclaimer_indicators if indicator in text_lower)
        
        score = min(0.5, accuracy_count * 0.1) + min(0.3, disclaimer_count * 0.15)
        score -= misinformation_count * 0.4
        return max(0.0, min(1.0, score))


class LegalComplianceReward(RewardFunction):
    """Reward for legal compliance and accuracy."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        accuracy_indicators = ['legal precedent', 'statute', 'regulation', 'attorney', 'compliance']
        accuracy_count = sum(1 for indicator in accuracy_indicators if indicator in text_lower)
        
        misinformation_indicators = ['guaranteed legal', 'legal loophole', 'legal trick']
        misinformation_count = sum(1 for indicator in misinformation_indicators if indicator in text_lower)
        
        disclaimer_indicators = ['not legal advice', 'consult an attorney', 'legal disclaimer']
        disclaimer_count = sum(1 for indicator in disclaimer_indicators if indicator in text_lower)
        
        score = min(0.4, accuracy_count * 0.1) + min(0.3, disclaimer_count * 0.15)
        score -= misinformation_count * 0.4
        return max(0.0, min(1.0, score))


class FinancialAccuracyReward(RewardFunction):
    """Reward for financial information accuracy."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        accuracy_indicators = ['financial analysis', 'sec filing', 'audited', 'cpa', 'disclosure']
        accuracy_count = sum(1 for indicator in accuracy_indicators if indicator in text_lower)
        
        misinformation_indicators = ['guaranteed returns', 'risk-free investment', 'get rich quick']
        misinformation_count = sum(1 for indicator in misinformation_indicators if indicator in text_lower)
        
        disclaimer_indicators = ['not financial advice', 'financial disclaimer', 'past performance']
        disclaimer_count = sum(1 for indicator in disclaimer_indicators if indicator in text_lower)
        
        score = min(0.4, accuracy_count * 0.1) + min(0.2, disclaimer_count * 0.1)
        score -= misinformation_count * 0.4
        return max(0.0, min(1.0, score))


class PolitenessReward(RewardFunction):
    """Reward for politeness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        polite_terms = ["please", "thank you", "excuse me", "kindly", "appreciate"]
        polite_count = sum(1 for t in polite_terms if t in text_lower)
        
        impolite_terms = ["shut up", "stupid", "idiot", "dumb", "annoying"]
        impolite_count = sum(1 for t in impolite_terms if t in text_lower)
        
        return max(0.0, min(1.0, polite_count * 0.2 - impolite_count * 0.3))


class HelpfulnessReward(RewardFunction):
    """Reward for helpfulness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        helpful_terms = ["here's how", "i can help", "suggest", "recommend", "guide"]
        helpful_count = sum(1 for t in helpful_terms if t in text_lower)
        
        unhelpful_terms = ["i can't help", "i don't know", "i'm not sure"]
        unhelpful_count = sum(1 for t in unhelpful_terms if t in text_lower)
        
        return max(0.0, min(1.0, helpful_count * 0.2 - unhelpful_count * 0.3))


class HonestyReward(RewardFunction):
    """Reward for honesty."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        honest_terms = ["honestly", "truthfully", "frankly", "i must admit"]
        honest_count = sum(1 for t in honest_terms if t in text_lower)
        
        dishonest_terms = ["lie", "fake", "deceive", "trick"]
        dishonest_count = sum(1 for t in dishonest_terms if t in text_lower)
        
        return max(0.0, min(1.0, honest_count * 0.2 - dishonest_count * 0.4))


class InstructionFollowingReward(RewardFunction):
    """Reward for following instructions accurately."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        instruction = kwargs.get('instruction', None)
        if not instruction: return 0.5
        text_lower, instr_lower = text.lower(), instruction.lower()
        
        score = 0.0
        keywords = ['write', 'explain', 'describe', 'list', 'compare']
        if any(kw in instr_lower and kw in text_lower for kw in keywords): score += 0.3
        if any(ind in text_lower for ind in ['done', 'complete', 'answer']): score += 0.2
        if any(kw in instr_lower for kw in ['format', 'structure']) and any(c in text for c in ['\n', ':', '-']): score += 0.2
        
        return min(1.0, score)


class ContextRelevanceReward(RewardFunction):
    """Reward for maintaining context relevance in conversations."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        context = kwargs.get('context', [])
        if not context: return 0.5
        text_words = set(text.lower().split())
        context_words = set(' '.join(context).lower().split())
        if not context_words: return 0.0
        overlap = len(text_words & context_words) / len(context_words)
        return min(1.0, overlap * 2.0)


class TemporalConsistencyReward(RewardFunction):
    """Reward for temporal coherence across responses."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        prev = kwargs.get('previous_texts', [])
        if not prev: return 0.5
        text_lower = text.lower()
        prev_text = ' '.join(prev).lower()
        score = 1.0
        pairs = [('yesterday', 'tomorrow'), ('past', 'future'), ('before', 'after')]
        for p, f in pairs:
            if p in prev_text and f in text_lower: score -= 0.3
            if f in prev_text and p in text_lower: score -= 0.3
        return max(0.0, score)


class EngagementReward(RewardFunction):
    """Reward for engaging and interesting content."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        indicators = ['interesting', 'fascinating', 'engaging', 'thought-provoking']
        score = min(0.4, sum(1 for i in indicators if i in text_lower) * 0.1)
        score += min(0.2, text.count('?') * 0.1)
        score += min(0.2, len(re.findall(r'\b(extremely|incredibly|remarkably)\s+\w+', text_lower)) * 0.1)
        return min(1.0, score)


class ImageRelevanceReward(RewardFunction):
    """Experimental stub: image-relevance scoring is not implemented.

    A faithful image-relevance reward needs the image alongside the text (e.g.
    a CLIP image-text similarity), which this text-only interface does not
    provide.  Until a multimodal backend is wired in, ``compute`` returns a
    neutral ``0.5`` and warns once so callers are not misled into treating it
    as a real signal.
    """

    _warned = False

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not ImageRelevanceReward._warned:
            logger.warning(
                "ImageRelevanceReward is an unimplemented stub (needs a "
                "multimodal image-text backend); returning a neutral 0.5."
            )
            ImageRelevanceReward._warned = True
        return 0.5


class AudioQualityReward(RewardFunction):
    """Experimental stub: audio-quality scoring is not implemented.

    A faithful audio-quality reward needs the audio signal, which this
    text-only interface does not provide.  Returns a neutral ``0.5`` and warns
    once until an audio backend is wired in.
    """

    _warned = False

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not AudioQualityReward._warned:
            logger.warning(
                "AudioQualityReward is an unimplemented stub (needs an audio "
                "backend); returning a neutral 0.5."
            )
            AudioQualityReward._warned = True
        return 0.5
