"""
Basic text quality reward functions for SafeTune.
"""

from typing import Optional
import numpy as np
import logging
from .base import RewardFunction

logger = logging.getLogger(__name__)

class LengthReward(RewardFunction):
    """Reward based on text length."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        min_length = self.config.params.get("min_length", 10)
        max_length = self.config.params.get("max_length", 500)
        target_length = self.config.params.get("target_length", None)
        
        word_count = len(text.split())
        
        if target_length:
            distance = abs(word_count - target_length)
            max_distance = max(target_length, 100)
            return max(0.0, 1.0 - (distance / max_distance))
        else:
            if word_count < min_length or word_count > max_length:
                return 0.0
            return 1.0


class CoherenceReward(RewardFunction):
    """Reward for text coherence."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.0
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        sentence_length_variance = np.var([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0
        
        length_score = min(1.0, avg_sentence_length / 20.0)
        variance_penalty = max(0.0, 1.0 - (sentence_length_variance / 100.0))
        
        return (length_score + variance_penalty) / 2.0


class DiversityReward(RewardFunction):
    """Reward function for response diversity and vocabulary richness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        params = self.config.params
        check_vocabulary = params.get('check_vocabulary', True)
        check_sentence_variety = params.get('check_sentence_variety', True)
        
        score = 0.0
        
        if check_vocabulary:
            words = text.lower().split()
            if len(words) > 0:
                unique_words = len(set(words))
                vocab_diversity = unique_words / len(words)
                score += min(vocab_diversity * 1.5, 0.5)
        
        if check_sentence_variety:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 1:
                lengths = [len(s.split()) for s in sentences]
                if lengths:
                    import statistics
                    try:
                        length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0
                        variety_score = min(length_std / 10, 0.5)
                        score += variety_score
                    except Exception:
                        score += 0.25
        
        return min(score, 1.0)


class FluencyReward(RewardFunction):
    """Reward function for language fluency and grammatical quality."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        params = self.config.params
        check_grammar = params.get('check_grammar', True)
        check_coherence = params.get('check_coherence', True)
        
        score = 0.0
        
        if check_grammar:
            grammar_score = 0.0
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences:
                capitalized = sum(1 for s in sentences if s and s[0].isupper())
                grammar_score += (capitalized / len(sentences)) * 0.3
            
            has_periods = text.count('.') > 0
            has_commas = text.count(',') > 0
            if has_periods:
                grammar_score += 0.2
            if has_commas:
                grammar_score += 0.1
            
            words = text.split()
            avg_sentence_length = len(words) / max(len(sentences), 1)
            if 5 <= avg_sentence_length <= 25:
                grammar_score += 0.4
            
            score += min(grammar_score, 0.6)
        
        if check_coherence:
            transitions = ['however', 'therefore', 'moreover', 'furthermore', 
                          'additionally', 'consequently', 'thus', 'hence', 
                          'meanwhile', 'similarly', 'in contrast', 'for example']
            has_transitions = any(trans in text.lower() for trans in transitions)
            if has_transitions:
                score += 0.2
            
            connectors = ['and', 'but', 'or', 'because', 'so', 'then', 'when', 'if']
            connector_count = sum(1 for conn in connectors if conn in text.lower())
            score += min(connector_count / 10, 0.2)
        
        return min(score, 1.0)


class ReadabilityReward(RewardFunction):
    """Reward for text readability."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        sentence_score = 1.0 - min(1.0, (avg_sentence_length - 15) / 30)
        word_score = 1.0 - min(1.0, (avg_word_length - 4.5) / 5)
        
        complex_punct = text.count(';') + text.count(':') + text.count('—')
        punct_penalty = min(0.2, complex_punct * 0.05)
        
        readability_score = (sentence_score * 0.5 + word_score * 0.5) - punct_penalty
        return max(0.0, min(1.0, readability_score))


class ConcisenessReward(RewardFunction):
    """Reward for being concise without losing information."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        words = text.split()
        word_count = len(words)
        
        unique_words = len(set(word.lower() for word in words))
        redundancy_ratio = 1.0 - (unique_words / word_count) if word_count > 0 else 0.0
        
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 
                       'literally', 'really', 'very', 'quite', 'rather', 'somewhat']
        filler_count = sum(1 for word in words if word.lower() in filler_words)
        filler_ratio = filler_count / word_count if word_count > 0 else 0.0
        
        information_density = unique_words / word_count if word_count > 0 else 0.0
        
        score = information_density * 0.5
        score -= redundancy_ratio * 0.3
        score -= filler_ratio * 0.2
        
        return max(0.0, min(1.0, score))


class BrevityReward(RewardFunction):
    """Reward function for conciseness and brevity."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        ideal_length = self.config.params.get('ideal_length', 100)
        max_length = self.config.params.get('max_length', 300)
        
        words = text.split()
        word_count = len(words)
        
        if word_count <= ideal_length:
            return 1.0
        elif word_count <= max_length:
            penalty = (word_count - ideal_length) / (max_length - ideal_length)
            return (1.0 - penalty * 0.5)
        else:
            excess = word_count - max_length
            penalty = min(excess / max_length, 1.0)
            return max(0.0, 0.5 - penalty)
