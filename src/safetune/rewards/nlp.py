"""
NLP-based reward functions for SafeTune.
"""

from typing import Optional
import logging
import numpy as np
from .base import RewardFunction, RewardConfig

logger = logging.getLogger(__name__)

class SentimentReward(RewardFunction):
    """Reward based on sentiment alignment."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._sentiment_pipeline = None
        self._max_length = 512
    
    def _get_sentiment_pipeline(self):
        if self._sentiment_pipeline is None:
            from transformers import pipeline  # lazy import
            model_name = self.config.model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            try:
                if hasattr(self._sentiment_pipeline, 'tokenizer') and self._sentiment_pipeline.tokenizer is not None:
                    self._max_length = min(getattr(self._sentiment_pipeline.tokenizer, 'model_max_length', 512), 512)
            except Exception:
                pass
        return self._sentiment_pipeline
    
    def _truncate_text(self, text: str) -> str:
        if not text: return text
        pipe = self._get_sentiment_pipeline()
        if hasattr(pipe, 'tokenizer') and pipe.tokenizer is not None:
            try:
                tokens = pipe.tokenizer.encode(text, add_special_tokens=True, max_length=self._max_length, truncation=True)
                return pipe.tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception:
                pass
        return text[:self._max_length * 3]
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        target_sentiment = self.config.params.get("target_sentiment", "positive")
        try:
            text = self._truncate_text(text)
            pipe = self._get_sentiment_pipeline()
            result = pipe(text)[0]
            if result['label'].lower() == target_sentiment.lower():
                return result['score']
            else:
                return 1.0 - result['score']
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.5


class BLEUReward(RewardFunction):
    """Reward based on BLEU score."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        try:
            from sacrebleu import BLEU
            self.bleu = BLEU()
        except ImportError:
            self.bleu = None
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None: return 0.0
        if self.bleu:
            try:
                score = self.bleu.sentence_score(text, [reference])
                return score.score / 100.0
            except Exception:
                return 0.0
        return self._simple_bleu(text, reference)
    
    def _simple_bleu(self, text: str, reference: str) -> float:
        text_tokens = text.lower().split()
        ref_tokens = reference.lower().split()
        if not text_tokens or not ref_tokens: return 0.0
        text_1grams = set(text_tokens)
        ref_1grams = set(ref_tokens)
        precision_1 = len(text_1grams & ref_1grams) / len(text_1grams)
        bp = min(1.0, len(text_tokens) / len(ref_tokens))
        return bp * precision_1


class ROUGEReward(RewardFunction):
    """Reward based on ROUGE score."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except ImportError:
            self.scorer = None
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None: return 0.0
        if self.scorer:
            try:
                scores = self.scorer.score(reference, text)
                return scores['rougeL'].fmeasure
            except Exception:
                return 0.0
        return self._simple_rouge(text, reference)
    
    def _simple_rouge(self, text: str, reference: str) -> float:
        text_words = text.lower().split()
        ref_words = reference.lower().split()
        if not text_words or not ref_words: return 0.0
        m, n = len(text_words), len(ref_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text_words[i-1] == ref_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        p, r = lcs / m, lcs / n
        return 2 * p * r / (p + r) if p + r > 0 else 0.0


class METEORReward(RewardFunction):
    """Reward based on METEOR score."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None: return 0.0
        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            text_tokens = nltk.word_tokenize(text.lower())
            ref_tokens = nltk.word_tokenize(reference.lower())
            return float(meteor_score([ref_tokens], text_tokens))
        except Exception:
            return 0.0


class BERTScoreReward(RewardFunction):
    """Reward based on BERTScore."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._bertscorer = None
    
    def _get_bertscorer(self):
        if self._bertscorer is None:
            try:
                from bert_score import score
                self._bertscorer = score
            except ImportError:
                self._bertscorer = None
        return self._bertscorer
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None: return 0.0
        scorer = self._get_bertscorer()
        if scorer is None: return 0.0
        try:
            P, R, F1 = scorer([text], [reference], lang="en")
            return float(F1[0])
        except Exception:
            return 0.0


class SemanticSimilarityReward(RewardFunction):
    """Reward based on advanced semantic similarity."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._sentence_model = None
    
    def _get_sentence_model(self):
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.model_name or "all-MiniLM-L6-v2"
                self._sentence_model = SentenceTransformer(model_name)
            except ImportError:
                self._sentence_model = None
        return self._sentence_model
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None: return 0.0
        model = self._get_sentence_model()
        if model is None:
            text_words, ref_words = set(text.lower().split()), set(reference.lower().split())
            if not text_words or not ref_words: return 0.0
            return len(text_words & ref_words) / len(text_words | ref_words)
        try:
            embeddings = model.encode([text, reference])
            similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            return (similarity + 1) / 2
        except Exception:
            return 0.0
