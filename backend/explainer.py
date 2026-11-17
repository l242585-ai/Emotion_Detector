"""
Explainability Module — word-level importance for emotion predictions.

Uses a LIME-inspired perturbation approach:
  1. Tokenise the input sentence into words.
  2. For each word, create a version of the sentence with that word removed.
  3. Measure how much the predicted-class probability drops.
  4. That drop is the word's "importance score."

The result is a list of highlighted words that the frontend can render
with colour-coded emphasis.

Member 1 (Annushay Fatima, 24L-2585) — Deliverable 3
"""

from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Colour tier thresholds (normalised scores are in [-1, 1])
_STRONG_POS   =  0.3
_MODERATE_POS =  0.1
_NEGATIVE     = -0.1


class EmotionExplainer:
    """
    Generates per-word importance scores for any model loaded by ModelService.

    Args:
        model_service : A ModelService instance with a working predict() method.
        num_perturbations (int): Reserved for future LIME sampling; currently
            each word is removed exactly once (deterministic leave-one-out).
    """

    def __init__(self, model_service, num_perturbations: int = 200):
        self.model_service    = model_service
        self.num_perturbations = num_perturbations   # kept for API compatibility

    # ── Public interface ─────────────────────────────────────────────────────

    def explain(self, text: str, model_name: str = "distilbert") -> dict:
        """
        Generate word-level importance scores for a single prediction.

        Args:
            text (str)       : Input sentence.
            model_name (str) : Model to explain ('baseline', 'bilstm', 'distilbert').

        Returns:
            dict: {
                'words'      : list[str],
                'scores'     : list[float],   # normalised to [-1, 1]
                'prediction' : dict,          # full ModelService.predict() result
                'highlights' : list[{word, score, color}]
            }
        """
        words      = text.split()
        base_pred  = self.model_service.predict(text, model_name)

        if not words:
            return self._empty_result(base_pred)

        base_conf  = base_pred["confidence"]
        emotion    = base_pred["emotion"]
        raw_scores = np.zeros(len(words), dtype=float)

        for idx in range(len(words)):
            perturbed = self._drop_word(words, idx)
            if not perturbed:
                # Single-word sentence: the word is 100 % responsible.
                raw_scores[idx] = base_conf
                continue
            pert_pred      = self.model_service.predict(perturbed, model_name)
            # Positive score  → removing this word hurts confidence (word supports prediction)
            # Negative score  → removing this word raises confidence  (word contradicts)
            raw_scores[idx] = base_conf - pert_pred["probabilities"][emotion]

        normalised = self._normalise(raw_scores)
        highlights = self._build_highlights(words, normalised)

        return {
            "words"     : words,
            "scores"    : [round(float(s), 4) for s in normalised],
            "prediction": base_pred,
            "highlights": highlights,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _drop_word(words: list[str], idx: int) -> str:
        """Return the sentence with the word at *idx* removed."""
        return " ".join(words[:idx] + words[idx + 1:])

    @staticmethod
    def _normalise(scores: np.ndarray) -> np.ndarray:
        """Normalise raw importance scores to [-1, 1]."""
        max_abs = max(np.abs(scores).max(), 1e-10)
        return scores / max_abs

    @staticmethod
    def _build_highlights(words: list[str], scores: np.ndarray) -> list[dict]:
        """Map each word to a colour tier based on its normalised score."""
        highlights = []
        for word, score in zip(words, scores):
            if score >= _STRONG_POS:
                color = "strong_positive"      # major positive contributor
            elif score >= _MODERATE_POS:
                color = "moderate_positive"    # minor positive contributor
            elif score <= _NEGATIVE:
                color = "negative"             # contradicts the prediction
            else:
                color = "neutral"              # negligible influence

            highlights.append({
                "word" : word,
                "score": round(float(score), 4),
                "color": color,
            })
        return highlights

    @staticmethod
    def _empty_result(base_pred: dict) -> dict:
        return {
            "words"     : [],
            "scores"    : [],
            "prediction": base_pred,
            "highlights": [],
        }
