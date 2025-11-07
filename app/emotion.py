"""Emotion detection utilities for the Empathy Engine."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Tuple

from transformers import Pipeline, pipeline


DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Map raw model labels to canonical categories used by the Empathy Engine.
EMOTION_CANONICAL_MAP: Mapping[str, str] = {
    "joy": "positive",
    "love": "positive",
    "optimism": "positive",
    "trust": "positive",
    "admiration": "positive",
    "amusement": "positive",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "pessimism": "negative",
    "disappointment": "negative",
    "guilt": "negative",
    "remorse": "negative",
    "neutral": "neutral",
    "surprise": "surprised",
    "curiosity": "inquisitive",
}

# Ensure we always expose at least the required categories, even if the model
# does not emit them for a particular input.
CANONICAL_FALLBACKS: Tuple[str, ...] = ("positive", "negative", "neutral")


@dataclass(frozen=True)
class EmotionProfile:
    """Structured representation of detected emotion and its intensity."""

    label: str
    intensity: float
    canonical_scores: Dict[str, float]
    raw_scores: Dict[str, float]


class EmotionDetector:
    """Wraps a Hugging Face pipeline for emotion classification."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: int | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline: Pipeline | None = None

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = pipeline(
                task="text-classification",
                model=self.model_name,
                top_k=None,
                device=self.device if self.device is not None else -1,
            )
        return self._pipeline

    def analyze(self, text: str) -> EmotionProfile:
        """Return the dominant emotion, its intensity, and score breakdown."""

        normalized_text = text.strip()
        if not normalized_text:
            scores = {label: 1.0 if label == "neutral" else 0.0 for label in CANONICAL_FALLBACKS}
            return EmotionProfile(label="neutral", intensity=0.0, canonical_scores=scores, raw_scores={"neutral": 1.0})

        raw_distribution = self._score_text(normalized_text)
        canonical_scores = self._canonicalize_scores(raw_distribution)

        dominant_label, dominant_score = max(canonical_scores.items(), key=lambda item: item[1])
        intensity = float(max(0.0, min(1.0, dominant_score)))

        return EmotionProfile(
            label=dominant_label,
            intensity=intensity,
            canonical_scores=canonical_scores,
            raw_scores=raw_distribution,
        )

    def _score_text(self, text: str) -> Dict[str, float]:
        raw_output = self.pipeline(text, top_k=None)[0]

        if isinstance(raw_output, dict):
            iterable = [raw_output]
        else:
            iterable = list(raw_output)

        scores: Dict[str, float] = {}
        for item in iterable:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label", "")).lower()
            score = item.get("score")
            if not label or score is None:
                continue
            scores[label] = float(score)

        if not scores:
            raise RuntimeError("Emotion model returned no scores")

        return scores

    def _canonicalize_scores(self, raw_scores: Mapping[str, float]) -> Dict[str, float]:
        canonical: Dict[str, float] = {}
        for label, score in raw_scores.items():
            canonical_label = EMOTION_CANONICAL_MAP.get(label, label)
            canonical[canonical_label] = canonical.get(canonical_label, 0.0) + score

        # Ensure all fallback categories exist so downstream consumers can rely on them.
        for fallback in CANONICAL_FALLBACKS:
            canonical.setdefault(fallback, 0.0)

        total = sum(canonical.values()) or 1.0
        return {label: value / total for label, value in canonical.items()}


@lru_cache(maxsize=1)
def get_detector(model_name: str = DEFAULT_MODEL) -> EmotionDetector:
    """Convenience accessor for a cached detector instance."""

    return EmotionDetector(model_name=model_name)


__all__: List[str] = ["EmotionDetector", "EmotionProfile", "get_detector"]
