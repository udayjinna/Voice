"""Voice modulation logic mapping detected emotions to TTS settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from .emotion import EmotionProfile


@dataclass(frozen=True)
class VoiceProfile:
    """Encapsulates prosody parameters for the TTS engine."""

    voice: str
    rate: str
    pitch: str
    volume: str
    style: str | None = None

    def as_edge_tts_kwargs(self) -> Dict[str, str]:
        return {
            "voice": self.voice,
            "rate": self.rate,
            "pitch": self.pitch,
            "volume": self.volume,
        }


EDGE_VOICE = "en-US-AriaNeural"

EMOTION_BASE_PARAMS: Mapping[str, Dict[str, float]] = {
    "positive": {"rate_pct": 18.0, "pitch_hz": 35.0, "volume_pct": 4.0, "style": "cheerful"},
    "negative": {"rate_pct": -12.0, "pitch_hz": -30.0, "volume_pct": -2.0, "style": "sad"},
    "neutral": {"rate_pct": 0.0, "pitch_hz": 0.0, "volume_pct": 0.0},
    "surprised": {"rate_pct": 24.0, "pitch_hz": 50.0, "volume_pct": 6.0, "style": "excited"},
    "inquisitive": {"rate_pct": 8.0, "pitch_hz": 18.0, "volume_pct": 2.0, "style": "chat"},
}


def map_emotion_to_voice(profile: EmotionProfile, *, voice: str = EDGE_VOICE) -> VoiceProfile:
    """Translate an emotion profile into a `VoiceProfile` for TTS synthesizers."""

    base = EMOTION_BASE_PARAMS.get(profile.label, EMOTION_BASE_PARAMS["neutral"])
    intensity = max(0.0, min(1.0, profile.intensity))
    intensity_multiplier = 0.6 + intensity * 0.8  # keep low-energy emotions audible

    rate = _format_percent(base["rate_pct"] * intensity_multiplier)
    pitch = _format_hz(base["pitch_hz"] * intensity_multiplier)
    volume = _format_percent(base["volume_pct"] * intensity_multiplier)
    style = base.get("style")

    return VoiceProfile(voice=voice, rate=rate, pitch=pitch, volume=volume, style=style)


def _format_percent(value: float) -> str:
    """Edge TTS expects percent deltas like '+15%' or '-10%'."""

    return f"{value:+.0f}%"


def _format_hz(value: float) -> str:
    return f"{value:+.0f}Hz"


__all__ = ["VoiceProfile", "map_emotion_to_voice", "EDGE_VOICE"]
