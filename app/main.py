"""FastAPI application for the Empathy Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import edge_tts
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .emotion import EmotionProfile, get_detector
from .voice import VoiceProfile, map_emotion_to_voice


OUTPUT_DIR = Path("static/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Empathy Engine", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _build_context(
    request: Request,
    text: str,
    emotion: EmotionProfile,
    voice_profile: VoiceProfile,
    audio_filename: Optional[str],
) -> Dict[str, object]:
    audio_url = f"/static/audio/{audio_filename}" if audio_filename else None
    return {
        "request": request,
        "input_text": text,
        "emotion": emotion,
        "voice_profile": voice_profile,
        "canonical_scores": sorted(emotion.canonical_scores.items(), key=lambda item: item[1], reverse=True),
        "audio_url": audio_url,
    }


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to transform into expressive speech")


class SynthesisResponse(BaseModel):
    emotion: str
    intensity: float
    canonical_scores: Dict[str, float]
    audio_url: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/synthesize", response_class=HTMLResponse)
async def synthesize_form(request: Request, text: str = Form(...)) -> HTMLResponse:
    detector = get_detector()
    emotion_profile = detector.analyze(text)
    voice_profile = map_emotion_to_voice(emotion_profile)

    filename = await _generate_audio_file(text, voice_profile)
    context = _build_context(request, text, emotion_profile, voice_profile, filename)
    return templates.TemplateResponse("index.html", context)


@app.post("/api/synthesize", response_model=SynthesisResponse)
async def synthesize_api(payload: SynthesisRequest) -> SynthesisResponse:
    detector = get_detector()
    emotion_profile = detector.analyze(payload.text)
    voice_profile = map_emotion_to_voice(emotion_profile)

    filename = await _generate_audio_file(payload.text, voice_profile)

    return SynthesisResponse(
        emotion=emotion_profile.label,
        intensity=emotion_profile.intensity,
        canonical_scores=emotion_profile.canonical_scores,
        audio_url=f"/static/audio/{filename}",
    )


async def _generate_audio_file(text: str, voice_profile: VoiceProfile) -> str:
    filename = f"speech_{uuid4().hex}.mp3"
    output_path = OUTPUT_DIR / filename

    communicate_kwargs = voice_profile.as_edge_tts_kwargs()
    communicate = edge_tts.Communicate(text=text, **communicate_kwargs)
    try:
        await communicate.save(str(output_path))
    except Exception as exc:  # pragma: no cover - network and I/O errors
        if output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=500, detail=f"Audio synthesis failed: {exc}") from exc

    return filename


__all__ = ["app"]
