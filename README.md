# The Empathy Engine

Transform plain text into expressive, emotionally aware speech using FastAPI, Hugging Face emotion models, and Microsoft's neural voices.

## ✨ Features

- Emotion detection across multiple categories (positive, negative, neutral, surprised, inquisitive)
- Intensity-aware modulation of speech rate, pitch, and volume
- Online neural voice synthesis via `edge-tts`
- Web interface with real-time audio preview and downloadable output
- REST API endpoint for programmatic access

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Internet access (first run downloads the Hugging Face model and uses the Edge TTS service)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the App

```bash
uvicorn app.main:app --reload
```

Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to open the Empathy Engine UI.

## 🧠 How It Works

1. **Emotion Detection** — The Hugging Face model `j-hartmann/emotion-english-distilroberta-base` classifies the text into granular emotions.
2. **Canonical Mapping** — Raw labels (e.g., `joy`, `sadness`, `surprise`) are aggregated into canonical categories (`positive`, `negative`, `neutral`, `surprised`, `inquisitive`).
3. **Intensity Scaling** — The confidence score for the dominant emotion influences the magnitude of vocal adjustments.
4. **Voice Modulation** — Rate, pitch, and volume offsets are derived from the canonical emotion and intensity, then translated into Edge TTS parameters.
5. **Speech Synthesis** — `edge-tts` streams the audio into `static/audio/` and the UI exposes a player plus download link.

### Emotion → Voice Mapping

| Canonical Emotion | Base Rate Δ | Base Pitch Δ (Hz) | Base Volume Δ | Style Hint |
| ----------------- | ----------- | ----------------- | ------------- | ---------- |
| Positive          | +18%        | +35Hz             | +4%           | cheerful   |
| Negative          | −12%        | −30Hz             | −2%           | sad        |
| Neutral           | +0%         | +0Hz              | +0%           | —          |
| Surprised         | +24%        | +50Hz             | +6%           | excited    |
| Inquisitive       | +8%         | +18Hz             | +2%           | chat       |

> The runtime intensity score (0–1) scales each delta by 60–140% to keep subtle emotions audible while accentuating strong sentiments.

## 🔌 REST API

`POST /api/synthesize`

```json
{
  "text": "We're thrilled to share your approval came through!"
}
```

Response:

```json
{
  "emotion": "positive",
  "intensity": 0.82,
  "canonical_scores": {
    "positive": 0.82,
    "neutral": 0.12,
    "negative": 0.06
  },
  "audio_url": "/static/audio/speech_<id>.mp3"
}
```

## 📝 Notes & Limitations

- The Hugging Face model will download on first use; cache it locally if deploying in production.
- `edge-tts` relies on Microsoft's online voices and requires outbound network access.
- Generated files accumulate under `static/audio/`; implement a cleanup policy for long-running deployments.
- Ensure that your deployment respects the usage terms of both Hugging Face models and Microsoft voices.

## 📄 License

MIT-licensed template. Adapt as needed for your organization or competitions.
