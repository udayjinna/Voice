import streamlit as st
from pathlib import Path
from uuid import uuid4
import asyncio

from app.emotion import get_detector
from app.voice import map_emotion_to_voice
import edge_tts

# Ensure audio folder exists
OUTPUT_DIR = Path("static/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Empathy Engine", page_icon="🎤", layout="wide")

st.title("🎤 Empathy Engine")
st.caption("Transform plain text into emotionally-aware expressive speech.")

# -----------------------
# Text Input
# -----------------------
text_input = st.text_area(
    "Enter text:",
    placeholder="Write something and the Empathy Engine will analyze its emotion and convert to expressive speech.",
    height=180
)

if st.button("Synthesize Voice"):
    if not text_input.strip():
        st.error("Please enter some text.")
    else:
        with st.spinner("Analyzing emotion..."):
            detector = get_detector()
            emotion_profile = detector.analyze(text_input)

        voice_profile = map_emotion_to_voice(emotion_profile)

        # -----------------------
        # Generate Audio
        # -----------------------
        with st.spinner("Generating expressive audio..."):
            filename = f"speech_{uuid4().hex}.mp3"
            output_path = OUTPUT_DIR / filename

            communicate = edge_tts.Communicate(
                text=text_input,
                voice=voice_profile.voice,
                rate=voice_profile.rate,
                pitch=voice_profile.pitch,
                volume=voice_profile.volume,
            )

            asyncio.run(communicate.save(str(output_path)))

        st.success("Speech synthesized successfully!")

        # -----------------------
        # Show Emotion Summary
        # -----------------------
        st.subheader("Detected Emotion")
        st.markdown(
            f"""
            **Emotion:** `{emotion_profile.label}`  
            **Intensity:** `{emotion_profile.intensity * 100:.0f}%`
            """
        )

        # -----------------------
        # Emotion Confidence Table
        # -----------------------
        st.subheader("Emotion Confidence Scores")
        emo_scores = emotion_profile.canonical_scores
        st.table({k: f"{v*100:.1f}%" for k, v in emo_scores.items()})

        # -----------------------
        # Voice Profile
        # -----------------------
        st.subheader("Voice Profile")
        st.json({
            "Voice": voice_profile.voice,
            "Rate": voice_profile.rate,
            "Pitch": voice_profile.pitch,
            "Volume": voice_profile.volume,
            "Style": voice_profile.style,
        })

        # -----------------------
        # Show Player + Download Button
        # -----------------------
        st.subheader("Audio Preview")
        audio_bytes = output_path.read_bytes()
        st.audio(audio_bytes, format="audio/mp3")

        st.download_button(
            label="Download Audio",
            data=audio_bytes,
            file_name=filename,
            mime="audio/mpeg"
        )
