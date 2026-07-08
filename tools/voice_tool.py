"""
Voice Tool — Groq Whisper for speech-to-text, gTTS for text-to-speech.
"""

import os
import tempfile
from groq import Groq
from gtts import gTTS


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def speech_to_text(audio_bytes: bytes, filename: str = "recording.wav") -> str:
    """
    Transcribe audio using Groq Whisper.
    
    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename (for format detection)
    
    Returns:
        Transcribed text string
    """
    client = get_client()
    
    try:
        # Write to temp file for Groq API
        suffix = os.path.splitext(filename)[1] if "." in filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language="en",
                response_format="text"
            )
        
        os.unlink(tmp_path)
        return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"


def text_to_speech(text: str, lang: str = "en") -> bytes:
    """
    Convert text to speech using gTTS.
    
    Args:
        text: Text to convert to speech
        lang: Language code (default: English)
    
    Returns:
        Audio bytes (MP3 format)
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tts.save(tmp.name)
            tmp_path = tmp.name
        
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        
        os.unlink(tmp_path)
        return audio_bytes
        
    except Exception as e:
        raise Exception(f"Error generating speech: {str(e)}")


def generate_alert_audio(alert_message: str) -> bytes:
    """Generate audio alert for municipal teams."""
    prefix = "Attention! CleanAir Alert System. "
    full_message = prefix + alert_message
    return text_to_speech(full_message)
