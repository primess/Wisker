"""Audio transcription using OpenAI Whisper API."""

import io
import os
import tempfile
import wave

from openai import OpenAI


def get_client() -> OpenAI:
    """Create an OpenAI client, raising a clear error if no API key is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )
    return OpenAI(api_key=api_key)


def transcribe_audio(audio_data: bytes, sample_rate: int = 16000, channels: int = 1) -> str:
    """Transcribe raw audio bytes using OpenAI Whisper.

    Args:
        audio_data: Raw PCM audio bytes.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.

    Returns:
        Raw transcription text.
    """
    client = get_client()

    # Write raw PCM to a WAV buffer
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    wav_buffer.seek(0)
    wav_buffer.name = "audio.wav"

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_buffer,
        response_format="text",
    )
    return response.strip()


def transcribe_file(file_path: str) -> str:
    """Transcribe an audio file using OpenAI Whisper.

    Args:
        file_path: Path to an audio file (wav, mp3, m4a, etc.).

    Returns:
        Raw transcription text.
    """
    client = get_client()

    with open(file_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    return response.strip()
