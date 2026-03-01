"""Audio transcription using SpeechRecognition (Google free API — no key needed)."""

from __future__ import annotations

import signal
from collections.abc import Generator

import speech_recognition as sr


def listen_and_transcribe(
    phrase_time_limit: int = 10,
    pause_threshold: float = 1.5,
    listen_timeout: int = 30,
) -> Generator[str, None, None]:
    """Record from the microphone and yield raw transcription strings.

    Each yield is one recognised phrase. Runs until the generator is closed
    (typically via KeyboardInterrupt).

    Args:
        phrase_time_limit: Max seconds per phrase before forcing a recognition attempt.
        pause_threshold: Seconds of silence that mark the end of a phrase.
        listen_timeout: Max seconds to wait for speech before cycling (allows Ctrl+C).
    """
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = pause_threshold
    recognizer.dynamic_energy_threshold = True

    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

    while True:
        with mic as source:
            try:
                audio = recognizer.listen(
                    source,
                    timeout=listen_timeout,
                    phrase_time_limit=phrase_time_limit,
                )
            except sr.WaitTimeoutError:
                # No speech detected within timeout — loop back so Ctrl+C can fire
                continue

        try:
            text = recognizer.recognize_google(audio)
            if text:
                yield text
        except sr.UnknownValueError:
            pass
        except sr.RequestError as exc:
            yield f"[recognition error: {exc}]"


def transcribe_file(file_path: str) -> str:
    """Transcribe an audio file using Google's free speech recognition.

    Args:
        file_path: Path to a WAV/AIFF/FLAC audio file.

    Returns:
        Raw transcription text.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)
