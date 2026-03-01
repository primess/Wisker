"""Audio transcription using SpeechRecognition (Google free API — no key needed)."""

from __future__ import annotations

import threading
from collections.abc import Generator

import speech_recognition as sr


class LiveTranscriber:
    """Threaded live transcriber that responds to Ctrl+C reliably.

    Uses a background thread for blocking mic I/O so the main thread
    can always handle signals.
    """

    def __init__(
        self,
        phrase_time_limit: int = 10,
        pause_threshold: float = 1.5,
    ):
        self.phrase_time_limit = phrase_time_limit
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True
        self._stop = threading.Event()
        self._results: list[str] = []
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop.set()

    def _listen_loop(self) -> None:
        """Background thread: record and recognise in a loop."""
        mic = sr.Microphone()
        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

        while not self._stop.is_set():
            with mic as source:
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=2,
                        phrase_time_limit=self.phrase_time_limit,
                    )
                except sr.WaitTimeoutError:
                    continue

            if self._stop.is_set():
                break

            try:
                text = self.recognizer.recognize_google(audio)
                if text:
                    with self._lock:
                        self._results.append(text)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as exc:
                with self._lock:
                    self._results.append(f"[recognition error: {exc}]")

    def run(self) -> Generator[str, None, None]:
        """Start listening and yield transcribed phrases.

        The generator exits cleanly when stop() is called.
        """
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()

        try:
            while not self._stop.is_set():
                self._stop.wait(timeout=0.1)
                with self._lock:
                    batch = list(self._results)
                    self._results.clear()
                for text in batch:
                    yield text
        finally:
            self.stop()
            thread.join(timeout=3)


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
