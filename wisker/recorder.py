"""Microphone audio recording for live transcription."""

import threading
from collections.abc import Generator

CHUNK_SIZE = 1024
FORMAT_WIDTH = 2  # 16-bit
CHANNELS = 1
SAMPLE_RATE = 16000
RECORD_SECONDS_PER_CHUNK = 5  # Transcribe every N seconds


def record_chunks() -> Generator[bytes, None, None]:
    """Record audio from the microphone and yield chunks for transcription.

    Yields PCM audio data in chunks of RECORD_SECONDS_PER_CHUNK seconds.
    Runs until the generator is closed (e.g., via KeyboardInterrupt).
    """
    try:
        import pyaudio
    except ImportError:
        raise ImportError(
            "PyAudio is required for live recording.\n"
            "Install it with: pip install pyaudio\n"
            "On macOS you may need: brew install portaudio && pip install pyaudio"
        )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    try:
        while True:
            frames = []
            num_chunks = int(SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS_PER_CHUNK)
            for _ in range(num_chunks):
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
            yield b"".join(frames)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
