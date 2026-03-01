"""Tests for the LiveTranscriber."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from wisker.transcriber import LiveTranscriber


class TestLiveTranscriber:
    def test_stop_terminates_generator(self):
        """Calling stop() causes run() to exit."""
        with patch("wisker.transcriber.sr") as mock_sr:
            mock_mic = MagicMock()
            mock_sr.Microphone.return_value = mock_mic
            mock_recognizer = MagicMock()
            mock_sr.Recognizer.return_value = mock_recognizer
            mock_sr.WaitTimeoutError = type("WaitTimeoutError", (Exception,), {})
            # Always timeout so no results
            mock_recognizer.listen.side_effect = mock_sr.WaitTimeoutError()

            transcriber = LiveTranscriber()
            results = []

            def consume():
                for text in transcriber.run():
                    results.append(text)

            t = threading.Thread(target=consume)
            t.start()
            time.sleep(0.3)
            transcriber.stop()
            t.join(timeout=5)

            assert not t.is_alive()

    def test_yields_recognised_text(self):
        """Transcribed phrases are yielded from run()."""
        with patch("wisker.transcriber.sr") as mock_sr:
            mock_mic = MagicMock()
            mock_sr.Microphone.return_value = mock_mic
            mock_recognizer = MagicMock()
            mock_sr.Recognizer.return_value = mock_recognizer
            mock_sr.WaitTimeoutError = type("WaitTimeoutError", (Exception,), {})
            mock_sr.UnknownValueError = type("UnknownValueError", (Exception,), {})
            mock_sr.RequestError = type("RequestError", (Exception,), {})

            call_count = 0

            def fake_listen(source, timeout=None, phrase_time_limit=None):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return MagicMock()  # valid audio
                raise mock_sr.WaitTimeoutError()

            mock_recognizer.listen.side_effect = fake_listen
            mock_recognizer.recognize_google.side_effect = ["hello world", "second phrase"]

            transcriber = LiveTranscriber()
            results = []

            def consume():
                for text in transcriber.run():
                    results.append(text)
                    if len(results) >= 2:
                        transcriber.stop()

            t = threading.Thread(target=consume)
            t.start()
            t.join(timeout=10)

            assert "hello world" in results
            assert "second phrase" in results

    def test_skips_unknown_value_errors(self):
        """UnknownValueError from recognizer is silently skipped."""
        with patch("wisker.transcriber.sr") as mock_sr:
            mock_mic = MagicMock()
            mock_sr.Microphone.return_value = mock_mic
            mock_recognizer = MagicMock()
            mock_sr.Recognizer.return_value = mock_recognizer
            mock_sr.WaitTimeoutError = type("WaitTimeoutError", (Exception,), {})
            mock_sr.UnknownValueError = type("UnknownValueError", (Exception,), {})
            mock_sr.RequestError = type("RequestError", (Exception,), {})

            call_count = 0

            def fake_listen(source, timeout=None, phrase_time_limit=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return MagicMock()
                raise mock_sr.WaitTimeoutError()

            mock_recognizer.listen.side_effect = fake_listen
            mock_recognizer.recognize_google.side_effect = mock_sr.UnknownValueError()

            transcriber = LiveTranscriber()
            results = []

            def consume():
                for text in transcriber.run():
                    results.append(text)

            t = threading.Thread(target=consume)
            t.start()
            time.sleep(0.5)
            transcriber.stop()
            t.join(timeout=5)

            assert results == []


class TestTranscriberInit:
    def test_default_params(self):
        with patch("wisker.transcriber.sr"):
            transcriber = LiveTranscriber()
            assert transcriber.phrase_time_limit == 10

    def test_custom_params(self):
        with patch("wisker.transcriber.sr"):
            transcriber = LiveTranscriber(phrase_time_limit=5, pause_threshold=2.0)
            assert transcriber.phrase_time_limit == 5
            assert transcriber.recognizer.pause_threshold == 2.0
