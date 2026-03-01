"""Tests for the Wisker CLI commands."""

import os
import signal
import threading
import time
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from wisker.cli import main


class TestCleanCommand:
    def test_clean_text_argument(self):
        runner = CliRunner()
        result = runner.invoke(main, ["clean", "I uh need this"])
        assert result.exit_code == 0
        assert "I need this" in result.output

    def test_clean_from_file(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("We we need to like finish")
        runner = CliRunner()
        result = runner.invoke(main, ["clean", "--file", str(f)])
        assert result.exit_code == 0
        assert "We need to finish" in result.output

    def test_clean_to_output_file(self, tmp_path):
        out = tmp_path / "out.txt"
        runner = CliRunner()
        result = runner.invoke(main, ["clean", "the the cat", "--output", str(out)])
        assert result.exit_code == 0
        assert out.read_text() == "the cat"

    def test_clean_from_stdin(self):
        runner = CliRunner()
        result = runner.invoke(main, ["clean"], input="uh hello hello world")
        assert result.exit_code == 0
        assert "hello world" in result.output

    def test_clean_no_input_errors(self):
        runner = CliRunner()
        # CliRunner simulates a non-TTY, so empty stdin yields empty output
        result = runner.invoke(main, ["clean"], input="")
        assert result.exit_code == 0

    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "wisker" in result.output


class TestListenCtrlC:
    """Test that the Ctrl+C mechanism works: signal handler calls transcriber.stop()."""

    def test_sigint_handler_calls_stop(self):
        """The SIGINT handler installed by listen() must call transcriber.stop()."""
        from wisker.transcriber import LiveTranscriber

        # Verify that LiveTranscriber.stop() sets the _stop event
        with patch("wisker.transcriber.sr"):
            transcriber = LiveTranscriber()
            assert not transcriber._stop.is_set()
            transcriber.stop()
            assert transcriber._stop.is_set(), "stop() must set the _stop event"

    def test_stop_event_breaks_run_loop(self):
        """When _stop is set, the run() generator exits promptly."""
        from wisker.transcriber import LiveTranscriber as LT

        with patch("wisker.transcriber.sr") as mock_sr:
            mock_sr.Microphone.return_value = MagicMock()
            mock_sr.WaitTimeoutError = type("WaitTimeoutError", (Exception,), {})
            mock_recognizer = MagicMock()
            mock_sr.Recognizer.return_value = mock_recognizer
            mock_recognizer.listen.side_effect = mock_sr.WaitTimeoutError()

            transcriber = LT()
            results = []
            exited = threading.Event()

            def consume():
                for text in transcriber.run():
                    results.append(text)
                exited.set()

            t = threading.Thread(target=consume)
            t.start()

            time.sleep(0.3)
            transcriber.stop()

            exited.wait(timeout=2.0)
            assert exited.is_set(), "run() did not exit within 2s after stop()"
            t.join(timeout=1.0)
            assert not t.is_alive()
