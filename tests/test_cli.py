"""Tests for the Wisker CLI commands."""

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
