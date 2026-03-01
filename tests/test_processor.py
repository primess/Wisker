"""Tests for the DocumentProcessor (LLM integration, mocked)."""

from unittest.mock import MagicMock, patch, call

import pytest

from wisker.processor import DocumentProcessor, SYSTEM_PROMPT, _get_github_token


class TestGetGithubToken:
    def test_from_env_var(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
        assert _get_github_token() == "ghp_test123"

    def test_from_gh_cli(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_result = MagicMock(returncode=0, stdout="gho_clitoken\n")
        with patch("wisker.processor.subprocess.run", return_value=mock_result):
            assert _get_github_token() == "gho_clitoken"

    def test_raises_when_no_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_result = MagicMock(returncode=1, stdout="")
        with patch("wisker.processor.subprocess.run", return_value=mock_result):
            with pytest.raises(EnvironmentError, match="No GitHub token found"):
                _get_github_token()

    def test_raises_when_gh_not_found(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("wisker.processor.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(EnvironmentError):
                _get_github_token()


class TestDocumentProcessor:
    @patch("wisker.processor.get_client")
    def test_process_returns_document(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Buy eggs"))]
        mock_client.chat.completions.create.return_value = mock_response

        proc = DocumentProcessor(model="test-model")
        result = proc.process("buy eggs please")

        assert result == "Buy eggs"
        assert proc.document == "Buy eggs"
        assert proc.history == ["buy eggs please"]

    @patch("wisker.processor.get_client")
    def test_process_sends_current_document(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First call
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock(message=MagicMock(content="1. Eggs"))]
        # Second call
        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock(message=MagicMock(content="1. Eggs\n2. Milk"))]
        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]

        proc = DocumentProcessor(model="test-model")
        proc.process("eggs")
        result = proc.process("add milk")

        assert result == "1. Eggs\n2. Milk"
        # Verify the second call included the document from the first call
        second_call = mock_client.chat.completions.create.call_args_list[1]
        user_msg = second_call.kwargs["messages"][1]["content"]
        assert "1. Eggs" in user_msg
        assert "add milk" in user_msg

    @patch("wisker.processor.get_client")
    def test_process_uses_system_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test"))]
        mock_client.chat.completions.create.return_value = mock_response

        proc = DocumentProcessor()
        proc.process("hello")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT

    @patch("wisker.processor.get_client")
    def test_process_uses_specified_model(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        proc = DocumentProcessor(model="gpt-5")
        proc.process("test")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5"

    @patch("wisker.processor.get_client")
    def test_history_tracks_all_phrases(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="doc"))]
        mock_client.chat.completions.create.return_value = mock_response

        proc = DocumentProcessor()
        proc.process("first phrase")
        proc.process("second phrase")
        proc.process("third phrase")

        assert proc.history == ["first phrase", "second phrase", "third phrase"]

    @patch("wisker.processor.get_client")
    def test_empty_document_at_start(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        proc = DocumentProcessor()
        assert proc.document == ""
        assert proc.history == []

    @patch("wisker.processor.get_client")
    def test_no_temperature_param_sent(self, mock_get_client):
        """gpt-5-mini rejects 'temperature' — verify we don't send it."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        proc = DocumentProcessor()
        proc.process("test")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "temperature" not in call_kwargs, (
            f"'temperature' should not be sent — gpt-5-mini rejects it. Got: {call_kwargs}"
        )

    @patch("wisker.processor.get_client")
    def test_no_max_tokens_param_sent(self, mock_get_client):
        """gpt-5-mini rejects 'max_tokens' — only 'max_completion_tokens' is valid."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        proc = DocumentProcessor()
        proc.process("test")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs, (
            f"'max_tokens' should not be sent — use max_completion_tokens. Got: {call_kwargs}"
        )

    @patch("wisker.processor.get_client")
    def test_only_safe_api_params(self, mock_get_client):
        """Only model, messages, and max_completion_tokens should be sent."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        proc = DocumentProcessor()
        proc.process("test")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        allowed = {"model", "messages", "max_completion_tokens"}
        extra = set(call_kwargs.keys()) - allowed
        assert not extra, f"Unexpected API params {extra} — may cause 400 errors"
