"""Tests for the Wisker text cleanup pipeline."""

import pytest
from wisker.cleaner import clean, remove_fillers, remove_repetitions, resolve_corrections


class TestRemoveFillers:
    def test_removes_uh(self):
        assert clean("I uh need this") == "I need this"

    def test_removes_um(self):
        assert clean("Um let me think") == "let me think"

    def test_removes_hmm(self):
        assert clean("I need to write hmm a letter") == "I need to write a letter"

    def test_removes_like(self):
        assert clean("We need to like finish") == "We need to finish"

    def test_removes_you_know(self):
        assert clean("It was you know really good") == "It was really good"

    def test_removes_multiple_fillers(self):
        assert clean("I uh like um need this") == "I need this"

    def test_case_insensitive(self):
        assert clean("I UH need this") == "I need this"


class TestRemoveRepetitions:
    def test_single_word_repeat(self):
        assert clean("the the project") == "the project"

    def test_triple_repeat(self):
        assert clean("I I I need") == "I need"

    def test_double_word_at_end(self):
        assert clean("due on on Friday") == "due on Friday"

    def test_two_word_phrase_repeat(self):
        assert clean("I need I need to go") == "I need to go"


class TestSelfCorrections:
    def test_no_correction(self):
        result = clean("schedule a meeting at 6, no 5 o'clock")
        assert "5" in result
        assert "6" not in result

    def test_wait_correction(self):
        result = clean("meet on Monday, wait Tuesday")
        assert "Tuesday" in result

    def test_i_mean_correction(self):
        result = clean("call John, I mean Jane")
        assert "Jane" in result


class TestFullPipeline:
    def test_example_filler_removal(self):
        assert clean("I, uh, need to write hmm a letter") == "I, need to write a letter"

    def test_example_self_correction(self):
        result = clean("schedule a meeting at 6, no 5 o'clock")
        assert "5" in result

    def test_empty_string(self):
        assert clean("") == ""

    def test_clean_text_unchanged(self):
        text = "Schedule a meeting at 5 o'clock"
        assert clean(text) == text

    def test_complex_cleanup(self):
        raw = "We we need to like you know finish the the report by uh Friday"
        result = clean(raw)
        assert "like" not in result.lower().split()
        assert "uh" not in result.lower().split()
        assert "We need" in result
        assert "the report" in result
        assert "Friday" in result
