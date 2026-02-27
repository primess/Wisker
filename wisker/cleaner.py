"""Core text cleanup pipeline for Wisker.

Transforms raw speech transcripts into clean, readable text by removing
filler words, repetitions, and resolving self-corrections.
"""

import re

# Filler words and verbal tics to remove
FILLERS = [
    r"\buh\b",
    r"\buhh+\b",
    r"\bum\b",
    r"\bumm+\b",
    r"\bhmm+\b",
    r"\bhm\b",
    r"\blike\b,?",
    r"\byou know\b,?",
    r"\bi mean\b,?",
    r"\bsort of\b",
    r"\bkind of\b",
    r"\bbasically\b,?",
    r"\bactually\b,?",
    r"\bliterally\b,?",
    r"\bright\b\??,?(?=\s)",
    r"\bokay so\b,?",
    r"\bso yeah\b,?",
    r"\byeah so\b,?",
    r"\banyway\b,?",
    r"\banyways\b,?",
]

# Patterns for self-corrections: "X, no Y" / "X, I mean Y" / "X, wait Y" / "X, actually Y"
CORRECTION_PATTERNS = [
    # "at 6, no 5 o'clock" -> "at 5 o'clock"
    r"(\b\w+(?:\s+\w+){0,2}),?\s*\bno\b,?\s+((?:\w+(?:\s+\w+){0,3}))",
    # "at 6, wait 5 o'clock" -> "at 5 o'clock"
    r"(\b\w+(?:\s+\w+){0,2}),?\s*\bwait\b,?\s+((?:\w+(?:\s+\w+){0,3}))",
    # "on Monday, I mean Tuesday" -> "on Tuesday"
    r"(\b\w+(?:\s+\w+){0,2}),?\s*\bI mean\b,?\s+((?:\w+(?:\s+\w+){0,3}))",
    # "at 6, actually 5 o'clock" -> "at 5 o'clock"
    r"(\b\w+(?:\s+\w+){0,2}),?\s*\bactually\b,?\s+((?:\w+(?:\s+\w+){0,3}))",
    # "X, or rather Y" -> "Y"
    r"(\b\w+(?:\s+\w+){0,2}),?\s*\bor rather\b,?\s+((?:\w+(?:\s+\w+){0,3}))",
]


def remove_fillers(text: str) -> str:
    """Remove filler words and verbal tics from text."""
    for pattern in FILLERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Clean up comma pairs left behind: ", ," or ",," -> ","
    text = re.sub(r",\s*,", ",", text)
    return text


def remove_repetitions(text: str) -> str:
    """Collapse repeated words and short phrases.

    "the the project" -> "the project"
    "I I I need" -> "I need"
    "on on Friday" -> "on Friday"
    """
    # Single word repetitions: "the the the" -> "the"
    text = re.sub(r"\b(\w+)(?:\s+\1)+\b", r"\1", text, flags=re.IGNORECASE)

    # Two-word phrase repetitions: "I need I need to" -> "I need to"
    text = re.sub(r"\b(\w+\s+\w+)(?:\s+\1)+\b", r"\1", text, flags=re.IGNORECASE)

    return text


def resolve_corrections(text: str) -> str:
    """Handle self-corrections where the speaker corrects themselves.

    "at 6, no 5 o'clock" -> "at 5 o'clock"
    "on Monday, I mean Tuesday" -> "on Tuesday"
    """
    for pattern in CORRECTION_PATTERNS:
        text = re.sub(pattern, r"\2", text, flags=re.IGNORECASE)
    return text


def normalize_whitespace(text: str) -> str:
    """Clean up extra whitespace and fix punctuation spacing."""
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    # Collapse duplicate punctuation (e.g., ",," -> ",")
    text = re.sub(r"([.,!?;:])\1+", r"\1", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # Remove spaces after opening punctuation
    text = re.sub(r"([(\[\"'])\s+", r"\1", text)
    # Ensure space after punctuation (but not for abbreviations)
    text = re.sub(r"([.,!?;:])(\w)", r"\1 \2", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Collapse any remaining multi-spaces
    text = re.sub(r"  +", " ", text)
    return text


def clean(text: str) -> str:
    """Run the full Wisker cleanup pipeline on a text transcript.

    Pipeline order matters:
    1. Resolve self-corrections first (before fillers are stripped)
    2. Remove filler words
    3. Remove repetitions
    4. Normalize whitespace
    """
    text = resolve_corrections(text)
    text = remove_fillers(text)
    text = remove_repetitions(text)
    text = normalize_whitespace(text)
    return text
