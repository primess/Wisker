"""LLM-powered document processor using GitHub Models (no OpenAI key needed).

Maintains a running document and interprets each spoken phrase in context:
- New content is appended / integrated
- Commands like "change the second item" edit the document
- Filler words and repetitions are cleaned automatically
"""

from __future__ import annotations

import os
import subprocess

from openai import OpenAI

SYSTEM_PROMPT = """\
You are Wisker, a smart note-taking assistant that builds a document from spoken input.

You will receive:
1. The CURRENT DOCUMENT (may be empty at the start)
2. A new SPOKEN PHRASE from the user

Your job:
- If the phrase adds new content, integrate it into the document naturally.
- If the phrase is a command or edit instruction (e.g. "change the second item to milk", \
"delete the last line", "move the first item to the end"), apply that edit to the document.
- Always remove filler words (uh, um, hmm, like, you know, I mean, etc.).
- Remove stutters and repetitions.
- When the speaker self-corrects ("at 6, no 5 o'clock"), keep only the correction.
- Keep the document well-structured. Use numbered lists, bullet points, or paragraphs as appropriate.
- Preserve the overall meaning and intent.

IMPORTANT RULES:
- Return ONLY the updated document text. No explanations, no commentary, no markdown code fences.
- If the spoken phrase is just noise or silence markers, return the document unchanged.
- Do NOT add any prefix like "Updated document:" — just return the raw document content.
"""


def _get_github_token() -> str:
    """Get a GitHub token from env or gh CLI."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise EnvironmentError(
        "No GitHub token found.\n"
        "Either set GITHUB_TOKEN or log in with: gh auth login"
    )


def get_client() -> OpenAI:
    """Create an OpenAI-compatible client pointed at GitHub Models."""
    token = _get_github_token()
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )


class DocumentProcessor:
    """Maintains a running document and processes spoken phrases via LLM."""

    def __init__(self, model: str = "gpt-5-mini"):
        self.client = get_client()
        self.model = model
        self.document = ""
        self.history: list[str] = []  # raw phrases for debugging

    def process(self, spoken_phrase: str) -> str:
        """Process a spoken phrase and return the updated document.

        Args:
            spoken_phrase: Raw transcribed text from the microphone.

        Returns:
            The full updated document text.
        """
        self.history.append(spoken_phrase)

        user_msg = f"CURRENT DOCUMENT:\n{self.document}\n\n---\n\nSPOKEN PHRASE:\n{spoken_phrase}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=2048,
        )

        self.document = response.choices[0].message.content.strip()
        return self.document
