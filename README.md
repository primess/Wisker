# 🐱 Wisker

**Smart speech-to-text cleanup.** Wisker listens to what you say and writes down what you *mean*.

It captures your speech via microphone (or processes existing transcripts) and intelligently cleans the output — removing filler words, false starts, repetitions, and self-corrections — so you get clean, readable notes.

## Examples

| You say | Wisker writes |
|---|---|
| "I, uh, need to write hmm a letter" | "I need to write a letter" |
| "Schedule a meeting at 6, no 5 o'clock" | "Schedule a meeting at 5 o'clock" |
| "We need to, like, you know, finish the report" | "We need to finish the report" |
| "The the the project is due on on Friday" | "The project is due on Friday" |

## Features

- 🎙️ **Live transcription** — speak into your mic and get clean text in real time
- 🧹 **Filler removal** — strips out *uh, um, hmm, like, you know, I mean*, etc.
- 🔁 **Repetition removal** — collapses stutters and repeated words/phrases
- ✏️ **Self-correction handling** — when you say "6, no 5", Wisker keeps the correction
- 📋 **Clipboard mode** — cleaned text copied straight to your clipboard
- 📝 **File mode** — process an existing transcript file

## Installation

```bash
pip install -e .
```

### Requirements

- Python 3.10+
- A working microphone (for live mode)
- An OpenAI API key (for Whisper transcription)

```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Live transcription

```bash
wisker listen
```

Speak into your microphone. Press `Ctrl+C` to stop. Cleaned text is printed and copied to your clipboard.

### Clean a transcript from text

```bash
wisker clean "I, uh, need to write hmm a letter"
# Output: I need to write a letter
```

### Clean a transcript from a file

```bash
wisker clean --file transcript.txt --output cleaned.txt
```

## How It Works

1. **Capture** — Records audio from your microphone in chunks
2. **Transcribe** — Sends audio to OpenAI Whisper for raw transcription
3. **Clean** — Applies Wisker's text cleanup pipeline:
   - Remove filler words and verbal tics
   - Collapse repeated words and phrases
   - Resolve self-corrections (keeps the corrected version)
   - Normalize whitespace and punctuation
4. **Output** — Returns clean, readable text

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check wisker/
```

## License

MIT
