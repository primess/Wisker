"""Wisker CLI — smart speech-to-text cleanup."""

import sys

import click
from rich.console import Console
from rich.panel import Panel

from wisker import __version__
from wisker.cleaner import clean

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="wisker")
def main():
    """🐱 Wisker — captures what you say, writes what you mean."""
    pass


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="Input transcript file")
@click.option("--output", "-o", "output_path", type=click.Path(), help="Output file (default: stdout)")
def clean_cmd(text: str | None, file_path: str | None, output_path: str | None):
    """Clean up a speech transcript.

    Pass text directly or use --file to read from a file.
    """
    if text:
        source = text
    elif file_path:
        with open(file_path) as f:
            source = f.read()
    elif not sys.stdin.isatty():
        source = sys.stdin.read()
    else:
        console.print("[red]Error:[/red] Provide text, --file, or pipe input via stdin.")
        raise SystemExit(1)

    result = clean(source)

    if output_path:
        with open(output_path, "w") as f:
            f.write(result)
        console.print(f"[green]✓[/green] Cleaned text written to {output_path}")
    else:
        console.print(Panel(result, title="Wisker Output", border_style="green"))


@main.command()
@click.option("--clipboard/--no-clipboard", default=True, help="Copy result to clipboard")
def listen(clipboard: bool):
    """Start live transcription from your microphone.

    Speak naturally — Wisker will clean up the output in real time.
    Press Ctrl+C to stop.
    """
    from wisker.recorder import record_chunks, SAMPLE_RATE, CHANNELS
    from wisker.transcriber import transcribe_audio

    console.print("[bold cyan]🎙️ Wisker is listening...[/bold cyan] (Ctrl+C to stop)\n")

    all_cleaned: list[str] = []

    try:
        for audio_chunk in record_chunks():
            raw_text = transcribe_audio(audio_chunk, sample_rate=SAMPLE_RATE, channels=CHANNELS)
            if raw_text:
                cleaned = clean(raw_text)
                if cleaned:
                    console.print(f"  {cleaned}")
                    all_cleaned.append(cleaned)
    except KeyboardInterrupt:
        pass

    full_text = " ".join(all_cleaned)

    if full_text:
        console.print(f"\n[bold green]── Final Output ──[/bold green]\n{full_text}\n")
        if clipboard:
            try:
                import pyperclip
                pyperclip.copy(full_text)
                console.print("[dim]📋 Copied to clipboard[/dim]")
            except Exception:
                console.print("[dim]⚠️  Could not copy to clipboard[/dim]")
    else:
        console.print("\n[yellow]No speech detected.[/yellow]")


# Register the clean command with the right name
main.add_command(clean_cmd, "clean")

if __name__ == "__main__":
    main()
