"""Wisker CLI — smart speech-to-text cleanup."""

import os
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
@click.option("--output", "-o", "output_path", type=click.Path(), help="Save final document to a file")
@click.option("--clipboard/--no-clipboard", default=True, help="Copy result to clipboard")
@click.option("--model", "-m", default="gpt-4o-mini", help="GitHub Models model name")
def listen(output_path: str | None, clipboard: bool, model: str):
    """Start live transcription from your microphone.

    Speak naturally — Wisker uses an LLM to build a smart document from
    your speech. It understands context, so you can say things like
    "change the second item to milk" and it will edit your notes.

    Press Ctrl+C to stop.
    """
    from wisker.transcriber import listen_and_transcribe
    from wisker.processor import DocumentProcessor

    console.print("[bold cyan]🎙️  Wisker is listening...[/bold cyan] (Ctrl+C to stop)")
    console.print(f"[dim]Using GitHub Models ({model}) • speech via Google free API[/dim]\n")

    try:
        processor = DocumentProcessor(model=model)
    except EnvironmentError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    try:
        for raw_text in listen_and_transcribe():
            console.print(f"  [dim]heard:[/dim] {raw_text}")
            document = processor.process(raw_text)
            console.print()
            console.print(Panel(document, title="📝 Document", border_style="green"))
            console.print()
    except KeyboardInterrupt:
        pass

    final = processor.document

    if final:
        console.print(f"\n[bold green]── Final Document ──[/bold green]\n{final}\n")

        if output_path:
            with open(output_path, "w") as f:
                f.write(final + "\n")
            console.print(f"[green]✓[/green] Saved to {output_path}")

        if clipboard:
            try:
                import pyperclip
                pyperclip.copy(final)
                console.print("[dim]📋 Copied to clipboard[/dim]")
            except Exception:
                console.print("[dim]⚠️  Could not copy to clipboard[/dim]")
    else:
        console.print("\n[yellow]No speech detected.[/yellow]")


# Register the clean command with the right name
main.add_command(clean_cmd, "clean")

if __name__ == "__main__":
    main()
