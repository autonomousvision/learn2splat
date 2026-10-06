"""Shared rich console for consistent, professional terminal output.

A single `CONSOLE` instance keeps styling uniform across the codebase. rich
degrades gracefully when stdout is not a tty (e.g. SLURM log files): no
animations, plain box-drawing characters — still fully readable.

Usage:
    from learn2splat.misc.console import CONSOLE, banner, rule, warn

    banner("learn2splat", ["host  gpu-node-01", "mode  test"])
    rule("Testing scene 3: room_0")
    warn("Skipping batch 7 due to OOM")
"""
from __future__ import annotations

import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

LEARN2SPLAT_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "metric": "bold magenta",
        "path": "blue underline",
        "muted": "dim",
    }
)

# Off-tty (SLURM logs, pipes) rich defaults to width 80, which wraps config
# rows awkwardly. Pin a wider, fixed width there; let interactive terminals
# auto-size so output adapts to the real window.
_console_kwargs: dict = {"theme": LEARN2SPLAT_THEME}
if not sys.stdout.isatty():
    _console_kwargs["width"] = 120

CONSOLE = Console(**_console_kwargs)


def banner(title: str, lines: list[str] | None = None, style: str = "info") -> None:
    """Print a titled panel; `lines` form the panel body."""
    body = "\n".join(lines) if lines else ""
    CONSOLE.print(
        Panel(body, title=title, title_align="left", border_style=style, expand=False)
    )


def rule(title: str, style: str = "info") -> None:
    """Print a horizontal section divider with a centered title."""
    CONSOLE.rule(f"[{style}]{title}[/{style}]", style=style)


def warn(msg: str) -> None:
    """Print a styled warning line."""
    CONSOLE.print(f"[warning]⚠ {msg}[/warning]")


def error(msg: str) -> None:
    """Print a styled error line."""
    CONSOLE.print(f"[error]✖ {msg}[/error]")


def success(msg: str) -> None:
    """Print a styled success line."""
    CONSOLE.print(f"[success]✔ {msg}[/success]")


def metrics_table(
    rows: list[tuple], headers: list[str], title: str | None = None
) -> None:
    """Print a metrics table. `rows` is a list of tuples, one per row."""
    table = Table(title=title, header_style="bold", box=box.SIMPLE_HEAD)
    for h in headers:
        table.add_column(str(h))
    for row in rows:
        table.add_row(*[str(c) for c in row])
    CONSOLE.print(table)


def config_table(
    sections: dict[str, list[tuple[str, str]]], title: str = "Config"
) -> None:
    """Print a grouped key/value table.

    `sections` maps a group name to a list of (key, value) pairs. Empty groups
    are skipped; group names appear as styled separator rows.
    """
    table = Table(title=title, box=box.SIMPLE, show_header=False, pad_edge=False)
    table.add_column("key", style="muted", no_wrap=True)
    table.add_column("value")
    first = True
    for group, pairs in sections.items():
        if not pairs:
            continue
        if not first:
            table.add_row("", "")
        first = False
        table.add_row(f"[info]{group}[/info]", "")
        for key, value in pairs:
            table.add_row(f"  {key}", str(value))
    CONSOLE.print(table)
