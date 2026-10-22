"""
Terminal styling for glyco output.

Colors are used only on a terminal, unless overridden: ``NO_COLOR``
disables them, ``FORCE_COLOR`` / ``CLICOLOR_FORCE`` force them (``NO_COLOR``
wins), and ``--color always/never`` overrides both. Output to files, pipes
and the Streamlit app (``GLYCO_STREAMLIT=1``) stays plain.
"""

import os
import sys
from pathlib import Path

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
}
_MODE = "auto"  # auto | always | never


def set_color_mode(mode):
    """
    Set the color mode for this process.

    :param str mode: ``'auto'``, ``'always'`` or ``'never'``.
    :raises ValueError: On an unknown mode.
    """
    global _MODE
    if mode not in ("auto", "always", "never"):
        raise ValueError(f"color mode must be auto, always or never, got {mode!r}")
    _MODE = mode


def use_color(stream=None):
    """
    Whether ANSI colors should be written to ``stream`` (default stdout).

    :rtype: bool
    """
    stream = sys.stdout if stream is None else stream
    if _MODE == "never":
        return False
    if _MODE == "always":
        return True
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE"):
        return True
    if os.environ.get("GLYCO_STREAMLIT") == "1":
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError):  # closed or replaced streams
        return False


def style(text, *names, stream=None):
    """
    ``text`` wrapped in ANSI codes (e.g. ``style(x, 'bold', 'red')``), or
    unchanged if colors are off for ``stream``.

    :rtype: str
    """
    text = str(text)
    if not names or not use_color(stream):
        return text
    return f"\x1b[{';'.join(_CODES[n] for n in names)}m{text}\x1b[0m"


def _line(tag, color, message, stream):
    print(
        f"{style(f'[{tag}]', 'bold', color, stream=stream)} {message}",
        file=stream,
        flush=True,
    )


def info(message):
    """``[info] message`` on stdout (blue tag)."""
    _line("info", "blue", message, sys.stdout)


def warning(message):
    """``[warning] message`` on stderr (yellow tag)."""
    _line("warning", "yellow", message, sys.stderr)


def error(message):
    """``[error] message`` on stderr (red tag)."""
    _line("error", "red", message, sys.stderr)


def heading(text, stream=None):
    """A bold section heading."""
    stream = sys.stdout if stream is None else stream
    print(style(text, "bold", stream=stream), file=stream, flush=True)


def diagnostic(*parts):
    """Dimmed diagnostic line (``-v``)."""
    print(style(" ".join(map(str, parts)), "dim"), flush=True)


def written(label, path, n_rows=None, base=None):
    """
    One aligned line per output file::

        ok  Residue summary          residue_summary.csv  (212 rows)

    :param str label: What was written.
    :param path: File path; shown relative to ``base`` when inside it.
    :param n_rows: Row count, or None.
    :param base: Directory the path is shown relative to.
    """
    path = Path(path)
    if base is not None:
        try:
            path = path.resolve().relative_to(Path(base).resolve())
        except ValueError:
            pass
    rows = f"  ({n_rows:,} rows)" if n_rows is not None else ""
    print(
        f"  {style('ok', 'green', 'bold')}  {label:<26} "
        f"{style(path, 'cyan')}{style(rows, 'dim')}",
        flush=True,
    )
