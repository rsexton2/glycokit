import collections
import contextlib
import logging
import os
import sys

from glycokit.config import (
    _WARNING_LINE,
)
from glycokit.console import (
    style,
    warning,
)


class _WarningSummary(logging.Handler):
    """Count captured Python warnings for an end-of-run console summary."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.counts = collections.Counter()

    def emit(self, record):
        lines = record.getMessage().splitlines()
        first = lines[0] if lines else ""
        m = _WARNING_LINE.match(first)
        self.counts[f"{m['cat']}: {m['msg']}" if m else first] += 1

    @property
    def total(self):
        """Number of warnings counted."""
        return sum(self.counts.values())


@contextlib.contextmanager
def warning_log(path):
    """
    Log warnings to ``path`` while active and print a summary on exit.

    The summary is printed also if the body raises (including
    ``SystemExit`` from ``parser.error``).

    :param str path: Log file path; its directory must exist.
    :return: The warning counter (see :func:`configure_warning_log`).
    :rtype: _WarningSummary
    """
    summary = configure_warning_log(path)
    try:
        yield summary
    finally:
        print_warning_summary(summary, path)


def configure_warning_log(path):
    """
    Route Python warnings (including MDAnalysis') into a log file and count them.

    Calling it again with the same path adds no second file handler. The
    returned counter is reset on every call.

    :param str path: Log file path.
    :return: Counter for :func:`print_warning_summary`.
    :rtype: _WarningSummary
    """
    logging.captureWarnings(True)
    logger = logging.getLogger("py.warnings")
    logger.setLevel(logging.WARNING)
    logger.propagate = False

    abspath = os.path.abspath(path)
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == abspath
        for h in logger.handlers
    ):
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)

    summary = next((h for h in logger.handlers if isinstance(h, _WarningSummary)), None)
    if summary is None:
        summary = _WarningSummary()
        logger.addHandler(summary)
    summary.counts.clear()
    return summary


def print_warning_summary(summary, log_path, max_lines=5, width=200):
    """
    Print a short summary of the warnings of a run to stderr.

    :param summary: Counter from :func:`configure_warning_log`, or None.
    :type summary: _WarningSummary or None
    :param str log_path: Log file holding the full warnings.
    :param int max_lines: Most frequent distinct warnings listed.
    :param int width: Maximum characters per listed warning.
    """
    if summary is None or summary.total == 0:
        return
    warning(
        f"{summary.total} warning(s) during the run; details in "
        f"{style(log_path, 'cyan', stream=sys.stderr)}"
    )
    for message, count in summary.counts.most_common(max_lines):
        text = message if len(message) <= width else message[: width - 3] + "..."
        times = style(f" (x{count})", "dim", stream=sys.stderr) if count > 1 else ""
        print(f"    - {text}{times}", file=sys.stderr)
    rest = len(summary.counts) - max_lines
    if rest > 0:
        print(
            style(
                f"    ... and {rest} more distinct warning(s)", "dim", stream=sys.stderr
            ),
            file=sys.stderr,
        )
