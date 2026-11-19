"""
Progress reporting shared by the glyco subcommands.

Three outputs, one object:

* a tqdm bar (MDAnalysis' ProgressBar, so it matches AnalysisBase.run)
* the ``GLYCO_PROGRESS_FILE`` JSON read by the Streamlit front end
* nothing, under --progress off / GLYCO_NO_PROGRESS=1 / GLYCO_STREAMLIT=1

Imports only the standard library, MDAnalysis and glycokit.console, so it
stays usable from utils.py and the analysis classes without a cycle.
"""

import contextlib
import json
import os
import sys

from MDAnalysis.lib.log import ProgressBar

from glycokit.console import is_tty

#: switches every bar off (batch jobs, tests)
NO_PROGRESS_ENV = "GLYCO_NO_PROGRESS"
#: machine-readable progress JSON (owned by the caller, not read here)
PROGRESS_FILE_ENV = "GLYCO_PROGRESS_FILE"
#: set by the Streamlit front end: no bar, but the JSON is still written
STREAMLIT_ENV = "GLYCO_STREAMLIT"


def progress_verbose(args):
    """Progress verbosity from parsed arguments: True / False / None (auto)."""
    return {
        "on": True,
        "off": False,
        "auto": None
    }[getattr(args, "progress", "auto")]


def progress_mode(verbose=None, stream=None):
    """
    ``'bar'`` or ``'off'`` for the given verbosity.

    ``verbose`` None means auto: a bar only on a terminal. The "1" tests
    match console.use_color and glyco.main.
    """
    env = os.environ
    if env.get(NO_PROGRESS_ENV) == "1" or env.get(STREAMLIT_ENV) == "1":
        return "off"
    if verbose is False:
        return "off"
    if verbose:
        return "bar"
    return "bar" if is_tty(stream or sys.stderr) else "off"


def write_progress_file(path, current, total, stage=None):
    """
    Progress JSON written atomically (moved here from GlycoAnalysis).

    The ``current`` / ``total`` / ``percent`` keys keep their meaning;
    ``stage`` is additional and optional.
    """
    if not path:
        return
    payload = {
        "current": int(current),
        "total": int(total),
        "percent": int(100 * current / total) if total else 0,
    }
    if stage:
        payload["stage"] = stage
    temp = f"{path}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(temp, path)


def mda_progressbar_kwargs(desc, verbose=None):
    """
    ``progressbar_kwargs`` for a SERIAL ``AnalysisBase.run``.

    MDAnalysis honours this only for the serial backend, which is why the
    parallel path needs a backend of its own.
    """
    return {
        "desc": desc,
        "unit": "frame",
        "leave": False,
        "disable": progress_mode(verbose) != "bar",
    }


class Progress:
    """
    One progress bar, optionally mirrored into a progress JSON file.

    ``progress_file`` is explicit on purpose: GlycoAnalysis already owns
    ``GLYCO_PROGRESS_FILE``, and two writers would race on it.
    """

    def __init__(
            self, total=None, desc="", unit="it", *, verbose=None, stream=None,
            leave=False, progress_file=None):
        self.total = None if total is None else int(total)
        self.desc = desc
        self.unit = unit
        self.n = 0
        self._stream = stream or sys.stderr
        self._file = progress_file or None
        self._mode = progress_mode(verbose, self._stream)
        self._every = max(1, (self.total or 100) // 100)  # as _write_progress
        self._bar = ProgressBar(
            total=self.total, desc=desc or None, unit=unit, leave=leave,
            file=self._stream, dynamic_ncols=True, verbose=self._mode == "bar")
        self._write_file()

    def _write_file(self):
        write_progress_file(self._file, self.n, self.total or 0, self.desc)

    def update(self, n=1):
        """Move the bar by ``n`` units."""
        n = int(n)
        if n <= 0:
            return
        self.n += n
        self._bar.update(n)
        if self.n % self._every == 0 or self.n == self.total:
            self._write_file()

    def set_total(self, total):
        """Set the total once it is known (e.g. after run() set n_frames)."""
        self.total = None if total is None else int(total)
        self._every = max(1, (self.total or 100) // 100)
        self._bar.total = self.total
        self._bar.refresh()
        self._write_file()

    def set_description(self, desc):
        self.desc = desc
        self._bar.set_description(desc or None)

    def log(self, message):
        """Write a line without breaking the bar."""
        if self._mode == "bar":
            ProgressBar.write(message, file=self._stream)
        else:
            print(message, file=self._stream, flush=True)

    @contextlib.contextmanager
    def pause(self):
        """Keep the bar out of the way of console output (stdout or stderr)."""
        if self._mode != "bar":
            yield
            return
        with ProgressBar.external_write_mode(file=self._stream):
            yield

    def close(self):
        self._bar.close()
        self._write_file()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def parallel_map(
        func, items, *, n_workers=1, desc="", unit="task", verbose=None):
    """
    ``[func(item) for item in items]`` in a process pool, with one bar.

    One worker or one item runs in this process, so a serial run pays for
    no pool. Results keep the order of ``items``; the first exception is
    re-raised in the parent.
    """
    items = list(items)
    if not items:
        return []
    n_workers = max(1, int(n_workers))

    with Progress(len(items), desc, unit, verbose=verbose) as bar:
        if n_workers == 1 or len(items) == 1:
            out = []
            for item in items:
                out.append(func(item))
                bar.update(1)
            return out

        from multiprocessing import (
            Pool,  # pylint: disable=import-outside-toplevel
        )
        with Pool(processes=n_workers) as pool:
            pending = [pool.apply_async(func, (item, )) for item in items]
            out = []
            for result in pending:
                out.append(result.get())
                bar.update(1)
        return out
