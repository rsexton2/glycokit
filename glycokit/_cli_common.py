"""
Shared building blocks for the glyco command-line parsers.

Standard library only: imported by parsers.py and compare_parsers.py,
and must not import any other glycokit module.
"""

import argparse
import re
import shlex
import sys

COLOR_CHOICES = ("auto", "always", "never")


# ------------------------------------------------------------- parser classes
class _HelpOnNoArgsParser(argparse.ArgumentParser):
    """
    Print help and exit if invoked with NO arguments via sys.argv.

    Does NOT intervene when an explicit args list is passed in, even an
    empty one: ``parser.parse_args([])`` is used by GlycoAnalysis to read
    the defaults.
    """

    def parse_args(self, args=None, namespace=None):
        if args is None:
            argv = sys.argv[1:]
            if not argv:
                self.print_help(sys.stderr)
                self.exit(1)
            args = argv
        return super().parse_args(args, namespace)


class _GlycoHelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Keep description/epilog formatting and append "(default: ...)"."""


# ------------------------------------------------------------- argparse types
def non_negative_int(text):
    """argparse type: an integer >= 0."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def chain_id(text):
    """argparse type: a single-character PDB chain ID (A-Z, a-z, 0-9)."""
    text = str(text).strip()
    if len(text) != 1 or not text.isalnum() or not text.isascii():
        raise argparse.ArgumentTypeError(
            f"must be one character A-Z, a-z or 0-9, got {text!r}"
        )
    return text


def segid(text):
    """argparse type: a PDB segment ID (1-4 characters, columns 73-76)."""
    text = str(text).strip()
    if not 1 <= len(text) <= 4:
        raise argparse.ArgumentTypeError(f"must be 1-4 characters, got {text!r}")
    return text


def resname_pair(text):
    """argparse type: 'OLD=NEW' -> ('OLD', 'NEW'); NEW has at most 4 characters."""
    old, sep, new = str(text).partition("=")
    old, new = old.strip(), new.strip()
    if not sep or not old or not new:
        raise argparse.ArgumentTypeError(f"expected OLD=NEW, got {text!r}")
    if len(new) > 4:
        raise argparse.ArgumentTypeError(
            f"new residue name {new!r} is longer than 4 characters"
        )
    return old, new


def positive_float(text):
    """argparse type: a float > 0 (rejects nan)."""
    value = float(text)
    if not value > 0:  # pylint: disable=unnecessary-negation # excludes NaN
        raise argparse.ArgumentTypeError(f"must be positive, got {text!r}")
    return value


def non_negative_float(text):
    """argparse type: a float >= 0 (rejects nan)."""
    value = float(text)
    if not value >= 0:  # pylint: disable=unnecessary-negation # excludes NaN
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text!r}")
    return value


def n_workers_int(text):
    """argparse type: -1 (all processors) or an integer >= 1."""
    value = int(text)
    if value != -1 and value < 1:
        raise argparse.ArgumentTypeError(f"must be -1 or >= 1, got {value}")
    return value


def name_list(text):
    """argparse type: 'BGL,BMA AMA' -> ('BGL', 'BMA', 'AMA'), duplicates removed."""
    names = tuple(dict.fromkeys(n for n in re.split(r"[,\s]+", str(text)) if n))
    if not names:
        raise argparse.ArgumentTypeError("no names given")
    return names


def command_line(text):
    """argparse type: split a command like a shell does -> tuple of words."""
    words = tuple(shlex.split(str(text)))
    if not words:
        raise argparse.ArgumentTypeError("empty command")
    return words


#: Options shared by ``glyco run`` and ``compare.py``, keyed by their
#: :data:`glycokit.config.COMPARE_DEFAULTS` name. Each entry holds the
#: flags and parsing settings; every parser gives its own help text and
#: default (and may override ``dest``).
SHARED_OPTIONS = {
    "PROTEIN_SELECTION": {
        "flags": ("--ps", "--prot-sel", "--protein-selection"),
        "dest": "prot_sel",
        "type": str,
        "metavar": "SELECTION",
    },
    "GLYCAN_SELECTION": {
        "flags": ("--gs", "--gly-sel", "--glycan-selection"),
        "dest": "gly_sel",
        "type": str,
        "metavar": "SELECTION",
    },
    "DISTANCE_CUTOFF": {
        "flags": ("--dc", "--distance-cutoff"),
        "dest": "distance_cutoff",
        "type": positive_float,
        "metavar": "ANGSTROM",
    },
    "CYLINDER_RADIUS": {
        "flags": ("--cr", "--cylinder-radius"),
        "dest": "cylinder_radius",
        "type": positive_float,
        "metavar": "ANGSTROM",
    },
    "AREA_CUTOFF": {
        "flags": ("--ac", "--area-cutoff"),
        "dest": "area_cutoff",
        "type": positive_float,
        "metavar": "ANGSTROM^2",
    },
    "PROBE": {
        "flags": ("--pr", "--probe-radius", "--probe"),
        "dest": "sasa_probe",
        "type": positive_float,
        "metavar": "ANGSTROM",
    },
    "N_WORKERS": {
        "flags": ("-j", "--n-workers"),
        "dest": "n_workers",
        "type": n_workers_int,
        "metavar": "N",
    },
}


def add_option(group, flags, text, **kwargs):
    """Add one option to ``group`` with help text ``text``."""
    group.add_argument(*flags, help=text, **kwargs)


def add_shared_option(group, key, text, **overrides):
    """
    Add the :data:`SHARED_OPTIONS` option ``key`` to ``group``.

    :param overrides: argparse settings that replace or extend the shared
        ones, usually ``default`` and, for compare.py, ``dest``.
    """
    spec = dict(SHARED_OPTIONS[key])
    flags = spec.pop("flags")
    spec.update(overrides)
    add_option(group, flags, text, **spec)


def add_color(group):
    """Add ``--color`` to ``group``."""
    add_option(
        group,
        ("--color",),
        "color terminal output: auto (only on a terminal, and not when "
        "NO_COLOR is set), always, or never",
        dest="color",
        choices=COLOR_CHOICES,
        default="auto",
    )


def glyco_parser(prog, description, epilog):
    """Parser of a glyco subcommand: shared formatter and help behaviour."""
    return _HelpOnNoArgsParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=_GlycoHelpFormatter,
        allow_abbrev=False,
    )
