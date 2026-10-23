"""
glyco command-line entry point (GlyCo, GLYcan COverage).

Subcommands::

    glyco run ...          glycan coverage analysis (glycokit.glyco)
    glyco standardize ...  normalise structure inputs (glycokit.standardize)

Everything after the subcommand is passed unchanged to that module's
``main(argv)``, so each subcommand keeps its own options and ``--help``.
"""

import difflib
import importlib
import sys
from importlib import metadata

from glycokit.console import error, style

# subcommand -> (module with main(argv), one-line description)
COMMANDS = {
    "run": ("glycokit.glyco", "glycan coverage of protein surface residues"),
    "standardize": (
        "glycokit.standardize",
        "normalise structure inputs for glyco / GLYCO",
    ),
    "app": ("glycokit.app", "run the glyco web application"),
}


def _version():
    """Installed glycokit version, or 'unknown'."""
    try:
        return metadata.version("glycokit")
    except metadata.PackageNotFoundError:
        return "unknown"


def _usage():
    """Top-level help text."""
    width = max(map(len, COMMANDS))
    lines = ["usage: glyco <command> [options]", "", "commands:"]
    lines += [
        f"  {style(f'{name:<{width}}', 'bold')}  {text}"
        for name, (_, text) in COMMANDS.items()
    ]
    lines += ["", "Run 'glyco <command> --help' for the options of a command."]
    return "\n".join(lines)


def main(argv=None):
    """
    Dispatch to a subcommand.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :type argv: list[str] or None
    :return: Exit status of the subcommand (0 on success).
    :rtype: int
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(_usage(), file=sys.stdout if argv else sys.stderr)
        return 0 if argv else 1
    if argv[0] in ("-V", "--version"):
        print(f"glyco (glycokit) {_version()}")
        return 0

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        close = difflib.get_close_matches(command, COMMANDS, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        error(f"unknown command {command!r}{hint}")
        print(f"\n{_usage()}", file=sys.stderr)
        return 2

    # 'glyco run' / 'glyco standardize' without options: show that command's help
    if not rest:
        rest = ["--help"]

    module = importlib.import_module(COMMANDS[command][0])  # lazy: only what is used

    # argparse parsers without an explicit prog name it after sys.argv[0]
    old_argv0 = sys.argv[0]
    sys.argv[0] = f"glyco {command}"
    try:
        status = module.main(rest)
    except ValueError as e:
        if command != "standardize":
            raise
        # standardize reports invalid input as ValueError; show it without a traceback
        print(f"glyco standardize: error: {e}", file=sys.stderr)
        return 1
    finally:
        sys.argv[0] = old_argv0
    return 0 if status is None else int(status)


if __name__ == "__main__":
    sys.exit(main())
