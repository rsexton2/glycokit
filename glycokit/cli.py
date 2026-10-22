"""
Command line functionality of glyco.

The `main()` function of this module gets the argument parser from each of the
scripts below and executes the `main()` function of the module called. The
function also collects help from the subparsers and provides it at the command
line.
"""


# __version__ = version("basicrta")

# define which scripts can be ran from cli
# can easily add functionality to cli as modules are added
# commands = ['contacts', 'gibbs', 'cluster', 'combine', 'kinetics', 'assess']


def main():
    """This module provides the functionality for a command line interface for
    glyco scripts.
    The scripts available to the cli are:

    Each script is called and ran using the `main()` function of each module
    and the parser is passed to the cli using the `get_parser()` function. Any
    module added to the cli needs to have both functions.
    """
    from parsers import base_parser

    parser = base_parser()
    print(f"parser: {parser}")


if __name__ == "__main__":
    main()
