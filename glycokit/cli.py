"""
Command line functionality of glyco.

The `main()` function of this module gets the argument parser from each of the
scripts below and executes the `main()` function of the module called. The
function also collects help from the subparsers and provides it at the command
line.
"""

from importlib.metadata import version
import argparse
import subprocess
import importlib
import sys

# __version__ = version("basicrta")

# define which scripts can be ran from cli
# can easily add functionality to cli as modules are added
# commands = ['contacts', 'gibbs', 'cluster', 'combine', 'kinetics', 'assess']

def main():
    """ This module provides the functionality for a command line interface for
    glyco scripts.
    The scripts available to the cli are:

    Each script is called and ran using the `main()` function of each module
    and the parser is passed to the cli using the `get_parser()` function. Any
    module added to the cli needs to have both functions.
    """
    from parsers import base_parser

    parser = base_parser()
    print(f'parser: {parser}')
    # subparsers = parser.add_subparsers(help="""step in the glyco workflow to
                                    #    execute""")

    # collect parser from each script in `commands`
    # for command in commands:
    #     subparser = importlib.import_module(f"glyco.{command}").get_parser()
    #     subparsers.add_parser(f'{command}', parents=[subparser], add_help=True,
    #                           description=subparser.description,
    #                           conflict_handler='resolve',
    #                           formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    #                           help=subparser.description)

    # print subparser help if no arguments given
    # if len(sys.argv) == 2 and sys.argv[1] in commands:
        # subparsers.choices[f'{sys.argv[1]}'].print_help()
        # sys.exit()

    # print basicrta help if no subcommand given
    # parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # execute basicrta script
    # importlib.import_module(f"glyco").main()

if __name__ == "__main__":
    main()
