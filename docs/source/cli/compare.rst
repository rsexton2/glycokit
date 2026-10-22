compare.py (development)
========================

Compares ``glyco --mode v1/v2`` with the legacy GLYCO tools. Value options
default to ``GLYCO_COMPARE_<NAME>`` environment variables.

.. argparse::
   :module: glycokit.parsers
   :func: compare_parser
   :prog: compare.py