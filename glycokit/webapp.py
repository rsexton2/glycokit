"""Launcher for the GlyCo Streamlit app (``glyco app``)."""

import importlib.util
import subprocess
import sys
from importlib import resources

USAGE = """\
usage: glyco app [STREAMLIT OPTIONS]

Start the GlyCo web app in a local Streamlit server and open it in the
browser. Options are passed to 'streamlit run', for example:

  glyco app --server.port 8502
  glyco app --server.maxUploadSize 2000
  glyco app --server.headless true      (do not open a browser)

Stop the server with Ctrl+C.
"""


def main(argv=None):
    """
    Start the Streamlit app.

    :param argv: Streamlit options (default: ``sys.argv[1:]``).
    :type argv: list[str] or None
    :return: Exit status of the Streamlit server.
    :rtype: int
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    if importlib.util.find_spec("streamlit") is None:
        print(
            "glyco app: error: Streamlit is not installed; install it with "
            "'pip install streamlit' or 'pip install \"glycokit[app]\"'",
            file=sys.stderr,
        )
        return 1

    script = resources.files("glycokit") / "app.py"
    with resources.as_file(script) as path:  # a real file for the server
        cmd = [sys.executable, "-m", "streamlit", "run", str(path), *argv]
        try:
            return subprocess.call(cmd)
        except KeyboardInterrupt:  # Ctrl+C stops the server
            return 130
