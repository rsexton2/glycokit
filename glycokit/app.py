"""
Streamlit dashboard for glyco (GlyCo, GLYcan COverage).

Sidebar: load files, atom selections. Main area: system metrics, a viewer
(input trajectory played by Mol* after loading; result structures colored
by glyco values with py3Dmol after a run) next to the run controls
(frames, parameters generated from :func:`glycokit.parsers.base_parser`,
run), and the result tables below.
"""

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
from datetime import datetime

import MDAnalysis as mda
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from glycokit.config import _LOAD_ERRORS as LOAD_ERRORS
from glycokit.config import _SELECTION_ERRORS as SELECTION_ERRORS
from glycokit.parsers import base_parser
from glycokit.standardize import glycan_selection, guess_glycans

try:  # optional: Mol* viewer for the input trajectory
    from streamlit_molstar import st_molstar
except ImportError:
    st_molstar = None

try:  # optional: py3Dmol viewer for the result structures
    import py3Dmol
except ImportError:
    py3Dmol = None

# st.fragment (Streamlit >= 1.37) reruns only the viewer on its own widgets
fragment = getattr(st, "fragment", lambda func: func)

# =====================================================================
# Configuration
# =====================================================================
# Run outputs: one subdirectory per run below this directory. Relative paths
# are resolved against the directory `streamlit run` was started from.
RESULTS_ROOT = os.environ.get("GLYCO_APP_RESULTS", "glyco_results")

# --- scroll / zoom ---------------------------------------------------------
INVERT_SCROLL_ZOOM = False  # reverse the mouse-wheel zoom in both viewers
SCROLL_DOWN_ZOOMS_IN = True  # py3Dmol: flip this if the direction is still wrong
ZOOM_PER_PIXEL = 1.002  # py3Dmol: zoom factor per pixel of wheel movement

# Errors from the glycan guesser on unusual inputs
GUESS_ERRORS = (ValueError, IndexError, KeyError)

# Errors from starting the glyco subprocess and its temporary files
RUN_ERRORS = (OSError, ValueError, subprocess.SubprocessError)

# Errors from writing the viewer files with MDAnalysis
VIEWER_ERRORS = (OSError, ValueError, TypeError, IndexError)

GLYCO_COMMAND = [sys.executable, "-m", "glycokit.glyco"]

PARSER = base_parser()

# Linear color scales: evenly spaced color stops from t = 0 to t = 1
COLOR_SCALES = {
    "White → Red": ("#ffffff", "#d7191c"),
    "White → Blue": ("#ffffff", "#2c7bb6"),
    "White → Green": ("#ffffff", "#1a9641"),
    "White → Purple": ("#ffffff", "#7b3294"),
    "Grey → Red": ("#d9d9d9", "#d7191c"),
    "Blue → White → Red": ("#2c7bb6", "#ffffff", "#d7191c"),
}

# Parser destinations set by the sidebar, the frame tab or the app itself;
# they are not shown as parameter widgets.
HANDLED_DESTS = frozenset(
    {
        "top",
        "traj",
        "indir",
        "in_types",  # sidebar: input files
        "prot_sel",
        "gly_sel",  # sidebar: selections
        "frame_start",
        "frame_stop",
        "frame_step",
        "frame_list",  # frames tab
        "outdir",
        "force",
        "verbose",
        "color",  # set by the app (output goes to a log file, not a terminal)
    }
)

# argparse type functions by name -> widget type
_TYPE_NAMES = {
    "int": "int",
    "float": "float",
    "str": "str",
    "positive_float": "float",
    "non_negative_float": "float",
    "n_workers_int": "int",
}

# Lower bound of the number widget per argparse type function
_MIN_VALUES = {"positive_float": 1e-6, "non_negative_float": 0.0, "n_workers_int": -1}

SLICE_MODE = "Start / Stop / Step"
LIST_MODE = "Frame list"

# --- input viewer (Mol*) -------------------------------------------------
MOLSTAR_HEIGHT = "600px"
VIEWER_MAX_FRAMES = 500  # frames sent to the browser at most
VIEWER_TRAJ_SUFFIX = ".xtc"  # compact; use ".dcd" if your Mol* build needs it
VIEW_INPUT = "input"

# --- result viewer (py3Dmol) --------------------------------------------
RESULT_PREFIX = "result:"
RESULT_WIDTH = 900  # pixels; py3Dmol needs a fixed width
RESULT_HEIGHT = 600
RESULT_STYLES = ("Cartoon", "Surface", "Sticks")
GRADIENTS = ("roygb", "rwb", "sinebow")

# Residue names colored as protein (the rest is drawn as grey sticks)
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL", "CYX", "ASH", "GLH", "LYN",
    "HSD", "HSE", "HSP", "HID", "HIE", "HIP",
]  # fmt: skip

# Hover label: residue and B-factor of the atom under the mouse
HOVER_JS = """function(atom, viewer) {
  if (!atom.label) {
    atom.label = viewer.addLabel(
      atom.resn + " " + atom.resi + " " + atom.chain + ": " + atom.b.toFixed(1),
      {position: atom, backgroundColor: "black", fontColor: "white", fontSize: 12});
  }
}"""
UNHOVER_JS = """function(atom, viewer) {
  if (atom.label) { viewer.removeLabel(atom.label); delete atom.label; }
}"""

# Result structures: file name -> (label, description)
RESULT_STRUCTURES = {
    "b_factors.pdb": (
        "Result: per-residue glycan counts",
        "B-factor column: glyco's per-residue result for the frame chosen "
        "with --bfactor-frame (default: the last analyzed frame).",
    ),
    "b_factors_stats.pdb": (
        "Result: trajectory statistics",
        "B-factor: histogram maximum of the glycan count; occupancy: width "
        "of its 95% interval (written with --write-stats).",
    ),
}

# Result tables: files listed first, and the number of rows shown
PREFERRED_TABLES = ("residue_summary.csv", "residue_statistics.csv")
MAX_TABLE_ROWS = 5000


# Re-dispatches every trusted wheel event in the Mol* frame with a negated
# delta. The re-dispatched event is untrusted, so it passes through unchanged.
MOLSTAR_INVERT_ZOOM_HTML = """
<script>
(function () {
  function invert(ev) {
    if (!ev.isTrusted) return;  // our own re-dispatched event
    ev.preventDefault();
    ev.stopImmediatePropagation();
    var Wheel = ev.target.ownerDocument.defaultView.WheelEvent;
    ev.target.dispatchEvent(new Wheel("wheel", {
      deltaX: -ev.deltaX, deltaY: -ev.deltaY, deltaZ: ev.deltaZ,
      deltaMode: ev.deltaMode, clientX: ev.clientX, clientY: ev.clientY,
      screenX: ev.screenX, screenY: ev.screenY, buttons: ev.buttons,
      ctrlKey: ev.ctrlKey, shiftKey: ev.shiftKey, altKey: ev.altKey,
      metaKey: ev.metaKey, bubbles: true, cancelable: true, composed: true
    }));
  }
  function attach() {
    var frames;
    try { frames = window.parent.document.querySelectorAll("iframe"); }
    catch (e) { return; }
    frames.forEach(function (f) {
      var w, d;
      try { w = f.contentWindow; d = f.contentDocument; } catch (e) { return; }
      if (!w || !d || w.__glycoInvertedZoom) return;
      if (!d.querySelector("[class*='msp-']")) return;  // not a Mol* frame (yet)
      w.addEventListener("wheel", invert, {capture: true, passive: false});
      w.__glycoInvertedZoom = true;
    });
  }
  attach();
  setInterval(attach, 1000);  // Mol* frames are recreated when their key changes
})();
</script>
"""


# =====================================================================
# Color Helpers
# =====================================================================


def _hex_to_rgb(color):
    """'#rrggbb' -> (r, g, b) with 0-255 integers."""
    color = color.lstrip("#")
    return tuple(int(color[k : k + 2], 16) for k in (0, 2, 4))


def _interpolate(stops, t):
    """Color at ``t`` in [0, 1], linear between evenly spaced ``stops``."""
    t = min(max(t, 0.0), 1.0)
    n = len(stops) - 1
    i = min(int(t * n), n - 1)  # segment index
    f = t * n - i  # position inside the segment
    a, b = _hex_to_rgb(stops[i]), _hex_to_rgb(stops[i + 1])
    r, g, bl = (round(x + (y - x) * f) for x, y in zip(a, b))
    return f"#{r:02x}{g:02x}{bl:02x}"


def _js_number(value):
    """``value`` formatted as JavaScript converts it to a property key."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _color_map(values, vmin, vmax, stops):
    """{B-factor as a JS key: color} with a linear 0-1 normalization."""
    span = vmax - vmin
    return {
        _js_number(v): _interpolate(stops, (v - vmin) / span if span else 0.0)
        for v in set(values)
    }


def _colorbar_html(stops, vmin, vmax):
    """Horizontal color bar labelled with 0 / 1 and the B-factor values."""
    gradient = ", ".join(
        f"{c} {100 * k / (len(stops) - 1):.0f}%" for k, c in enumerate(stops)
    )
    return f"""
<div style="font-family: sans-serif; font-size: 0.85rem; max-width: {RESULT_WIDTH}px;">
  <div style="height: 16px; border: 1px solid #999; border-radius: 3px;
              background: linear-gradient(to right, {gradient});"></div>
  <div style="display: flex; justify-content: space-between; margin-top: 2px;">
    <span>0 &nbsp;(B = {vmin:g})</span>
    <span>0.5</span>
    <span>1 &nbsp;(B = {vmax:g})</span>
  </div>
</div>"""


def _invert_zoom_js(uniqueid):
    """Wheel handler that zooms the py3Dmol viewer in the chosen direction."""
    sign = 1 if SCROLL_DOWN_ZOOMS_IN else -1
    return f"""
<script>
(function () {{
  window.addEventListener("wheel", function (ev) {{
    var div = document.getElementById("3dmolviewer_{uniqueid}");
    var viewer = window["viewer_{uniqueid}"];
    if (!div || !viewer || !div.contains(ev.target)) return;
    ev.preventDefault();
    ev.stopImmediatePropagation();
    var dy = ev.deltaMode === 1 ? 16 * ev.deltaY : ev.deltaY;  // lines -> pixels
    dy = Math.max(-200, Math.min(200, dy));
    viewer.zoom(Math.pow({ZOOM_PER_PIXEL}, {sign} * dy));  // > 1 zooms in
  }}, {{capture: true, passive: false}});
}})();
</script>"""


def _hover_js(vmin, vmax):
    """Hover label: residue, B-factor and its normalized 0-1 value."""
    span = (vmax - vmin) or 1.0
    return f"""function(atom, viewer) {{
  if (!atom.label) {{
    var t = Math.min(Math.max((atom.b - {vmin}) / {span}, 0), 1);
    atom.label = viewer.addLabel(
      atom.resn + " " + atom.resi + " " + atom.chain + ": " +
      atom.b.toFixed(1) + " (" + t.toFixed(2) + ")",
      {{position: atom, backgroundColor: "black", fontColor: "white", fontSize: 12}});
  }}
}}"""


# =====================================================================
# Parser description
# =====================================================================


def get_analysis_arguments(parser):
    """
    Describe every option of ``parser`` as a dict (help action excluded).

    :param parser: Parser to describe.
    :type parser: argparse.ArgumentParser
    :return: One dict per option with ``name``, ``dest``, ``option_strings``,
        ``required``, ``default``, ``type`` (widget type), ``type_name``
        (argparse type function name), ``action``, ``nargs``, ``choices``
        and ``help``.
    :rtype: list[dict]
    """
    # argparse has no public API for listing actions
    # pylint: disable=protected-access
    action_types = (
        (argparse._StoreTrueAction, "store_true"),
        (argparse._StoreFalseAction, "store_false"),
        (argparse._AppendAction, "append"),
    )
    arguments = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if action.default == argparse.SUPPRESS:
            continue
        action_type = next(
            (name for cls, name in action_types if isinstance(action, cls)), None
        )
        type_name = getattr(action.type, "__name__", "str") if action.type else "str"
        arguments.append(
            {
                "name": action.option_strings[0]
                if action.option_strings
                else action.dest,
                "dest": action.dest,
                "option_strings": action.option_strings,
                "required": action.required,
                "default": action.default,
                "type": _TYPE_NAMES.get(type_name, "str"),
                "type_name": type_name,
                "action": action_type,
                "nargs": action.nargs,
                "choices": action.choices,
                "help": action.help or "",
            }
        )
    return arguments


def get_argument_groups(parser):
    """
    The options of :func:`get_analysis_arguments`, grouped like ``--help``.

    :rtype: list[tuple[str, list[dict]]]
    """
    # argparse has no public API for argument groups
    # pylint: disable=protected-access
    by_dest = {a["dest"]: a for a in get_analysis_arguments(parser)}
    groups = []
    for group in parser._action_groups:
        args = [by_dest[a.dest] for a in group._group_actions if a.dest in by_dest]
        if args:
            groups.append((group.title, args))
    return groups


def supported_mdanalysis_formats():
    """Return supported MDAnalysis topology/trajectory file extensions."""
    # MDAnalysis exposes the registered formats via private attributes.
    # pylint: disable=protected-access
    return [k.lower() for k in mda._PARSERS], [k.lower() for k in mda._READERS]


# =====================================================================
# Parameter widgets
# =====================================================================


def _widget_label(arg):
    """Widget label: the longest option string without leading dashes."""
    return max(arg["option_strings"] or [arg["name"]], key=len).lstrip("-")


def _format_choice(value):
    return "(default)" if value is None else str(value)


def _choice_widget(arg, label, key):
    choices = list(arg["choices"])
    if arg["default"] is None and not arg["required"]:
        choices = [None, *choices]
    index = choices.index(arg["default"]) if arg["default"] in choices else 0
    return st.selectbox(
        label,
        choices,
        index=index,
        help=arg["help"],
        key=key,
        format_func=_format_choice,
    )


def _list_widget(arg, label, key):
    """Comma-separated values; returns ``(values, error or None)``."""
    default = arg["default"]
    if isinstance(default, (list, tuple)):
        defaults = list(default)
    elif default is None:
        defaults = []
    else:
        defaults = [default]
    text = st.text_input(
        label,
        value=", ".join(map(str, defaults)),
        key=key,
        help=f"{arg['help']} Enter multiple values separated by commas.",
    )
    convert = {"int": int, "float": float}.get(arg["type"], str)
    try:
        return [convert(x.strip()) for x in text.split(",") if x.strip()], None
    except ValueError:
        return [], f"{label}: expected {arg['type']} values"


def _number_widget(arg, label, key):
    """Number input; returns ``(value, error or None)``."""
    convert = int if arg["type"] == "int" else float
    if arg["default"] is None:
        # optional number: blank means "use the program default"
        text = st.text_input(
            label, value="", key=key, help=f"{arg['help']} (blank = default)"
        )
        try:
            return (convert(text) if text.strip() else None), None
        except ValueError:
            return None, f"{label}: expected a number"

    min_value = _MIN_VALUES.get(arg["type_name"])
    value = st.number_input(
        label,
        value=convert(arg["default"]),
        step=1 if convert is int else 0.1,
        help=arg["help"],
        key=key,
        min_value=None if min_value is None else convert(min_value),
    )
    return convert(value), None


def _cli_widget(arg):
    """Widget for one parser option; returns ``(value, error or None)``."""
    label, key = _widget_label(arg), f"cli_{arg['dest']}"
    default = arg["default"]
    if arg["action"] in ("store_true", "store_false"):
        value = default if isinstance(default, bool) else arg["action"] == "store_false"
        return st.checkbox(label, value=value, help=arg["help"], key=key), None
    if arg["choices"] is not None:
        return _choice_widget(arg, label, key), None
    if arg["nargs"] in ("+", "*") or arg["action"] == "append":
        return _list_widget(arg, label, key)
    if arg["type"] in ("int", "float"):
        return _number_widget(arg, label, key)
    value = st.text_input(
        label, value="" if default is None else str(default), help=arg["help"], key=key
    )
    return value, None


def create_cli_inputs(parser, exclude=None):
    """
    Streamlit widgets for every parser option not in ``exclude``, one
    collapsible section per parser argument group.

    :param parser: Parser whose options are shown.
    :type parser: argparse.ArgumentParser
    :param exclude: Destinations handled elsewhere (default: HANDLED_DESTS).
    :type exclude: collections.abc.Set[str] or None
    :return: ``(values, errors)``: widget value per destination, and the
        input errors that must be fixed before running.
    :rtype: tuple[dict, list[str]]
    """
    if exclude is None:
        exclude = HANDLED_DESTS

    values, errors = {}, []
    first = True
    for title, args in get_argument_groups(parser):
        shown = [a for a in args if a["dest"] not in exclude]
        if not shown:
            continue
        with st.expander(title.capitalize(), expanded=first):
            first = False
            for arg in shown:
                dest = arg["dest"]
                values[dest], problem = _cli_widget(arg)
                if problem:
                    errors.append(problem)
                # n_workers_int also rejects 0, which a widget minimum cannot exclude
                if arg["type_name"] == "n_workers_int" and values[dest] == 0:
                    errors.append(
                        f"{_widget_label(arg)}: must be -1 (all processors) or >= 1"
                    )

    for message in errors:
        st.error(message)
    return values, errors


# =====================================================================
# Running glyco
# =====================================================================


def _read_percent(progress_file):
    """Progress percentage from glyco's JSON file, or None if not readable yet."""
    try:
        with open(progress_file, encoding="utf-8") as handle:
            return int(json.load(handle).get("percent", 0))
    except (OSError, ValueError):  # missing / empty file (JSONDecodeError)
        return None


def run_analysis(command):
    """
    Run glyco with a live progress bar.

    Progress is read from the JSON file glyco writes to
    ``GLYCO_PROGRESS_FILE``.

    :param list[str] command: Command line to run.
    :return: ``(exit code, combined stdout and stderr)``
    :rtype: tuple[int, str]
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as progress_handle:
        progress_file = progress_handle.name
    env = {**os.environ, "GLYCO_PROGRESS_FILE": progress_file, "GLYCO_STREAMLIT": "1"}

    log_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", delete=False, suffix=".log"
        ) as log_file:
            log_path = log_file.name
            progress_bar, status_text = st.progress(0), st.empty()
            with subprocess.Popen(
                command, stdout=log_file, stderr=subprocess.STDOUT, env=env
            ) as process:
                while process.poll() is None:
                    percent = _read_percent(progress_file)
                    if percent is not None:
                        progress_bar.progress(min(percent, 100))
                        status_text.text(f"Analyzing: {percent}%")
                    time.sleep(0.2)
            log_file.flush()
            log_file.seek(0)
            output = log_file.read()
            progress_bar.empty()
            status_text.empty()
        return process.returncode, output
    finally:
        for path in (progress_file, log_path):
            remove_file(path)


def _option_args(arg, value):
    """Command-line arguments for one parameter widget value."""
    option = max(arg["option_strings"], key=len)
    if arg["action"] == "store_true":
        return [option] if value else []
    if arg["action"] == "store_false":
        return [] if value else [option]
    if isinstance(value, list):
        return [option, *map(str, value)] if value else []
    if value is None or value == "" or value == arg["default"]:
        return []
    return [option, str(value)]


def build_command(frames, cli_inputs):
    """
    glyco command line for the current inputs (without ``--outdir``).

    :param frames: ``slice(start, stop, step)`` or a list of frame indices.
    :type frames: slice or list[int]
    :param dict cli_inputs: Values from :func:`create_cli_inputs`.
    :rtype: list[str]
    """
    state = st.session_state
    command = [*GLYCO_COMMAND, "--top", state.topology_path]
    if state.trajectory_paths:
        command += ["--traj", *state.trajectory_paths]  # one option, all files

    if isinstance(frames, slice):
        for flag, value in (
            ("--start", frames.start),
            ("--stop", frames.stop),
            ("--step", frames.step),
        ):
            if value is not None:
                command += [flag, str(int(value))]
    else:
        command += ["--frames", *map(str, frames)]

    command += ["--prot-sel", state.prot_sel, "--gly-sel", state.gly_sel]

    argument_map = {a["dest"]: a for a in get_analysis_arguments(PARSER)}
    for dest, value in cli_inputs.items():
        arg = argument_map.get(dest)
        if dest in HANDLED_DESTS or arg is None or not arg["option_strings"]:
            continue
        command += _option_args(arg, value)
    return command


# =====================================================================
# Files, Universe and session state
# =====================================================================


def remove_file(path):
    """
    Remove a temporary file if it exists (errors are ignored).

    :param path: File path, or None.
    :type path: str or None
    """
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _temp_path(suffix):
    """Path of a new, closed temporary file with ``suffix``."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        return tmp.name


def _new_run_dir():
    """
    Create ``<RESULTS_ROOT>/<input name>_<timestamp>`` and return its path.

    A counter is appended if two runs start in the same second, so an
    existing run is never overwritten.

    :rtype: str
    """
    stem = st.session_state.get("input_name") or "glyco"
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    base = os.path.join(
        os.path.abspath(RESULTS_ROOT),
        f"{stem}_{datetime.now():%Y%m%d-%H%M%S}",
    )
    path, n = base, 1
    while True:
        try:
            os.makedirs(path)  # fails if it exists
            return path
        except FileExistsError:
            n += 1
            path = f"{base}_{n}"


def _session_defaults():
    """Fresh session-state defaults (new list objects for every session)."""
    return {
        "u": None,
        "universe_id": 0,  # changes with every loaded Universe (widget keys)
        "topology_path": None,
        "trajectory_paths": [],
        "outdir": None,
        "results_zip": None,
        "last_run": None,
        "run_id": 0,  # changes with every run (widget keys)
        "viewer_files": None,  # PDB + trajectory written for Mol*
        "viewer_files_id": 0,  # changes when they are rewritten (Mol* key)
        "prot_sel": "protein",
        "gly_sel": "",
        "gly_sel_auto": None,  # last selection written by the guesser
        "glycan_guess": None,  # {"names", "messages", "log"} of the guesser
        "input_name": None,  # original topology file name, for run directories
    }


def init_session_state():
    """Fill missing session-state entries with their defaults."""
    for key, value in _session_defaults().items():
        st.session_state.setdefault(key, value)


def _clear_viewer_files():
    """Remove the PDB and trajectory written for the Mol* viewer."""
    files = st.session_state.get("viewer_files")
    if files:
        remove_file(files["pdb"])
        remove_file(files["traj"])
    st.session_state["viewer_files"] = None


def clear_files():
    """Remove the files of the current Universe and forget it."""
    _clear_viewer_files()
    remove_file(st.session_state.topology_path)
    for path in st.session_state.trajectory_paths:
        remove_file(path)
    st.session_state.topology_path = None
    st.session_state.trajectory_paths = []
    st.session_state.u = None


def clear_results():
    """Forget the last run (its output directory stays on disk)."""
    st.session_state.outdir = None
    st.session_state.last_run = None
    st.session_state.results_zip = None


def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary file, keeping its extension.

    The extension matters: MDAnalysis picks the reader from it.

    :param uploaded_file: Streamlit upload.
    :type uploaded_file: streamlit.runtime.uploaded_file_manager.UploadedFile
    :return: Path of the temporary file.
    :rtype: str
    """
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
    return tmp.name


def create_universe(topology_file, trajectory_files):
    """
    Save the uploads and build an MDAnalysis Universe from them.

    Files of a previous Universe are removed first; on failure the new
    temporary files are removed again.

    :param topology_file: Uploaded topology / structure file.
    :param trajectory_files: Uploaded trajectory files (may be empty).
    :return: The new Universe (also stored in the session state).
    :rtype: MDAnalysis.Universe
    """
    clear_files()
    topology_path = save_uploaded_file(topology_file)
    trajectory_paths = []
    loaded = False
    try:
        for trajectory_file in trajectory_files:
            trajectory_paths.append(save_uploaded_file(trajectory_file))
        universe = (
            mda.Universe(topology_path, trajectory_paths)
            if trajectory_paths
            else mda.Universe(topology_path)
        )
        loaded = True
    finally:
        if not loaded:
            remove_file(topology_path)
            for path in trajectory_paths:
                remove_file(path)
    state = st.session_state
    state.topology_path = topology_path
    state.trajectory_paths = trajectory_paths
    state.input_name = os.path.splitext(topology_file.name)[0]
    state.u = universe
    state["universe_id"] = state.get("universe_id", 0) + 1
    return universe


def _check_selection(universe, selection_name, selection):
    """Validation result ``{"valid", "message", "n_atoms"}`` of one selection."""
    if not selection:
        return {"valid": False, "message": "Selection cannot be empty.", "n_atoms": 0}
    try:
        atoms = universe.select_atoms(selection)
    except SELECTION_ERRORS as exc:
        return {
            "valid": False,
            "message": f"Invalid selection\n\n`{selection}`\n\n{exc}",
            "n_atoms": 0,
        }
    if len(atoms) == 0:
        return {
            "valid": False,
            "message": f"0 atoms selected\n\n`{selection}`",
            "n_atoms": 0,
        }
    if selection_name == "Protein" and len(atoms.select_atoms("name CA")) == 0:
        return {
            "valid": False,
            "message": f"No CA atoms in\n\n`{selection}`",
            "n_atoms": len(atoms),
        }
    return {"valid": True, "message": f"{len(atoms):,} atoms", "n_atoms": len(atoms)}


def apply_glycan_guess(overwrite=False):
    """
    Run the standardize glycan guesser on the current Universe (frame 0).

    Fills the glycan selection if it is empty, still holds the previous
    guess, or ``overwrite`` is set; a selection typed by the user is kept.
    """
    state = st.session_state
    universe = state.get("u")
    if universe is None:
        return
    universe.trajectory[0]  # pylint: disable=pointless-statement
    try:
        names, rep = guess_glycans(universe)
    except GUESS_ERRORS as exc:
        state.glycan_guess = {
            "names": [],
            "messages": [f"Glycan guesser failed: {exc}"],
            "log": [],
        }
        return
    state.glycan_guess = {
        "names": names,
        "messages": rep["warnings"] + rep["info"],
        "log": rep.get("glycan_classification", []),
    }
    selection = glycan_selection(names)
    current = state.get("gly_sel", "").strip()
    if overwrite or not current or current == state.get("gly_sel_auto"):
        state["gly_sel"] = selection
        state["gly_sel_auto"] = selection


def _redetect_glycans():
    """Button callback: replace the glycan selection with a new guess."""
    apply_glycan_guess(overwrite=True)
    validate_atom_selections()


def _glycan_guess_details():
    """Detected names and the per-name decisions of the guesser."""
    guess = st.session_state.get("glycan_guess")
    if not guess:
        return
    with st.expander("Glycan guesser", expanded=False):
        names = guess["names"]
        st.write(f"Detected: {', '.join(names)}" if names else "No glycans detected.")
        for message in guess["messages"]:
            st.caption(message)
        if guess["log"]:
            st.dataframe(pd.DataFrame(guess["log"]), use_container_width=True)


def validate_atom_selections():
    """
    Validate the protein and glycan selections against the current Universe.

    Used as the selection widgets' callback and after loading a Universe.
    Results are stored in ``selection_validation_results`` and
    ``selections_valid``; the validated strings in ``_validated_prot_sel`` /
    ``_validated_gly_sel``, so later edits invalidate them.
    """
    state = st.session_state
    universe = state.get("u")
    prot_sel = state.get("prot_sel", "").strip()
    gly_sel = state.get("gly_sel", "").strip()
    state["prot_sel"] = prot_sel
    state["gly_sel"] = gly_sel
    if universe is None:
        return

    results = {
        name: _check_selection(universe, name, sel)
        for name, sel in (("Protein", prot_sel), ("Glycan", gly_sel))
    }
    passed = all(r["valid"] for r in results.values())
    state["selection_validation_results"] = results
    state["selections_valid"] = passed
    if passed:
        state["_validated_prot_sel"] = prot_sel
        state["_validated_gly_sel"] = gly_sel
    else:
        state.pop("_validated_prot_sel", None)
        state.pop("_validated_gly_sel", None)


def _selections_ready():
    """Selections are valid and unchanged since they were validated."""
    state = st.session_state
    return bool(
        state.get("selections_valid", False)
        and state.get("prot_sel", "").strip() == state.get("_validated_prot_sel")
        and state.get("gly_sel", "").strip() == state.get("_validated_gly_sel")
    )


def zip_directory(path):
    """
    All files below ``path`` as ZIP bytes (paths relative to ``path``).

    :param str path: Directory to pack.
    :rtype: bytes
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(path):
            for name in files:
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, path))
    return buf.getvalue()


def _find_outputs(outdir, suffix=None, name=None):
    """Files below ``outdir`` called ``name`` or ending in ``suffix`` (sorted)."""
    found = []
    for root, _, files in os.walk(outdir):
        for f in files:
            if (name and f == name) or (suffix and f.endswith(suffix)):
                found.append(os.path.join(root, f))
    return sorted(found)


def _run_succeeded():
    """The last run finished with exit code 0 and its output still exists."""
    state = st.session_state
    last = state.last_run
    return bool(
        last
        and last["returncode"] == 0
        and state.outdir
        and os.path.isdir(state.outdir)
    )


# =====================================================================
# Input viewer: PDB + trajectory played by Mol*
# =====================================================================


def _invert_molstar_zoom():
    """Hidden helper that reverses the wheel zoom of every Mol* viewer."""
    if INVERT_SCROLL_ZOOM:
        components.html(MOLSTAR_INVERT_ZOOM_HTML, height=0)


def _default_stride(n_frames):
    """Smallest stride that keeps at most VIEWER_MAX_FRAMES frames."""
    return max(1, -(-n_frames // VIEWER_MAX_FRAMES))  # ceiling division


def _write_viewer_files(universe, selection, stride):
    """
    Topology PDB and strided trajectory of the selected atoms for Mol*.

    Written once per (Universe, selection, stride); later calls reuse the
    files, so the Mol* key stays the same and the viewer is not reloaded.

    :param universe: Loaded Universe.
    :param str selection: MDAnalysis selection; empty for all atoms.
    :param int stride: Every ``stride``-th frame is written.
    :return: ``{"wanted", "pdb", "traj" (or None), "n_atoms", "n_frames"}``
    :rtype: dict
    """
    state = st.session_state
    wanted = (state.universe_id, selection, stride)
    files = state.get("viewer_files")
    if files and files["wanted"] == wanted and os.path.exists(files["pdb"]):
        return files

    _clear_viewer_files()
    atoms = universe.select_atoms(selection) if selection else universe.atoms
    if len(atoms) == 0:
        raise ValueError("the selection contains no atoms")
    frames = list(range(0, len(universe.trajectory), stride))[:VIEWER_MAX_FRAMES]

    pdb_path = _temp_path(".pdb")
    traj_path = _temp_path(VIEWER_TRAJ_SUFFIX) if len(frames) > 1 else None
    try:
        with warnings.catch_warnings():  # missing PDB attributes are filled in
            warnings.simplefilter("ignore")
            universe.trajectory[frames[0]]  # pylint: disable=pointless-statement
            atoms.write(pdb_path)
            if traj_path:
                with mda.Writer(traj_path, n_atoms=atoms.n_atoms) as writer:
                    for _ in universe.trajectory[frames]:
                        writer.write(atoms)
    except BaseException:  # also Streamlit's rerun/stop interrupts
        remove_file(pdb_path)
        remove_file(traj_path)
        raise

    state["viewer_files"] = {
        "wanted": wanted,
        "pdb": pdb_path,
        "traj": traj_path,
        "n_atoms": len(atoms),
        "n_frames": len(frames),
    }
    state["viewer_files_id"] = state.get("viewer_files_id", 0) + 1
    return state["viewer_files"]


def _input_view(universe):
    """Viewer content: the input system, played by Mol* in the browser."""
    state = st.session_state
    if st_molstar is None:
        st.info(
            "Install streamlit-molstar (`pip install streamlit-molstar`) to "
            "view the input structure."
        )
        return

    n_frames = len(universe.trajectory)
    col1, col2 = st.columns([2, 3])
    with col1:
        stride = (
            st.number_input(
                "Load every Nth frame",
                min_value=1,
                max_value=n_frames,
                value=_default_stride(n_frames),
                step=1,
                key=f"viewer_stride_{state.universe_id}",  # new range per Universe
                help=f"At most {VIEWER_MAX_FRAMES} frames are sent to the browser.",
            )
            if n_frames > 1
            else 1
        )
    with col2:
        ready = _selections_ready()
        subset = st.checkbox(
            "Only protein and glycan selections",
            key="viewer_subset",
            disabled=not ready,
            help="Shows the atoms of the sidebar selections; requires valid "
            "selections. Also makes large systems faster to load.",
        )
    selection = f"({state.prot_sel}) or ({state.gly_sel})" if subset and ready else ""

    try:
        with st.spinner("Preparing the structure for the viewer..."):
            files = _write_viewer_files(universe, selection, int(stride))
    except VIEWER_ERRORS + tuple(SELECTION_ERRORS) as exc:
        st.error(f"Could not prepare the structure for the viewer: {exc}")
        return

    size_mb = sum(os.path.getsize(p) for p in (files["pdb"], files["traj"]) if p) / 1e6
    caption = (
        f"{files['n_atoms']:,} atoms · {files['n_frames']} frame(s) · {size_mb:.1f} MB"
    )
    if files["traj"]:
        caption += " · use the Mol* animation controls to play or step through frames"
    st.caption(caption)

    key = f"molstar_input_{state.viewer_files_id}"  # stable while files are unchanged

    if files["traj"]:
        st_molstar(files["pdb"], files["traj"], height=MOLSTAR_HEIGHT, key=key)
    else:
        st_molstar(files["pdb"], height=MOLSTAR_HEIGHT, key=key)
    _invert_molstar_zoom()


# =====================================================================
# Result viewer: py3Dmol colored by the B-factor column
# =====================================================================


def _read_bfactors(pdb_text):
    """B-factors (columns 61-66) of every ATOM/HETATM record."""
    values = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                values.append(float(line[60:66]))
            except ValueError:
                pass
    return values


def _py3dmol_html(pdb_text, style, cmap, vmin, vmax):
    """HTML of a py3Dmol view: protein colored by ``cmap``, rest as grey sticks."""
    view = py3Dmol.view(width=RESULT_WIDTH, height=RESULT_HEIGHT)
    view.addModel(pdb_text, "pdb")
    scheme = {"prop": "b", "map": cmap}
    protein = {"resn": AMINO_ACIDS}

    view.setStyle({}, {"stick": {"color": "lightgray", "radius": 0.15}})  # glycans etc.
    if style == "Surface":
        view.setStyle(protein, {"cartoon": {"color": "white", "opacity": 0.3}})
        view.addSurface(py3Dmol.VDW, {"opacity": 0.95, "colorscheme": scheme}, protein)
    elif style == "Sticks":
        view.setStyle(protein, {"stick": {"colorscheme": scheme}})
    else:
        view.setStyle(protein, {"cartoon": {"colorscheme": scheme}})
    view.setHoverable({}, True, _hover_js(vmin, vmax), UNHOVER_JS)
    view.zoomTo()

    view.setHoverable({}, True, _hover_js(vmin, vmax), UNHOVER_JS)
    view.zoomTo()
    html = view._make_html()  # pylint: disable=protected-access  # as in stmol
    if INVERT_SCROLL_ZOOM:
        html += _invert_zoom_js(view.uniqueid)
    return html


def _result_view(path):
    """Viewer content: one result PDB colored by a linear 0-1 scale."""
    state = st.session_state
    try:
        with open(path, encoding="utf-8") as handle:
            pdb_text = handle.read()
    except OSError as exc:
        st.error(f"Could not read {os.path.basename(path)}: {exc}")
        return

    if py3Dmol is None:  # fall back to Mol* without automatic coloring
        st.info(
            "Install py3Dmol (`pip install py3Dmol`) to color the structure "
            "by glyco's values automatically."
        )
        if st_molstar is not None:
            result_key = f"molstar_result_{state.get('run_id', 0)}"
            st_molstar(path, height=MOLSTAR_HEIGHT, key=result_key)
            _invert_molstar_zoom()
        return

    values = _read_bfactors(pdb_text)
    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        style = st.radio("Style", RESULT_STYLES, horizontal=True, key="result_style")
    with col2:
        scale = st.selectbox("Color scale", list(COLOR_SCALES), key="result_scale")
    with col3:
        reverse = st.checkbox("Reverse", key="result_reverse")
    stops = COLOR_SCALES[scale][::-1] if reverse else COLOR_SCALES[scale]

    if hi > lo:
        start = 0.0 if lo >= 0 else float(lo)  # counts: 0 maps to t = 0
        vmin, vmax = st.slider(
            "B-factor mapped to 0 … 1",
            min_value=start,
            max_value=float(hi),
            value=(start, float(hi)),
            key=f"result_range_{state.get('run_id', 0)}_{os.path.basename(path)}",
            help="Values below the range get color 0, values above it color 1.",
        )
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = lo, lo + 1.0  # constant values: avoid a zero-width range

    cmap = _color_map(values, vmin, vmax, stops)
    components.html(
        _py3dmol_html(pdb_text, style, cmap, vmin, vmax),
        height=RESULT_HEIGHT + 20,
    )
    st.html(_colorbar_html(stops, vmin, vmax))
    st.caption(
        "Linear scale: t = (B − min) / (max − min), clipped to 0 … 1. "
        "Non-protein residues are grey sticks. Hover over an atom to see its "
        "value and t."
    )


def _result_structures():
    """``{viewer key: (path, label, description)}`` of the last run's PDBs."""
    state = st.session_state
    if not _run_succeeded():
        return {}
    found = {}
    for name, (label, description) in RESULT_STRUCTURES.items():
        for path in _find_outputs(state.outdir, name=name):
            relative = os.path.relpath(path, state.outdir)
            shown = label if relative == name else f"{label} ({relative})"
            found[RESULT_PREFIX + path] = (path, shown, description)
    return found


@fragment
def viewer_panel(universe):
    """Input trajectory after loading; result structures after a run."""
    state = st.session_state
    st.subheader("Structure viewer")

    results = _result_structures()
    options = {VIEW_INPUT: "Input structure / trajectory"}
    options.update({key: label for key, (_, label, _) in results.items()})
    if state.get("viewer_source") not in options:
        state["viewer_source"] = VIEW_INPUT  # e.g. results were cleared
    source = st.selectbox(
        "Show", list(options), format_func=options.get, key="viewer_source"
    )

    if source == VIEW_INPUT:
        _input_view(universe)
    else:
        path, _, description = results[source]
        st.caption(description)
        _result_view(path)


# =====================================================================
# Sidebar: files and selections
# =====================================================================


def _load_universe(topology_file, trajectory_files):
    """Build the Universe from the uploads and reset results and viewer."""
    state = st.session_state
    try:
        with st.spinner("Loading MDAnalysis Universe..."):
            universe = create_universe(topology_file, trajectory_files)
    except LOAD_ERRORS as exc:
        st.error(f"Failed to load Universe: {exc}")
        return
    clear_results()  # results of the old Universe
    apply_glycan_guess()  # fill the glycan selection from the new Universe
    validate_atom_selections()  # selections against the NEW Universe
    state["viewer_source"] = VIEW_INPUT  # viewer widgets not drawn yet
    state["viewer_subset"] = False  # must match the files prepared below

    # written once here; the viewer reuses them on every rerun
    try:
        with st.spinner("Preparing the 3D viewer..."):
            _write_viewer_files(universe, "", _default_stride(len(universe.trajectory)))
    except VIEWER_ERRORS as exc:
        st.warning(f"Universe loaded, but the viewer could not be prepared: {exc}")
    st.success("Universe loaded.")


def _selection_feedback():
    """Compact validation messages below the selection boxes."""
    results = st.session_state.get("selection_validation_results")
    if not results:
        return
    for name in ("Protein", "Glycan"):
        result = results.get(name)
        if result and result["valid"]:
            st.success(f"{name}: {result['message']}")
        elif result:
            st.error(f"{name}: {result['message']}")


def sidebar():
    """Upload files, load the Universe and edit the atom selections."""
    with st.sidebar:
        st.header("1. Input files")
        parser_types, reader_types = supported_mdanalysis_formats()
        topology_file = st.file_uploader(
            "Topology / structure file", key="topology_upload", type=parser_types
        )
        trajectory_files = st.file_uploader(
            "Trajectory files (optional)",
            accept_multiple_files=True,
            key="trajectory_upload",
            type=reader_types,
        )
        if st.button(
            "Load Universe",
            type="primary",
            disabled=topology_file is None,
            use_container_width=True,
        ):
            _load_universe(topology_file, trajectory_files)

        if st.session_state.u is None:
            return
        st.header("2. Atom selections")
        st.text_area(
            "Protein selection",
            key="prot_sel",
            height=80,
            placeholder="protein",
            on_change=validate_atom_selections,
            help="MDAnalysis selection of the protein residues "
            "(each needs one CA atom), e.g. 'protein'",
        )
        st.text_area(
            "Glycan selection",
            key="gly_sel",
            height=80,
            placeholder="resname NAG",
            on_change=validate_atom_selections,
            help="MDAnalysis selection of the glycan atoms, e.g. 'resname NAG BMA'",
        )
        st.button(
            "Detect glycans",
            on_click=_redetect_glycans,
            use_container_width=True,
            help="Replace the glycan selection with the residues detected by "
            "the glyco standardize guesser (name lists and sugar-ring test).",
        )
        _selection_feedback()
        _glycan_guess_details()


# =====================================================================
# Main area
# =====================================================================


def metrics_row(universe):
    """System and last-run metrics across the top of the page."""
    state = st.session_state
    results = state.get("selection_validation_results") or {}

    def count(name):
        result = results.get(name)
        return f"{result['n_atoms']:,}" if result and result["valid"] else "—"

    last = state.last_run
    if not last:
        run_status = "not run"
    elif last["returncode"] == 0:
        run_status = f"ok ({last['seconds']:.0f} s)"
    else:
        run_status = f"failed ({last['returncode']})"

    cols = st.columns(7)
    cols[0].metric("Atoms", f"{universe.atoms.n_atoms:,}")
    cols[1].metric("Residues", f"{len(universe.residues):,}")
    cols[2].metric("Segments", f"{len(universe.segments):,}")
    cols[3].metric("Frames", f"{len(universe.trajectory):,}")
    cols[4].metric("Protein atoms", count("Protein"))
    cols[5].metric("Glycan atoms", count("Glycan"))
    cols[6].metric("Last run", run_status)
    if state.trajectory_paths:
        st.caption(
            f"{len(state.trajectory_paths)} trajectory file(s) · "
            f"timestep {round(universe.trajectory.dt, 2)} ps · "
            f"total time {round(universe.trajectory.totaltime, 2)} ps"
        )


def _slice_frames(n_frames):
    """Start / stop / step inputs; returns a slice, or None if it is empty."""
    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.number_input(
            "Start",
            min_value=0,
            max_value=n_frames - 1,
            value=None,
            step=1,
            placeholder="0",
            help="Blank = beginning",
        )
    with col2:
        stop = st.number_input(
            "Stop (exclusive)",
            min_value=-n_frames,
            max_value=n_frames,
            value=None,
            step=1,
            placeholder=str(n_frames),
            help="Blank = end; negative values count from the end",
        )
    with col3:
        step = st.number_input(
            "Step",
            min_value=1,
            value=None,
            step=1,
            placeholder="1",
            help="Blank = every frame",
        )
    # same semantics as Python slicing, so negative stops work
    frames = slice(start, stop, step)
    selected = range(n_frames)[frames]
    if len(selected) == 0:
        st.error("This start / stop / step selects no frames.")
        return None
    st.caption(f"{len(selected)} frame(s) selected")
    return frames


def _list_frames(n_frames):
    """Explicit frame list; returns sorted unique frames, or None if invalid."""
    frame_text = st.text_area(
        "Frames",
        value="0",
        help="Frame numbers separated by commas, spaces or new lines.",
    )
    try:
        frames = sorted({int(x) for x in frame_text.replace(",", " ").split()})
    except ValueError:
        st.error("Invalid frame list.")
        return None
    if not frames:
        st.error("Enter at least one frame.")
        return None
    if any(f < 0 or f >= n_frames for f in frames):
        st.error(f"Frames must be between 0 and {n_frames - 1}.")
        return None
    st.caption(f"{len(frames)} frame(s) selected")
    return frames


def frame_controls(universe):
    """Frame selection; returns a slice, a frame list, or None if invalid."""
    n_frames = len(universe.trajectory)
    frame_mode = st.radio(
        "Frame selection method",
        options=[SLICE_MODE, LIST_MODE],
        horizontal=True,
        key="frame_mode",
    )
    if frame_mode == SLICE_MODE:
        return _slice_frames(n_frames)
    return _list_frames(n_frames)


def _execute_run(frames, cli_inputs):
    """Run glyco in a fresh output directory and store the result."""
    state = st.session_state
    missing = [
        p
        for p in [state.topology_path, *state.trajectory_paths]
        if not p or not os.path.exists(p)
    ]
    if missing:
        st.error(f"Input file(s) no longer exist: {missing}. Reload the Universe.")
        st.stop()

    command = build_command(frames, cli_inputs)

    # fresh output directory for every run, below RESULTS_ROOT
    clear_results()
    try:
        state.outdir = _new_run_dir()
    except OSError as exc:
        st.error(f"Could not create the output directory: {exc}")
        st.stop()
    state["run_id"] = state.get("run_id", 0) + 1
    command += ["--outdir", state.outdir, "--force"]

    started = time.perf_counter()
    try:
        returncode, output = run_analysis(command)
    except RUN_ERRORS as exc:
        st.error(f"Failed to run glyco:\n\n{exc}")
        st.stop()
    state.last_run = {
        "command": shlex.join(command),
        "returncode": returncode,
        "output": output,
        "seconds": time.perf_counter() - started,
    }
    state.results_zip = zip_directory(state.outdir) if returncode == 0 else None

    # point the viewer at the first result structure (drawn after this)
    results = _result_structures()
    if results:
        state["viewer_source"] = next(iter(results))


def run_controls(frames, cli_inputs, cli_errors):
    """Readiness checks, the run button, run status and the download."""
    state = st.session_state
    selections_valid = _selections_ready()
    if not selections_valid:
        st.warning("Enter valid protein and glycan selections in the sidebar.")

    if frames is None:
        st.warning("Select a valid set of frames under 'Trajectory frames'.")
    if cli_errors:
        st.warning("Fix the analysis parameters below.")

    ready = selections_valid and frames is not None and not cli_errors
    if st.button(
        "Run Analysis", type="primary", use_container_width=True, disabled=not ready
    ):
        _execute_run(frames, cli_inputs)

    last = state.last_run
    if last and last["returncode"] == 0:
        st.success(f"Analysis completed in {last['seconds']:.1f} s.")
    elif last:
        st.error(
            f"Analysis failed (exit code {last['returncode']}). See the log below."
        )

    if state.outdir:
        st.caption(f"Output directory: `{state.outdir}`")

    if state.results_zip:
        st.download_button(
            "Download results (ZIP)",
            state.results_zip,
            file_name="glyco_results.zip",
            mime="application/zip",
            use_container_width=True,
        )


def control_panel(universe):
    """Run box on top (always visible), then frames and parameters."""
    st.subheader("Analysis")
    run_box = st.container(border=True)  # filled below, shown on top
    with st.expander("Trajectory frames", expanded=True):
        frames = frame_controls(universe)
    st.markdown("**Parameters**")
    cli_inputs, cli_errors = create_cli_inputs(PARSER)
    with run_box:
        run_controls(frames, cli_inputs, cli_errors)


def _table_order(path):
    """Sort key: preferred tables first, then by path."""
    name = os.path.basename(path)
    rank = (
        PREFERRED_TABLES.index(name)
        if name in PREFERRED_TABLES
        else len(PREFERRED_TABLES)
    )
    return rank, path


def _result_tables(outdir):
    """One selectable CSV of the run as a table."""
    tables = sorted(_find_outputs(outdir, suffix=".csv"), key=_table_order)
    if not tables:
        st.info("The run wrote no CSV files.")
        return
    path = st.selectbox(
        "Table",
        tables,
        format_func=lambda p: os.path.relpath(p, outdir),
        key=f"table_select_{st.session_state.run_id}",
    )
    try:
        frame = pd.read_csv(path, nrows=MAX_TABLE_ROWS)
    except (OSError, ValueError) as exc:  # includes pandas parse errors
        st.error(f"Could not read {os.path.basename(path)}: {exc}")
        return
    if len(frame) == MAX_TABLE_ROWS:
        st.caption(
            f"Showing the first {MAX_TABLE_ROWS:,} rows; the full file is in "
            "the ZIP download."
        )
    st.dataframe(frame, use_container_width=True, height=400)


def results_panel():
    """Full-width results: tables, log and command of the last run."""
    state = st.session_state
    last = state.last_run
    if not last:
        return
    st.divider()
    st.subheader("Results")
    tab_tables, tab_log, tab_command = st.tabs(["Tables", "Log", "Command"])
    with tab_tables:
        if _run_succeeded():
            _result_tables(state.outdir)
        else:
            st.info("No results: the run failed. See the Log tab.")
    with tab_log:
        st.code(last["output"] or "No output.", language="text")
    with tab_command:
        st.code(last["command"], language="bash")


def main():
    """Render the dashboard (Streamlit reruns this on every interaction)."""
    st.set_page_config(page_title="GLYCO: Glycan Coverage", layout="wide")
    init_session_state()
    st.title("GLYCO: Glycan Coverage")

    sidebar()
    universe = st.session_state.u
    if universe is None:
        st.info(
            "Upload a topology (and optionally trajectories) in the sidebar "
            "and press **Load Universe**."
        )
        return

    # filled last, so the "Last run" metric includes a run started below
    metrics_slot = st.container()
    viewer_col, control_col = st.columns([3, 2], gap="large")
    # controls first: a run started here must finish before the viewer is
    # drawn, so the viewer can switch to the new results in the same rerun
    with control_col:
        control_panel(universe)
    with viewer_col:
        viewer_panel(universe)
    with metrics_slot:
        metrics_row(universe)
    results_panel()


if __name__ == "__main__":  # `streamlit run` executes the script as __main__
    main()
