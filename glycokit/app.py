"""
Streamlit front end for glyco (GlyCo, GLYcan COverage).

Sections: 1 load files, 2 Universe information, 3 atom selections,
4 trajectory frames, 5 analysis parameters (generated from
:func:`glycokit.parsers.base_parser`), 6 run and download results.
"""

import argparse
import atexit
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import MDAnalysis as mda
import streamlit as st

from glycokit.parsers import base_parser

# =====================================================================
# Configuration
# =====================================================================

GLYCO_COMMAND = [sys.executable, "-m", "glycokit.glyco"]

parser = base_parser()

# Parser destinations set by sections 1-4 or by the app itself; they are
# not shown as widgets in section 5.
HANDLED_DESTS = {
    "top",
    "traj",
    "indir",
    "in_types",  # section 1
    "prot_sel",
    "gly_sel",  # section 3
    "frame_start",
    "frame_stop",
    "frame_step",
    "frame_list",  # section 4
    "outdir",
    "force",
    "verbose",  # set by the app
}

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


# =====================================================================
# Helper functions
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
    arguments = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if action.default == argparse.SUPPRESS:
            continue

        if isinstance(action, argparse._StoreTrueAction):
            action_type = "store_true"
        elif isinstance(action, argparse._StoreFalseAction):
            action_type = "store_false"
        elif isinstance(action, argparse._AppendAction):
            action_type = "append"
        else:
            action_type = None

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


def create_cli_inputs(parser, exclude=HANDLED_DESTS):
    """
    Streamlit widgets for every parser option not in ``exclude``.

    :param parser: Parser whose options are shown.
    :type parser: argparse.ArgumentParser
    :param exclude: Destinations handled elsewhere.
    :type exclude: set[str]
    :return: ``(values, errors)``: widget value per destination, and the
        input errors that must be fixed before running.
    :rtype: tuple[dict, list[str]]
    """
    values, errors = {}, []
    for arg in get_analysis_arguments(parser):
        dest = arg["dest"]
        if dest in exclude:
            continue

        default, arg_type = arg["default"], arg["type"]
        help_text = arg["help"]
        label = max(arg["option_strings"] or [arg["name"]], key=len).lstrip("-")
        key = f"cli_{dest}"
        min_value = _MIN_VALUES.get(arg["type_name"])

        if arg["action"] in ("store_true", "store_false"):
            value = (
                default if isinstance(default, bool) else arg["action"] == "store_false"
            )
            values[dest] = st.checkbox(label, value=value, help=help_text, key=key)

        elif arg["choices"] is not None:
            choices = list(arg["choices"])
            if default is None and not arg["required"]:
                choices = [None] + choices
            index = choices.index(default) if default in choices else 0
            values[dest] = st.selectbox(
                label,
                choices,
                index=index,
                help=help_text,
                key=key,
                format_func=lambda x: "(default)" if x is None else str(x),
            )

        elif arg["nargs"] in ("+", "*") or arg["action"] == "append":
            defaults = (
                list(default)
                if isinstance(default, (list, tuple))
                else []
                if default is None
                else [default]
            )
            text = st.text_input(
                label,
                value=", ".join(map(str, defaults)),
                key=key,
                help=f"{help_text} Enter multiple values separated by commas.",
            )
            raw = [x.strip() for x in text.split(",") if x.strip()]
            try:
                convert = {"int": int, "float": float}.get(arg_type, str)
                values[dest] = [convert(x) for x in raw]
            except ValueError:
                errors.append(f"{label}: expected {arg_type} values")
                values[dest] = []

        elif arg_type in ("int", "float") and default is None:
            # optional number: blank means "use the program default"
            text = st.text_input(
                label, value="", key=key, help=f"{help_text} (blank = default)"
            )
            try:
                convert = int if arg_type == "int" else float
                values[dest] = convert(text) if text.strip() else None
            except ValueError:
                errors.append(f"{label}: expected a number")
                values[dest] = None

        elif arg_type == "int":
            values[dest] = int(
                st.number_input(
                    label,
                    value=int(default),
                    step=1,
                    help=help_text,
                    key=key,
                    min_value=None if min_value is None else int(min_value),
                )
            )

        elif arg_type == "float":
            values[dest] = float(
                st.number_input(
                    label,
                    value=float(default),
                    step=0.1,
                    help=help_text,
                    key=key,
                    min_value=min_value,
                )
            )

        else:
            values[dest] = st.text_input(
                label,
                value="" if default is None else str(default),
                help=help_text,
                key=key,
            )

        # n_workers_int also rejects 0, which a widget minimum cannot exclude
        if arg["type_name"] == "n_workers_int" and values.get(dest) == 0:
            errors.append(f"{label}: must be -1 (all processors) or >= 1")

    for message in errors:
        st.error(message)
    return values, errors


def run_analysis(command):
    """
    Run glyco with a live progress bar.

    Progress is read from the JSON file glyco writes to
    ``GLYCO_PROGRESS_FILE``.

    :param list[str] command: Command line to run.
    :return: ``(exit code, combined stdout and stderr)``
    :rtype: tuple[int, str]
    """
    progress_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    log_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")

    env = os.environ.copy()
    env["GLYCO_PROGRESS_FILE"] = progress_file
    env["GLYCO_STREAMLIT"] = "1"

    try:
        process = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, env=env, text=True
        )
        progress_bar = st.progress(0)
        status_text = st.empty()
        while process.poll() is None:
            try:
                with open(progress_file) as f:
                    percent = int(json.load(f).get("percent", 0))
                progress_bar.progress(min(percent, 100))
                status_text.text(f"Analyzing: {percent}%")
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                pass
            time.sleep(0.2)

        log_file.flush()
        log_file.seek(0)
        output = log_file.read()
        progress_bar.empty()
        status_text.empty()
        return process.returncode, output
    finally:
        log_file.close()
        for path in (progress_file, log_file.name):
            if os.path.exists(path):
                os.remove(path)


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


def _remove_tree_at_exit(path):
    """
    Remove a temporary directory when the Streamlit server process exits.

    :param str path: Directory to remove.
    """
    atexit.register(shutil.rmtree, path, ignore_errors=True)


def clear_files():
    """Remove the files of the current Universe and forget it."""
    remove_file(st.session_state.topology_path)
    for path in st.session_state.trajectory_paths:
        remove_file(path)
    st.session_state.topology_path = None
    st.session_state.trajectory_paths = []
    st.session_state.u = None


def clear_results():
    """Forget the last run and remove its output directory."""
    if st.session_state.outdir:
        shutil.rmtree(st.session_state.outdir, ignore_errors=True)
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
    try:
        for trajectory_file in trajectory_files:
            trajectory_paths.append(save_uploaded_file(trajectory_file))
        u = (
            mda.Universe(topology_path, trajectory_paths)
            if trajectory_paths
            else mda.Universe(topology_path)
        )
    except Exception:
        remove_file(topology_path)
        for path in trajectory_paths:
            remove_file(path)
        raise
    st.session_state.topology_path = topology_path
    st.session_state.trajectory_paths = trajectory_paths
    st.session_state.u = u
    return u


def validate_atom_selections():
    """
    Validate the protein and glycan selections against the current Universe.

    Used as the selection widgets' callback and after loading a Universe.
    Results are stored in ``selection_validation_results`` and
    ``selections_valid``; the validated strings in ``_validated_prot_sel`` /
    ``_validated_gly_sel``, so later edits invalidate them.
    """
    u = st.session_state.get("u")
    prot_sel = st.session_state.get("prot_sel", "").strip()
    gly_sel = st.session_state.get("gly_sel", "").strip()
    st.session_state["prot_sel"] = prot_sel
    st.session_state["gly_sel"] = gly_sel
    if u is None:
        return

    results, passed = {}, True
    for name, selection in (("Protein", prot_sel), ("Glycan", gly_sel)):
        if not selection:
            results[name] = {"valid": False, "message": "Selection cannot be empty."}
            passed = False
            continue
        try:
            atoms = u.select_atoms(selection)
        except Exception as e:  # noqa: BLE001
            results[name] = {
                "valid": False,
                "message": f"Invalid selection\n\n`{selection}`\n\n{e}",
            }
            passed = False
            continue
        if len(atoms) == 0:
            results[name] = {
                "valid": False,
                "message": f"0 atoms selected\n\n`{selection}`",
            }
            passed = False
        elif name == "Protein" and len(atoms.select_atoms("name CA")) == 0:
            results[name] = {
                "valid": False,
                "message": f"No CA atoms in\n\n`{selection}`",
            }
            passed = False
        else:
            results[name] = {"valid": True, "message": f"{len(atoms):,} atoms"}

    st.session_state["selection_validation_results"] = results
    st.session_state["selections_valid"] = passed
    if passed:
        st.session_state["_validated_prot_sel"] = prot_sel
        st.session_state["_validated_gly_sel"] = gly_sel
    else:
        st.session_state.pop("_validated_prot_sel", None)
        st.session_state.pop("_validated_gly_sel", None)


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


def build_command(frame_mode, frames, frame_start, frame_stop, frame_step, cli_inputs):
    """
    glyco command line for the current inputs (without ``--outdir``).

    :param str frame_mode: ``'Start / Stop / Step'`` or ``'Frame list'``.
    :param frames: Frame list (frame-list mode).
    :param frame_start: Start frame, or None.
    :param frame_stop: Stop frame, or None.
    :param frame_step: Step, or None.
    :param dict cli_inputs: Values from :func:`create_cli_inputs`.
    :rtype: list[str]
    """
    cmd = [*GLYCO_COMMAND, "--top", st.session_state.topology_path]
    if st.session_state.trajectory_paths:
        cmd += ["--traj", *st.session_state.trajectory_paths]  # one option, all files

    if frame_mode == "Start / Stop / Step":
        for flag, value in (
            ("--start", frame_start),
            ("--stop", frame_stop),
            ("--step", frame_step),
        ):
            if value is not None:
                cmd += [flag, str(int(value))]
    else:
        cmd += ["--frames", *map(str, frames)]

    cmd += [
        "--prot-sel",
        st.session_state.prot_sel,
        "--gly-sel",
        st.session_state.gly_sel,
    ]

    argument_map = {a["dest"]: a for a in get_analysis_arguments(parser)}
    for dest, value in cli_inputs.items():
        arg = argument_map.get(dest)
        if dest in HANDLED_DESTS or arg is None or not arg["option_strings"]:
            continue
        option = max(arg["option_strings"], key=len)
        if arg["action"] == "store_true":
            if value:
                cmd.append(option)
        elif arg["action"] == "store_false":
            if not value:
                cmd.append(option)
        elif isinstance(value, list):
            if value:
                cmd += [option, *map(str, value)]
        elif value is not None and value != "" and value != arg["default"]:
            cmd += [option, str(value)]
    return cmd


# =====================================================================
# Session state
# =====================================================================

for key, value in {
    "u": None,
    "topology_path": None,
    "trajectory_paths": [],
    "outdir": None,
    "results_zip": None,
    "last_run": None,
    "prot_sel": "protein",
    "gly_sel": "",
}.items():
    st.session_state.setdefault(key, value)


# =====================================================================
# Application
# =====================================================================

st.title("GlyCo: Glycan Coverage")

# ---------------------------------------------------------------------
# 1. Load files
# ---------------------------------------------------------------------
with st.expander("1. Load Structure and Trajectories", expanded=True):
    st.write(
        "Upload an MDAnalysis-compatible topology/structure file and, "
        "optionally, one or more trajectory files."
    )
    topology_file = st.file_uploader(
        "Topology / structure file",
        key="topology_upload",
        type=[k.lower() for k in mda._PARSERS],
    )
    trajectory_files = st.file_uploader(
        "Trajectory files",
        accept_multiple_files=True,
        key="trajectory_upload",
        type=[k.lower() for k in mda._READERS],
    )

    if topology_file is not None and st.button("Load Universe", type="primary"):
        try:
            with st.spinner("Loading MDAnalysis Universe..."):
                create_universe(topology_file, trajectory_files)
            clear_results()  # results of the old Universe
            validate_atom_selections()  # selections against the NEW Universe
            st.success("Universe loaded successfully.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to load Universe: {e}")

u = st.session_state.u

# ---------------------------------------------------------------------
# 2. Universe information
# ---------------------------------------------------------------------
if u is not None:
    with st.expander("2. Universe Information", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Atoms", f"{u.atoms.n_atoms:,}")
        col2.metric("Residues", f"{len(u.residues):,}")
        col3.metric("Segments", f"{len(u.segments):,}")
        col4.metric("Frames", f"{len(u.trajectory):,}")
        if st.session_state.trajectory_paths:
            st.subheader("Trajectory Information")
            col1, col2 = st.columns(2)
            col1.metric(
                "Number of trajectories", len(st.session_state.trajectory_paths)
            )
            col2.metric("Total frames", f"{len(u.trajectory):,}")
            st.write(f"**Timestep:** {round(u.trajectory.dt, 2)} ps")
            st.write(f"**Total time:** {round(u.trajectory.totaltime, 2)} ps")

# ---------------------------------------------------------------------
# 3. Atom selections
# ---------------------------------------------------------------------
if u is not None:
    with st.expander("3. Atom Selections", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.text_area(
                "Protein Selection",
                key="prot_sel",
                height=100,
                placeholder="protein",
                on_change=validate_atom_selections,
                help="MDAnalysis selection of the protein residues "
                "(each needs one CA atom), e.g. 'protein'",
            )
        with col2:
            st.text_area(
                "Glycan Selection",
                key="gly_sel",
                height=100,
                placeholder="resname NAG",
                on_change=validate_atom_selections,
                help="MDAnalysis selection of the glycan atoms, e.g. 'resname NAG BMA'",
            )

        results = st.session_state.get("selection_validation_results")
        if results:
            col1, col2 = st.columns(2)
            for col, name in ((col1, "Protein"), (col2, "Glycan")):
                with col:
                    st.markdown(f"#### {name} Selection")
                    result = results.get(name)
                    if result and result["valid"]:
                        st.success(f"Valid\n\n**{result['message']}**")
                    elif result:
                        st.error(result["message"])
            if st.session_state.get("selections_valid", False):
                st.success("✓ Both selections are valid and contain atoms.")
            else:
                st.error("✗ One or more selections must be corrected.")

# ---------------------------------------------------------------------
# 4. Trajectory options
# ---------------------------------------------------------------------
frames = None
frame_mode = None
frame_start = frame_stop = frame_step = None

if u is not None:
    n_frames = len(u.trajectory)
    with st.expander("4. Trajectory Options", expanded=True):
        frame_mode = st.radio(
            "Frame selection method",
            options=["Start / Stop / Step", "Frame list"],
            horizontal=True,
            key="frame_mode",
        )

        if frame_mode == "Start / Stop / Step":
            col1, col2, col3 = st.columns(3)
            with col1:
                frame_start = st.number_input(
                    "Start (blank = beginning)",
                    min_value=0,
                    max_value=n_frames - 1,
                    value=None,
                    step=1,
                    placeholder="0",
                )
            with col2:
                frame_stop = st.number_input(
                    "Stop (exclusive; blank = end)",
                    min_value=-n_frames,
                    max_value=n_frames,
                    value=None,
                    step=1,
                    placeholder=str(n_frames),
                )
            with col3:
                frame_step = st.number_input(
                    "Step (blank = every frame)",
                    min_value=1,
                    value=None,
                    step=1,
                    placeholder="1",
                )

            # same semantics as Python slicing, so negative stops work
            selected = range(n_frames)[slice(frame_start, frame_stop, frame_step)]
            if len(selected) == 0:
                st.error("This start / stop / step selects no frames.")
            else:
                frames = slice(frame_start, frame_stop, frame_step)
                st.caption(f"{len(selected)} frame(s) selected")
        else:
            frame_text = st.text_area(
                "Frames",
                value="0",
                help="Frame numbers separated by commas, spaces or new lines.",
            )
            try:
                frames = sorted({int(x) for x in frame_text.replace(",", " ").split()})
                if not frames:
                    st.error("Enter at least one frame.")
                    frames = None
                elif any(f < 0 or f >= n_frames for f in frames):
                    st.error(f"Frames must be between 0 and {n_frames - 1}.")
                    frames = None
            except ValueError:
                st.error("Invalid frame list.")
                frames = None

# ---------------------------------------------------------------------
# 5. Analysis parameters
# ---------------------------------------------------------------------
cli_inputs, cli_errors = {}, []
if u is not None:
    with st.expander("5. Analysis Parameters", expanded=True):
        cli_inputs, cli_errors = create_cli_inputs(parser)

# ---------------------------------------------------------------------
# 6. Run analysis
# ---------------------------------------------------------------------
if u is not None:
    st.header("6. Run Analysis")

    selections_valid = (
        st.session_state.get("selections_valid", False)
        and st.session_state.get("prot_sel", "").strip()
        == st.session_state.get("_validated_prot_sel")
        and st.session_state.get("gly_sel", "").strip()
        == st.session_state.get("_validated_gly_sel")
    )
    if not selections_valid:
        st.warning(
            "Enter valid protein and glycan selections before running the analysis."
        )
    frames_valid = frames is not None
    if not frames_valid:
        st.warning(
            "A valid set of trajectory frames must be selected before "
            "running the analysis."
        )
    if cli_errors:
        st.warning("Fix the analysis parameters in section 5 before running.")

    ready = selections_valid and frames_valid and not cli_errors
    if st.button(
        "Run Analysis", type="primary", use_container_width=True, disabled=not ready
    ):
        missing = [
            p
            for p in [
                st.session_state.topology_path,
                *st.session_state.trajectory_paths,
            ]
            if not p or not os.path.exists(p)
        ]
        if missing:
            st.error(f"Input file(s) no longer exist: {missing}. Reload the Universe.")
            st.stop()

        cmd = build_command(
            frame_mode, frames, frame_start, frame_stop, frame_step, cli_inputs
        )

        # fresh output directory for every run
        clear_results()
        st.session_state.outdir = tempfile.mkdtemp(prefix="glyco_")
        _remove_tree_at_exit(st.session_state.outdir)
        cmd += ["--outdir", st.session_state.outdir, "--force"]

        try:
            returncode, output = run_analysis(cmd)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to run glyco:\n\n{e}")
            st.stop()
        st.session_state.last_run = {
            "command": shlex.join(cmd),
            "returncode": returncode,
            "output": output,
        }
        st.session_state.results_zip = (
            zip_directory(st.session_state.outdir) if returncode == 0 else None
        )

    # ---- results (kept across reruns, e.g. after clicking download) ----
    last = st.session_state.last_run
    if last:
        with st.expander("Command", expanded=False):
            st.code(last["command"], language="bash")
        with st.expander("Analysis Output", expanded=False):
            st.code(last["output"] or "No output.", language="text")
        if last["returncode"] == 0:
            st.success("Analysis completed successfully.")
        else:
            st.error(f"Analysis failed (exit code {last['returncode']}).")
    if st.session_state.results_zip:
        st.download_button(
            "Download results (ZIP)",
            st.session_state.results_zip,
            file_name="glyco_results.zip",
            mime="application/zip",
        )
