import os
import subprocess
import sys
import tempfile
import ast
import argparse
import shlex
import json
import time

import streamlit as st
import MDAnalysis as mda
import streamlit as st

from parsers import base_parser

# =====================================================================
# Configuration
# =====================================================================

ANALYSIS_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "glyco_analysis.py",
)

PARSER_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "parsers.py",
)

parser = base_parser()

exclude_args = ["top", "traj", "prot_sel", "gly_sel", "start", "stop", "step",
                "frames", "in_folder", "outdir"]

# =====================================================================
# Helper functions
# =====================================================================

def run_analysis(command):
    """Run analysis and display all subprocess output inside an expander."""

    progress_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".json"
    ).name

    log_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".log"
    )

    env = os.environ.copy()
    env["GLYCO_PROGRESS_FILE"] = progress_file
    env["GLYCO_STREAMLIT"] = "1"

    try:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        while process.poll() is None:
            try:
                with open(progress_file, "r") as f:
                    progress_data = json.load(f)

                percent = progress_data.get("percent", 0)
                progress_bar.progress(
                    min(int(percent), 100)
                )
                status_text.text(
                    f"Analyzing: {percent:.1f}%"
                )
            except (FileNotFoundError, json.JSONDecodeError):
                pass

            time.sleep(0.2)

        log_file.flush()
        log_file.seek(0)
        output = log_file.read()

        progress_bar.empty()
        status_text.empty()

        # All warnings and subprocess output appear here.
        with st.expander("Analysis Output", expanded=False):
            st.code(output or "No output.", language="text")

        if process.returncode == 0:
            st.success("Analysis completed successfully.")
        else:
            st.error(
                f"Analysis failed (exit code {process.returncode})."
            )

    finally:
        log_file.close()

        for path in (progress_file, log_file.name):
            if os.path.exists(path):
                os.remove(path)


def remove_file(path):
    """Remove a temporary file if it exists."""

    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def clear_files():
    """Remove files associated with the current Universe."""

    remove_file(st.session_state.topology_path)

    for path in st.session_state.trajectory_paths:
        remove_file(path)

    st.session_state.topology_path = None
    st.session_state.trajectory_paths = []
    st.session_state.u = None


def save_uploaded_file(uploaded_file):
    """
    Save a Streamlit UploadedFile to a temporary file.

    The original filename extension is preserved so that
    MDAnalysis can determine the appropriate file format.
    """

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as tmp:

        tmp.write(
            uploaded_file.getvalue()
        )

        return tmp.name


def create_universe(topology_file, trajectory_files):
    """
    Save topology and trajectories and create an MDAnalysis Universe.
    """

    # Remove any previously loaded files
    clear_files()

    # -------------------------------------------------------------
    # Save topology
    # -------------------------------------------------------------

    topology_path = save_uploaded_file(
        topology_file
    )

    trajectory_paths = []

    try:

        # ---------------------------------------------------------
        # Save trajectories
        # ---------------------------------------------------------

        for trajectory_file in trajectory_files:

            trajectory_path = save_uploaded_file(
                trajectory_file
            )

            trajectory_paths.append(
                trajectory_path
            )

        # ---------------------------------------------------------
        # Create Universe
        # ---------------------------------------------------------

        if trajectory_paths:

            u = mda.Universe(
                topology_path,
                trajectory_paths,
            )

        else:

            u = mda.Universe(
                topology_path,
            )

        # ---------------------------------------------------------
        # Store in session state
        # ---------------------------------------------------------

        st.session_state.topology_path = topology_path
        st.session_state.trajectory_paths = trajectory_paths
        st.session_state.u = u

        return u

    except Exception:

        # Clean up if Universe creation fails
        remove_file(topology_path)

        for path in trajectory_paths:
            remove_file(path)

        raise


def print_argument_help(parser):
    """Print CLI help with both short and long argument names."""

    print("\nCommand-line Arguments:\n")

    for action in parser._actions:

        # Skip positional arguments with no option strings
        if not action.option_strings:
            continue

        # Display all aliases, e.g. -f, --file
        flags = ", ".join(action.option_strings)

        # Format help text
        help_text = action.help or "No description available"

        print(f"{flags:<30} {help_text}")

    print()


def get_analysis_arguments(parser):
    """
    Extract argparse arguments from an ArgumentParser instance.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser containing the analysis CLI definitions.

    Returns
    -------
    list[dict]
        List of dictionaries describing each argument.
    """

    arguments = []

    for action in parser._actions:

        # Skip help arguments
        if isinstance(action, argparse._HelpAction):
            continue

        # Skip suppressed arguments
        if action.default == argparse.SUPPRESS:
            continue

        # Use the first option string, or dest for positional args
        name = (
            action.option_strings[0]
            if action.option_strings
            else action.dest
        )

        # Determine action type
        if isinstance(action, argparse._StoreTrueAction):
            action_type = "store_true"

        elif isinstance(action, argparse._StoreFalseAction):
            action_type = "store_false"

        elif isinstance(action, argparse._AppendAction):
            action_type = "append"

        else:
            action_type = None

        # Determine argument type
        arg_type = action.type

        if arg_type is None:
            arg_type = "str"
        elif arg_type is int:
            arg_type = "int"
        elif arg_type is float:
            arg_type = "float"
        elif arg_type is str:
            arg_type = "str"
        else:
            arg_type = getattr(arg_type, "__name__", "str")


        argument = {
            "name": name,
            "dest": action.dest,
            "option_strings": action.option_strings,
            "required": action.required,
            "default": action.default,
            "type": arg_type,
            "action": action_type,
            "nargs": action.nargs,
            "choices": action.choices,
            "help": action.help or "",
        }

        arguments.append(argument)

    return arguments


def create_cli_inputs(
    parser,
    exclude=None,
):
    """
    Generate Streamlit widgets from an ArgumentParser instance.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser containing the analysis CLI definitions.

    exclude : list, optional
        Argument names or destinations to exclude.

    Returns
    -------
    dict
        Dictionary containing user-entered argument values.
    """

    if exclude is None:
        exclude = exclude_args

    # Normalize exclusions to argparse destinations
    excluded = {
        arg.lstrip("-").replace("-", "_")
        for arg in exclude
    }

    arguments = get_analysis_arguments(parser)

    values = {}

    for arg in arguments:

        name = arg["name"]
        dest = arg["dest"]

        # Skip excluded arguments
        if dest in excluded or name.lstrip("-") in excluded:
            continue

        default = arg["default"]
        arg_type = arg["type"]
        action = arg["action"]
        nargs = arg["nargs"]
        choices = arg["choices"]
        help_text = arg["help"]

        # Use the longest argument name
        label = max(
            arg.get("option_strings", [arg["name"]]),
            key=len,
        ).lstrip("-")

        # ----------------------------------------------------
        # Boolean flags
        # ----------------------------------------------------
        if action in ("store_true", "store_false"):

            default_value = action == "store_false"

            if isinstance(default, bool):
                default_value = default

            values[dest] = st.checkbox(
                label,
                value=default_value,
                help=help_text,
            )

        # ----------------------------------------------------
        # Choices
        # ----------------------------------------------------
        elif choices is not None:

            choices = list(choices)

            # Preserve an optional None default
            if default is None and not arg["required"]:
                choices = [None] + choices

            default_index = (
                choices.index(default)
                if default in choices
                else 0
            )

            values[dest] = st.selectbox(
                label,
                choices,
                index=default_index,
                help=help_text,
            )

        # ----------------------------------------------------
        # Multiple values
        # ----------------------------------------------------
        elif nargs in ("+", "*") or action == "append":

            default_values = []

            if default is not None:
                if isinstance(default, (list, tuple)):
                    default_values = list(default)
                else:
                    default_values = [default]

            text_value = ", ".join(
                str(x) for x in default_values
            )

            user_input = st.text_input(
                label,
                value=text_value,
                help=(
                    f"{help_text} "
                    "Enter multiple values separated by commas."
                ),
            )

            if user_input.strip():

                raw_values = [
                    x.strip()
                    for x in user_input.split(",")
                    if x.strip()
                ]

                # Convert values to the expected type
                if arg_type in ("int", "float"):
                    converter = int if arg_type == "int" else float

                    values[dest] = [
                        converter(x) for x in raw_values
                    ]
                else:
                    values[dest] = raw_values

            else:
                values[dest] = []

        # ----------------------------------------------------
        # Integer
        # ----------------------------------------------------
        elif arg_type == "int":

            default_value = 0 if default is None else int(default)

            values[dest] = st.number_input(
                label,
                value=default_value,
                step=1,
                help=help_text,
            )

        # ----------------------------------------------------
        # Float
        # ----------------------------------------------------
        elif arg_type == "float":

            default_value = (
                0.0 if default is None else float(default)
            )

            values[dest] = st.number_input(
                label,
                value=default_value,
                step=0.1,
                help=help_text,
            )

        # ----------------------------------------------------
        # String / fallback
        # ----------------------------------------------------
        else:

            default_value = (
                "" if default is None else str(default)
            )

            values[dest] = st.text_input(
                label,
                value=default_value,
                help=help_text,
            )

    return values

# ============================================================
# Helper: Validate Atom Selections
# ============================================================


def validate_atom_selections():
    """Validate protein and glycan selections on widget change."""

    u = st.session_state.get("u")

    prot_sel = st.session_state.get("prot_sel", "").strip()
    gly_sel = st.session_state.get("gly_sel", "").strip()

    # Store trimmed selections
    st.session_state["prot_sel"] = prot_sel
    st.session_state["gly_sel"] = gly_sel

    if u is None:
        return

    results = {}
    validation_passed = True

    for name, selection in [
        ("Protein", prot_sel),
        ("Glycan", gly_sel),
    ]:

        if not selection:
            results[name] = {
                "valid": False,
                "message": "Selection cannot be empty.",
            }
            validation_passed = False
            continue

        try:
            atoms = u.select_atoms(selection)

            if len(atoms) == 0:
                results[name] = {
                    "valid": False,
                    "message": f"0 atoms selected\n\n`{selection}`",
                }
                validation_passed = False

            else:
                results[name] = {
                    "valid": True,
                    "message": f"{len(atoms):,} atoms",
                }

        except Exception as e:
            results[name] = {
                "valid": False,
                "message": (
                    f"Invalid selection\n\n"
                    f"`{selection}`\n\n{e}"
                ),
            }
            validation_passed = False

    # Store validation results
    st.session_state["selection_validation_results"] = results
    st.session_state["selections_valid"] = validation_passed

    if validation_passed:
        st.session_state["_validated_prot_sel"] = prot_sel
        st.session_state["_validated_gly_sel"] = gly_sel
    else:
        # Clear previously validated selections if validation fails
        st.session_state.pop("_validated_prot_sel", None)
        st.session_state.pop("_validated_gly_sel", None)


# =====================================================================
# Session state
# =====================================================================

if "u" not in st.session_state:
    st.session_state.u = None

if "topology_path" not in st.session_state:
    st.session_state.topology_path = None

if "trajectory_paths" not in st.session_state:
    st.session_state.trajectory_paths = []


# =====================================================================
# Application
# =====================================================================


st.title("MDAnalysis Analysis")


# =====================================================================
# 1. Load files
# =====================================================================
with st.expander("1. Load Structure and Trajectories", expanded=True):

    st.write(
        """
    Upload an MDAnalysis-compatible topology/structure file and,
    optionally, one or more trajectory files.
    """
    )


    topology_file = st.file_uploader(
        "Topology / structure file",
        key="topology_upload",
        type=list(mda._PARSERS),
    )


    trajectory_files = st.file_uploader(
        "Trajectory files",
        accept_multiple_files=True,
        key="trajectory_upload",
        type=list(mda._READERS)
    )


    # =====================================================================
    # Create Universe
    # =====================================================================

    if topology_file is not None:

        if st.button(
            "Load Universe",
            type="primary",
        ):

            try:

                with st.spinner(
                    "Loading MDAnalysis Universe..."
                ):

                    u = create_universe(
                        topology_file,
                        trajectory_files,
                    )

                st.success(
                    "Universe loaded successfully."
                )

            except Exception as e:

                st.error(
                    f"Failed to load Universe: {e}"
                )


# =====================================================================
# 2. Universe information
# =====================================================================

if st.session_state.u is not None:

    u = st.session_state.u

    with st.expander("2. Universe Information", expanded=True):

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Atoms",
            f"{u.atoms.n_atoms:,}",
        )

        col2.metric(
            "Residues",
            f"{len(u.residues):,}",
        )

        col3.metric(
            "Segments",
            f"{len(u.segments):,}",
        )

        col4.metric(
            "Frames",
            f"{len(u.trajectory):,}",
        )

        # -------------------------------------------------------------
        # Trajectory information
        # -------------------------------------------------------------

        if st.session_state.trajectory_paths:

            st.subheader("Trajectory Information")

            col1, col2 = st.columns(2)

            col1.metric(
                "Number of trajectories",
                len(st.session_state.trajectory_paths),
            )

            col2.metric(
                "Total frames",
                f"{len(u.trajectory):,}",
            )

            st.write(
                f"**Timestep:** {round(u.trajectory.dt, 2)} ps"
            )

            st.write(
                f"**Total time:** {round(u.trajectory.totaltime, 2)} ps"
            )


# ============================================================
# 3. Atom Selections
# ============================================================

if st.session_state.get("u") is not None:

    with st.expander("3. Atom Selections", expanded=True):

        # Do not initialize prot_sel or gly_sel with default values.
        # Placeholder text is displayed until the user enters a selection.

        col1, col2 = st.columns(2)

        with col1:
            st.text_area(
                "Protein Selection",
                key="prot_sel",
                height=100,
                placeholder="protein",
                help=(
                    "MDAnalysis selection string for the protein. "
                    "Examples: 'protein', 'protein and name CA'"
                ),
                on_change=validate_atom_selections,
            )

        with col2:
            st.text_area(
                "Glycan Selection",
                key="gly_sel",
                height=100,
                placeholder="resname NAG",
                help=(
                    "MDAnalysis selection string for the glycan. "
                    "Examples: 'resname NAG', 'resname MAN BMA'"
                ),
                on_change=validate_atom_selections,
            )

        # ========================================================
        # Display Validation Results
        # ========================================================

        results = st.session_state.get(
            "selection_validation_results"
        )

        if results:

            col1, col2 = st.columns(2)

            for col, name in [
                (col1, "Protein"),
                (col2, "Glycan"),
            ]:

                with col:
                    st.markdown(f"#### {name} Selection")

                    result = results.get(name)

                    if result:
                        if result["valid"]:
                            st.success(
                                f"Valid\n\n**{result['message']}**"
                            )
                        else:
                            st.error(result["message"])

            # Overall validation status
            if st.session_state.get("selections_valid", False):
                st.success(
                    "✓ Both selections are valid and contain atoms."
                )
            else:
                st.error(
                    "✗ One or more selections must be corrected."
                )

# ============================================================
# 4. Trajectory Options
# ============================================================

if st.session_state.u is not None:

    with st.expander("4. Trajectory Options", expanded=True):

        frame_mode = st.radio(
            "Frame selection method",
            options=["Start / Stop / Step", "Frame list"],
            horizontal=True,
            key="frame_mode",
        )

        frames = None
        frame_start = None
        frame_stop = None
        frame_step = None

        if frame_mode == "Start / Stop / Step":

            col1, col2, col3 = st.columns(3)

            with col1:
                frame_start = st.number_input(
                    "Start (blank = beginning)",
                    min_value=0,
                    value=None,
                    step=1,
                    max_value=len(st.session_state.u.trajectory)-1,
                    placeholder="0",
                )

            with col2:
                frame_stop = st.number_input(
                    "Stop (exclusive; blank = end)",
                    min_value=-len(st.session_state.u.trajectory),
                    value=None,
                    step=1,
                    max_value=len(st.session_state.u.trajectory),
                    placeholder=str(len(st.session_state.u.trajectory)),
                )

            with col3:
                frame_step = st.number_input(
                    "Step  (blank = every frame)",
                    min_value=1,
                    value=None,
                    step=1,
                    placeholder="1",
                )

            # None means use the start of the trajectory
            effective_start = (
                0
                if frame_start is None
                else frame_start
            )

            # None means use the end of the trajectory
            effective_stop = (
                len(st.session_state.u.trajectory)
                if frame_stop is None
                else frame_stop
            )

            # None means use every frame
            effective_step = (
                1
                if frame_step is None
                else frame_step
            )

            if effective_stop <= effective_start:
                st.error("Stop must be greater than Start.")
                frames = None
            else:
                frames = slice(
                    frame_start,
                    frame_stop,
                    frame_step,
                )

        else:
            frame_text = st.text_area(
                "Frames",
                value="0, 10, 20, 50, 100",
                help="Enter frame numbers separated by commas, spaces, or new lines.",
            )

            try:
                tokens = frame_text.replace(",", " ").split()
                frames = sorted(set(int(x) for x in tokens))

                n_frames = len(st.session_state.u.trajectory)

                if any(f < 0 or f >= n_frames for f in frames):
                    st.error(f"Frames must be between 0 and {n_frames - 1}.")
                    frames = None

                elif not frames:
                    st.error("Enter at least one frame.")
                    frames = None

            except ValueError:
                st.error("Invalid frame list.")
                frames = None


# =====================================================================
# 5. Specify Parameters
# =====================================================================

# if st.session_state.u is not None:
    # cli_args = create_cli_inputs(ANALYSIS_SCRIPT)
if st.session_state.u is not None:

    with st.expander("5. Analysis Parameters", expanded=True):
        cli_inputs = create_cli_inputs(
            parser,
    )

# =====================================================================
# 6. Run Analysis
# =====================================================================

if st.session_state.get("u") is not None:

    st.header("6. Run Analysis")

    # -------------------------------------------------------------
    # Check that selections have been validated
    # -------------------------------------------------------------

    selections_valid = (
        st.session_state.get("selections_valid", False)
        and st.session_state.get("prot_sel", "").strip()
        == st.session_state.get("_validated_prot_sel")
        and st.session_state.get("gly_sel", "").strip()
        == st.session_state.get("_validated_gly_sel")
    )

    if not selections_valid:
        st.warning(
            "Enter valid protein and glycan selections before "
            "running the analysis."
        )

    # -------------------------------------------------------------
    # Check that frame selection is valid
    # -------------------------------------------------------------

    frames_valid = (
        frames is not None
        and len(st.session_state.u.trajectory[frames]) > 0
    )

    if not frames_valid:
        st.warning(
            "A valid set of trajectory frames must be selected "
            "before running the analysis."
        )

    # -------------------------------------------------------------
    # Analysis button
    # -------------------------------------------------------------

    run_clicked = st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=not (selections_valid and frames_valid),
    )

    if run_clicked:

        # =========================================================
        # Retrieve inputs
        # =========================================================

        topology_path = st.session_state.get("topology_path")

        trajectory_paths = st.session_state.get(
            "trajectory_paths", []
        )

        prot_sel = st.session_state.get("prot_sel", "").strip()
        gly_sel = st.session_state.get("gly_sel", "").strip()

        # =========================================================
        # Final validation
        # =========================================================

        if not topology_path or not os.path.exists(topology_path):
            st.error("Topology file is missing or no longer exists.")
            st.stop()

        for trajectory_path in trajectory_paths:
            if not os.path.exists(trajectory_path):
                st.error(
                    f"Trajectory file no longer exists:\n"
                    f"{trajectory_path}"
                )
                st.stop()

        if not selections_valid:
            st.error("Atom selections must be validated.")
            st.stop()

        if not frames_valid:
            st.error("No valid trajectory frames were selected.")
            st.stop()

        # =========================================================
        # Build command
        # =========================================================

        cmd = [
            sys.executable,
            ANALYSIS_SCRIPT,
        ]

        # ---------------------------------------------------------
        # Retrieve parser argument definitions
        # ---------------------------------------------------------

        arguments = get_analysis_arguments(parser)

        # Map destinations to their actual CLI option strings
        argument_map = {
            arg["dest"]: arg
            for arg in arguments
        }

        # ---------------------------------------------------------
        # Topology
        # ---------------------------------------------------------

        cmd.extend([
            "--top",
            topology_path,
        ])

        # ---------------------------------------------------------
        # Trajectories
        # ---------------------------------------------------------

        for trajectory_path in trajectory_paths:
            cmd.extend([
                "--traj",
                trajectory_path,
            ])

        # ---------------------------------------------------------
        # Frames
        # ---------------------------------------------------------

        if frame_mode == "Start / Stop / Step":

            # Only pass --start, --stop, --step if the user specified it
            if frame_start is not None:
                cmd.extend(["--start", str(frame_start)])

            if frame_stop is not None:
                cmd.extend(["--stop", str(frame_stop)])

            if frame_step is not None:
                cmd.extend(["--step", str(frame_step)])


        else:
            cmd.append("--frames")
            cmd.extend(str(frame) for frame in frames)

        # ---------------------------------------------------------
        # Atom selections
        # ---------------------------------------------------------

        cmd.extend(["--prot-sel", prot_sel])
        cmd.extend(["--gly-sel", gly_sel])

        # ---------------------------------------------------------
        # Automatically generated CLI parameters
        # ---------------------------------------------------------

        for dest, value in cli_inputs.items():

            # Skip values handled separately above
            if dest in {
                "top",
                "traj",
                "frames",
                "prot_sel",
                "gly_sel",
            }:
                continue

            # Skip empty values
            if value is None or value == "":
                continue

            # Retrieve the actual argparse definition
            arg = argument_map.get(dest)

            if arg is None:
                continue

            option_strings = arg.get("option_strings", [])

            if not option_strings:
                continue

            # Use the longest option string (e.g. --output-file)
            option = max(option_strings, key=len)

            action = arg["action"]
            default = arg["default"]

            # -----------------------------------------------------
            # Boolean flags
            # -----------------------------------------------------

            if action == "store_true":
                if value:
                    cmd.append(option)

            elif action == "store_false":
                if not value:
                    cmd.append(option)

            # -----------------------------------------------------
            # Multiple values
            # -----------------------------------------------------

            elif isinstance(value, list):

                if value:
                    cmd.append(option)
                    cmd.extend(str(x) for x in value)

            # -----------------------------------------------------
            # Standard arguments
            # -----------------------------------------------------

            else:

                # Only include values that differ from the default
                if value != default:
                    cmd.extend([
                        option,
                        str(value),
                    ])


        output_box = st.empty()
        output_lines = []

        try:

            # =========================================================
            # Display command
            # =========================================================

            with st.expander("Command", expanded=False):
                st.code(shlex.join(cmd), language="bash")

            # =========================================================
            # Run analysis and display live progress
            # =========================================================

            # st.subheader("Analysis Output")

            return_code = run_analysis(cmd)

        except Exception as e:

            st.error(
                f"Failed to run analysis script:\n\n{e}"
            )
