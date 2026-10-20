from pathlib import Path

import numpy as np
import MDAnalysis as mda


def _tcl_quote(value):
    """Wrap a string as a Tcl braced argument."""
    value = str(value).replace("\\", "\\\\")
    value = value.replace("{", "\\{").replace("}", "\\}")
    return "{" + value + "}"


def write_vmd_user_script(
    atoms,
    data,
    output="load_user_attributes.vmd",
    frame_offset=0,
):
    """
    Generate a VMD Tcl script that loads the topology and
    trajectory from an MDAnalysis Universe or AtomGroup, then
    assigns per-residue data to VMD's 'user' attribute.

    Uses one atom selection and one bulk 'user' assignment
    per frame for improved performance.

    Parameters
    ----------
    atoms : MDAnalysis.Universe or MDAnalysis.AtomGroup
        Universe or AtomGroup defining the residues to annotate.

    data : array-like
        Shape (n_residues, n_frames), or (n_residues,) for
        single-frame data.

    output : str or Path
        Output VMD Tcl script path.

    frame_offset : int
        VMD frame index corresponding to data column 0.

    Returns
    -------
    Path
        Path to the generated VMD script.
    """

    # ---------------------------------------------------------
    # Get Universe and AtomGroup
    # ---------------------------------------------------------
    if isinstance(atoms, mda.Universe):
        u = atoms
        atoms = u.atoms

    elif isinstance(atoms, mda.core.groups.AtomGroup):
        u = atoms.universe

    else:
        raise TypeError(
            "atoms must be an MDAnalysis Universe or AtomGroup"
        )

    # ---------------------------------------------------------
    # Get topology path
    # ---------------------------------------------------------
    topology = getattr(u, "filename", None)

    if not topology:
        raise ValueError(
            "Could not determine topology filename from Universe"
        )

    topology = Path(topology).resolve()

    if not topology.is_file():
        raise FileNotFoundError(topology)

    # ---------------------------------------------------------
    # Get trajectory path(s)
    # ---------------------------------------------------------
    reader = u.trajectory

    trajectory_files = getattr(reader, "filenames", None)

    if trajectory_files is None:
        filename = getattr(reader, "filename", None)
        trajectory_files = [filename] if filename else []

    trajectory_files = [
        Path(p).resolve()
        for p in trajectory_files
        if p is not None
    ]

    if not trajectory_files:
        raise ValueError(
            "Could not determine trajectory filename(s)"
        )

    for path in trajectory_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------
    data = np.asarray(data, dtype=float)


    if data.ndim == 1:
        data = data[:, None]

    if data.ndim != 2:
        raise ValueError("data must be a 1D or 2D array")

    residues = atoms.residues
    n_residues = len(residues)

    if data.shape[0] != n_residues:
        raise ValueError(
            f"Data has {data.shape[0]} rows, but "
            f"AtomGroup contains {n_residues} residues"
        )

    if not np.all(np.isfinite(data)):
        raise ValueError("data contains NaN or infinite values")

    n_frames = data.shape[1]

    # ---------------------------------------------------------
    # Build one flat atom-index list and residue mapping
    # ---------------------------------------------------------
    atom_indices = []
    atom_residue_rows = []

    for res_i, residue in enumerate(residues):
        res_indices = residue.atoms.indices.tolist()

        atom_indices.extend(res_indices)
        atom_residue_rows.extend([res_i] * len(res_indices))

    atom_indices = np.asarray(atom_indices, dtype=int)
    atom_residue_rows = np.asarray(atom_residue_rows, dtype=int)

    # Tcl selection string, created once
    index_string = " ".join(map(str, atom_indices))

    # ---------------------------------------------------------
    # Write VMD Tcl script
    # ---------------------------------------------------------
    output = Path(output)

    with output.open("w") as f:

        f.write("# Auto-generated VMD Tcl script\n")
        f.write("# Bulk per-frame assignment of residue values\n\n")

        # Load topology; capture actual VMD molecule ID
        f.write(
            f"set molid [mol new {_tcl_quote(topology)} "
            "waitfor all]\n\n"
        )

        # Load trajectory file(s)
        for traj in trajectory_files:
            f.write(
                f"mol addfile {_tcl_quote(traj)} "
                "waitfor all molid $molid\n"
            )

        f.write("\n")
        f.write("mol modselect 0 $molid protein\n")
        f.write("mol modstyle 0 $molid NewCartoon\n")
        f.write("mol modcolor 0 $molid User\n")
        f.write("\n")

        # Check VMD trajectory length against data length
        f.write(f"set n_data_frames {n_frames}\n")
        f.write("set n_vmd_frames [molinfo $molid get numframes]\n\n")

        f.write("if {$n_vmd_frames == $n_data_frames + 1} {\n")
        f.write("    # Delete duplicate first frame\n")
        f.write("    animate delete beg 0 end 0 $molid\n")
        f.write("    puts \"Deleted duplicate first frame.\"\n")
        f.write("} elseif {$n_vmd_frames == $n_data_frames} {\n")
        f.write("    puts \"Frame counts match; no frame deleted.\"\n")
        f.write("} else {\n")
        f.write(
            '    error "Frame mismatch: VMD has $n_vmd_frames frames, '
            'but data has $n_data_frames frames. '
            'Expected equal counts or exactly one extra VMD frame."\n'
        )
        f.write("}\n\n")

        data_max = float(np.max(data))

        f.write("# Set User color scale\n")
        f.write("mol modcolor 0 $molid User\n")
        f.write(f"mol scaleminmax 0 0 0 {data_max:.10g}\n\n")

        # Assign all atom values in one operation per frame
        for frame in range(n_frames):

            vmd_frame = frame_offset + frame

            # Map residue-level data to the selected atoms
            atom_values = data[atom_residue_rows, frame]

            # One Tcl list containing all atom attributes
            values_string = " ".join(
                f"{value:.10g}" for value in atom_values
            )

            f.write(f"# Data frame {frame} -> VMD frame {vmd_frame}\n")

            f.write(
                f'set sel [atomselect $molid '
                f'"index {index_string}" frame {vmd_frame}]\n'
            )

            f.write(f"$sel set user {{{values_string}}}\n")
            f.write("$sel delete\n\n")

        f.write(
            'puts "Finished assigning per-residue user values."\n'
        )

    print(f"VMD script written to: {output}")

    return output