"""
glyco sasa: per-residue FreeSASA areas as PDB B-factors and CSV.

Frames of a trajectory are computed in parallel with -j/--n-workers
(one single-threaded FreeSASA process per worker).
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np

from glycokit import viewer_scripts as vs
from glycokit.config import _LOAD_ERRORS, _SELECTION_ERRORS
from glycokit.console import heading, set_color_mode, style
from glycokit.mdafreesasa import FreesasaAnalysis
from glycokit.parsers import sasa_parser
from glycokit.utils import confirm_overwrite, heavy_atoms, load_universe
from glycokit.warnlog import warning_log

OUTPUT_NAMES = (
    "sasa_bfactors.pdb", "sasa_bfactors.npy", "residue_sasa.csv",
    "frame_sasa.csv")
ATOM_AREAS_NAME = "sasa_atom_areas.npy"
STATS_NAMES = (
    "sasa_histograms.csv", "sasa_statistics.csv", "sasa_bfactors_stats.pdb")
# the script names come from FreesasaAnalysis (single source of truth)
VMD_NAME = FreesasaAnalysis._vmd_script_name  # pylint: disable=protected-access
CHIMERAX_NAME = FreesasaAnalysis._chimerax_script_name  # pylint: disable=protected-access


def planned_outputs(args):
    """Files this run writes (overwrite check and tests)."""
    names = list(OUTPUT_NAMES)
    if args.write_atom_areas:
        names.append(ATOM_AREAS_NAME)
    if args.write_stats or args.write_chimerax:
        names.extend(STATS_NAMES)
    if args.write_vmd:
        names.append(VMD_NAME)
    if args.write_chimerax:
        names.append(CHIMERAX_NAME)
    if args.write_vmd:
        names.append(VMD_NAME)
        if not vs.vmd_readable(args.top):
            names.append(vs.vmd_structure_name(VMD_NAME))
    return names


def _check_frame_options(parser, args):
    """Frame-option errors that need no Universe (argparse exit 2)."""
    slice_given = any(
        v is not None
        for v in (args.frame_start, args.frame_stop, args.frame_step))
    if args.frame_list and slice_given:
        parser.error("--frames is not allowed with --start/--stop/--step")
    if args.frame_step == 0:
        parser.error("--step cannot be 0")
    if args.frame_list and any(f < 0 for f in args.frame_list):
        parser.error("--frames must be non-negative")


def _check_frames_in_range(parser, args, n_traj):
    """--frames entries must exist in the trajectory."""
    bad = [f for f in (args.frame_list or []) if f >= n_traj]
    if bad:
        parser.error(
            f"--frames {bad[:5]} out of range for {n_traj} trajectory "
            f"frame(s); valid values are 0 to {n_traj - 1}")


def main(argv=None):
    """Entry point of ``glyco sasa``; returns an exit status."""
    parser = sasa_parser()
    args = parser.parse_args(argv)
    set_color_mode(args.color)
    if args.write_chimerax:
        args.write_stats = True  # as in glyco run

    _check_frame_options(parser, args)
    for path in [args.top, *(args.traj or [])]:
        if not os.path.isfile(path):
            parser.error(f"no such file: {path!r}")

    outdir = args.outdir or f"sasa_{Path(args.top).stem}"
    if not confirm_overwrite(outdir, args, names=planned_outputs(args)):
        return 1
    os.makedirs(outdir, exist_ok=True)

    with warning_log(os.path.join(outdir, "mdanalysis_warnings.log")):
        try:
            u = load_universe(args.top, args.traj)
        except _LOAD_ERRORS as e:
            parser.error(f"could not load {args.top}: {e}")
        _check_frames_in_range(parser, args, u.trajectory.n_frames)

        try:
            ag = heavy_atoms(u.select_atoms(args.selection))
        except _SELECTION_ERRORS as e:
            parser.error(f"invalid --sel ({args.selection!r}): {e}")
        if len(ag) == 0:
            parser.error(f"--sel selected 0 heavy atoms ({args.selection!r})")

        try:
            fsa = FreesasaAnalysis(
                ag, args.sasa_probe, outdir=outdir,
                keep_atom_areas=args.write_atom_areas,
                stats_bin_width=args.stats_bin_width)
        except RuntimeError as e:  # e.g. an atom without a radius
            parser.error(str(e))

        run = {
            "start": args.frame_start,
            "stop": args.frame_stop,
            "step": args.frame_step,
            "frames": args.frame_list,
        }

        # validate --bfactor-frame before the (slow) FreeSASA pass
        n_sel = fsa.n_selected_frames(**run)
        try:
            fsa._resolve_frame_index(args.bfactor_frame, n=n_sel)  # pylint: disable=protected-access
        except (ValueError, IndexError) as e:
            parser.error(str(e))

        if not fsa.structure_valid:
            warnings.warn(
                f"FreeSASA structure check failed ({fsa.structure_message}); "
                "areas use coordinates and radii only", stacklevel=1)

        fsa.run_parallel(args.n_workers, verbose=True, **run)

        heading(f"Writing results to {style(outdir + '/', 'cyan')}")
        fsa.write_bfactors(
            frame_index=args.bfactor_frame, default_name="sasa_bfactors.pdb",
            label="SASA B-factors")
        fsa.write_residue_sasa_csv(frame_index=args.bfactor_frame)
        fsa.write_frame_sasa_csv()

        stats_pdb = None
        if args.write_stats:
            if fsa.n_frames == 1:
                warnings.warn(
                    "--write-stats with one frame: every mode is that "
                    "frame's SASA bin and every width is 0", stacklevel=1)
            fsa.write_histograms_csv(
                default_name="sasa_histograms.csv", label="SASA histograms")
            fsa.write_statistics_csv(
                default_name="sasa_statistics.csv", label="SASA statistics")
            stats_pdb = fsa.write_statistics_pdb(  # name: _stats_pdb_name
                frame_index=args.bfactor_frame,
                label="SASA statistics B-factors")

        try:
            if args.write_vmd:
                fsa.write_vmd_user_script()
            if args.write_chimerax:
                fsa.write_chimerax_script(structure=stats_pdb)
        except (ValueError, FileNotFoundError) as e:
            parser.error(str(e))  # usage message, exit status 2

        if args.write_atom_areas:
            np.save(
                os.path.join(outdir, ATOM_AREAS_NAME), fsa.results.atom_areas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
