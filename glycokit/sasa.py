"""
glyco sasa: per-residue FreeSASA areas as PDB B-factors and CSV.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np

from glycokit.config import _LOAD_ERRORS, _SELECTION_ERRORS
from glycokit.console import heading, set_color_mode, style
from glycokit.mdafreesasa import FreesasaAnalysis
from glycokit.parsers import sasa_parser
from glycokit.utils import (
    confirm_overwrite, heavy_atoms, load_universe, resolve_workers,
)
from glycokit.warnlog import warning_log

OUTPUT_NAMES = (
    "sasa_bfactors.pdb", "sasa_bfactors.npy", "residue_sasa.csv",
    "frame_sasa.csv")
ATOM_AREAS_NAME = "sasa_atom_areas.npy"
STATS_NAMES = (
    "sasa_histograms.csv", "sasa_statistics.csv", "sasa_bfactors_stats.pdb")


def planned_outputs(args):
    """Files this run writes (overwrite check and tests)."""
    names = list(OUTPUT_NAMES)
    if args.write_atom_areas:
        names.append(ATOM_AREAS_NAME)
    if args.write_stats:
        names.extend(STATS_NAMES)
    return names


def _n_selected(u, args):
    if args.frame_list:
        return len(args.frame_list)
    return len(
        range(u.trajectory.n_frames)[slice(
            args.frame_start, args.frame_stop, args.frame_step)])


def main(argv=None):
    """Entry point of ``glyco sasa``; returns an exit status."""
    parser = sasa_parser()
    args = parser.parse_args(argv)
    set_color_mode(args.color)

    if args.frame_list and any(
            v is not None
            for v in (args.frame_start, args.frame_stop, args.frame_step)):
        parser.error("--frames is not allowed with --start/--stop/--step")
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

        n_sel = _n_selected(u, args)
        try:
            fsa._resolve_frame_index(args.bfactor_frame, n=n_sel)  # pylint: disable=protected-access
        except (ValueError, IndexError) as e:
            parser.error(str(e))
        n = min(resolve_workers(args.n_workers), max(1, n_sel))

        if not fsa.structure_valid:
            warnings.warn(
                f"FreeSASA structure check failed ({fsa.structure_message}); "
                "areas use coordinates and radii only", stacklevel=1)

        run = {
            "start": args.frame_start,
            "stop": args.frame_stop,
            "step": args.frame_step,
            "frames": args.frame_list
        }
        if n > 1:
            fsa.run(backend="multiprocessing", n_workers=n, **run)
        else:
            fsa.run(
                verbose=True,
                progressbar_kwargs={"desc": "Running FreeSASA Analysis"}, **run)

        heading(f"Writing results to {style(outdir + '/', 'cyan')}")
        fsa.write_bfactors(
            frame_index=args.bfactor_frame, default_name="sasa_bfactors.pdb",
            label="SASA B-factors")
        fsa.write_residue_sasa_csv(frame_index=args.bfactor_frame)
        fsa.write_frame_sasa_csv()
        if args.write_atom_areas:
            np.save(
                os.path.join(outdir, "sasa_atom_areas.npy"),
                fsa.results.atom_areas)

        if args.write_stats:
            if fsa.n_frames == 1:
                warnings.warn(
                    "--write-stats with one frame: every mode is the "
                    "frame's SASA and every width is 0", stacklevel=1)
            fsa.write_histograms_csv(
                default_name="sasa_histograms.csv", label="SASA histograms")
            fsa.write_statistics_csv(
                default_name="sasa_statistics.csv", label="SASA statistics")
            fsa.write_statistics_pdb(
                frame_index=args.bfactor_frame,
                default_name="sasa_bfactors_stats.pdb",
                label="SASA statistics B-factors")
        if args.write_atom_areas:
            np.save(
                os.path.join(outdir, ATOM_AREAS_NAME), fsa.results.atom_areas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
