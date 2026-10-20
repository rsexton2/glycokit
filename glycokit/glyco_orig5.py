#!/usr/bin/env python3
# ruff: noqa: EXE001
import argparse
import csv
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, repeat
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase, Results
from scipy.spatial import cKDTree

from glycokit.mdafreesasa import FreesasaAnalysis
from glycokit.parsers import base_parser

VMD_SCRIPT_NAME = "load_user_attributes.vmd"
DEFAULT_OUTDIR_NAME = "glyco_output"

# Hydrogen atom names: optional leading digits then 'H' (H1, HB2, 1HB, HO2).
# Anchored at the START so heavy atoms like NH1, OH, CH2 are not matched.
_HYDROGEN_NAME = re.compile(r"^\d*H")

# Protein residue names accepted by the original scripts
_V1_RESNAMES = ("ALA ARG ASN ASP CYS GLN GLU GLY HSD HIS HIE HID "
                "ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL")
_V2_RESNAMES = _V1_RESNAMES + " HSE HSP"

_BOX_STATE = {}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _v2_cylinder_blocked(P32, G32, B64, i, g, j, r):
    """Vectorized c_count_points: True where blocker j lies in the (i, g) cylinder."""
    vec   = P32[i] - G32[g]                                   # float32 (pt2 - pt1)
    const = np.float32(r) * np.linalg.norm(vec, axis=1)       # float32
    q_pt1 = B64[j] - G32[g]                                   # float64
    q_pt2 = B64[j] - P32[i]                                   # float64
    return ((np.einsum("ij,ij->i", q_pt1, vec) > 0)
            & (np.einsum("ij,ij->i", q_pt2, vec) < 0)
            & (np.linalg.norm(np.cross(q_pt1, vec), axis=1) <= const))


def _first_altloc(ag):
    """Keep blank altlocs plus the first altloc letter seen in each residue."""
    try:
        alt = np.asarray(ag.altLocs, dtype=str)
    except (AttributeError, mda.exceptions.NoDataError):
        return ag
    keep = np.ones(len(ag), dtype=bool)
    first = {}
    for k, (res, a) in enumerate(zip(ag.resindices, alt)):
        if a.strip():
            first.setdefault(res, a)
            keep[k] = (a == first[res])
    return ag[keep]


def _legacy_coords(positions):
    """
    float32 MDAnalysis positions -> the exact float64 value GLYCO v1 gets from
    float(line[30:38]). Valid for standard 3-decimal PDB coordinates.
    """
    return np.round(np.asarray(positions, dtype=np.float64), 3)


def _build_x_buckets(positions):
    """
    Reproduce glyco_mod.py's integer-x index (gen_pro_dict's outdict2).

    Keys are int(x), i.e. truncation toward zero, which matches
    int(float(line[30:38])) in GLYCO v1. Values are tuples of indices
    into `positions`, so the index is immutable, cheap to pickle and
    always consistent with the coordinates the workers receive.
    """
    positions = np.asarray(positions, dtype=np.float64)
    buckets = {}
    if positions.size:
        for index, x in enumerate(positions[:, 0].tolist()):
            buckets.setdefault(int(x), []).append(index)
    return {key: tuple(value) for key, value in buckets.items()}


def _init_box_worker(
    start_positions,
    start_rows,
    start_residue_keys,
    gly_positions,
    blocker_positions,
    blocker_residue_keys,
    blocker_x_buckets,
    cutoff,
    n_gly,
):
    """
    Initialize one process with plain NumPy/Python data.

    No MDAnalysis objects are passed to worker processes.
    """
    _BOX_STATE.clear()

    _BOX_STATE.update(
        start_positions=np.asarray(
            start_positions,
            dtype=np.float64,
        ),
        start_rows=np.asarray(
            start_rows,
            dtype=np.int64,
        ),
        start_residue_keys=start_residue_keys,
        gly_positions=np.asarray(
            gly_positions,
            dtype=np.float64,
        ),
        blocker_positions=np.asarray(
            blocker_positions,
            dtype=np.float64,
        ),
        blocker_residue_keys=blocker_residue_keys,
        blocker_x_buckets=blocker_x_buckets,
        cutoff=float(cutoff),
        n_gly=int(n_gly),
    )


def _box_chunk(start_range):
    """
    Process a chunk of start atoms.

    Returns:
        candidate keys,
        visible keys,
        number of blocked pairs,
        sum of blocked-pair distances
    """
    state = _BOX_STATE

    P = state["start_positions"]
    start_rows = state["start_rows"]
    start_residue_keys = state["start_residue_keys"]

    G = state["gly_positions"]
    B = state["blocker_positions"]
    blocker_residue_keys = state["blocker_residue_keys"]
    x_buckets = state["blocker_x_buckets"]

    cutoff = state["cutoff"]
    n_gly = state["n_gly"]

    lo, hi = start_range

    candidate_keys = set()
    visible_keys = set()

    n_blocked = 0
    blocked_distance_sum = 0.0

    for start_index in range(lo, hi):
        start_position = P[start_index]
        residue_row = int(start_rows[start_index])
        start_residue_key = tuple(
            start_residue_keys[start_index]
        )

        for gly_index, gly_position in enumerate(G):
            difference = gly_position - start_position
            distance = float(np.linalg.norm(difference))

            # Original GLYCO condition: distance < cutoff.
            if distance >= cutoff:
                continue

            key = residue_row * n_gly + gly_index
            candidate_keys.add(key)

            # glyco_mod.py check_protein(): dist == 0 -> (True, None),
            # i.e. visible
            if distance == 0.0:
                visible_keys.add(key)
                continue

            blocked = False
            step = 1.0 / distance

            # Matches:
            # numpy.arange(0.0, 1.0, 1 / dist)
            for t in np.arange(0.0, 1.0, step):
                point = start_position + t * difference

                # Matches Python int() truncation toward zero.
                x, y, z = (
                    int(point[0]),
                    int(point[1]),
                    int(point[2]),
                )

                # # Match glyco_mod.py:
                # # pro_dict.get(x) + pro_dict.get(x-1)
                # # + pro_dict.get(x+1)
                # blocker_indices = (
                #     x_buckets.get(x, ())
                #     + x_buckets.get(x - 1, ())
                #     + x_buckets.get(x + 1, ())
                # )
                # Match glyco_mod.py:
                #   pro_dict.get(x) + pro_dict.get(x-1) + pro_dict.get(x+1)
                # chain() avoids list/tuple concatenation entirely and keeps
                # the same visiting order as GLYCO v1.
                blocker_indices = chain(
                    x_buckets.get(x, ()),
                    x_buckets.get(x - 1, ()),
                    x_buckets.get(x + 1, ()),
                )

                for blocker_index in blocker_indices:
                    blocker_residue_key = tuple(
                        blocker_residue_keys[blocker_index]
                    )

                    # Original GLYCO excludes blockers belonging to
                    # the residue from which the line starts.
                    if blocker_residue_key == start_residue_key:
                        continue

                    blocker_x, blocker_y, blocker_z = (
                        B[blocker_index]
                    )

                    if (
                        abs(y - blocker_y) <= 1.0
                        and abs(z - blocker_z) <= 1.0
                        and abs(x - blocker_x) <= 1.0
                    ):
                        blocked = True
                        break

                if blocked:
                    break

            if blocked:
                n_blocked += 1
                blocked_distance_sum += distance
            else:
                visible_keys.add(
                    residue_row * n_gly + gly_index
                )

    return (
        np.asarray(
            sorted(candidate_keys),
            dtype=np.int64,
        ),
        np.asarray(
            sorted(visible_keys),
            dtype=np.int64,
        ),
        n_blocked,
        blocked_distance_sum,
    )


def atom_identity(atom):
    try:
        chain = str(atom.chainID).strip()
    except (AttributeError, mda.exceptions.NoDataError):
        chain = ""

    return (
        int(atom.id),
        str(atom.name).strip(),
        str(atom.resname).strip(),
        chain,
        str(atom.resid).strip(),
        tuple(round(float(x), 3) for x in atom.position),
    )


def pdb_atom_ids(atom_group):
    """
    PDB-style (1-based) serial numbers for every atom in an AtomGroup.
    Uses the topology 'ids' attribute when present, otherwise falls back to
    index + 1 (valid only for sequential PDB-like input).
    """
    try:
        return np.asarray(atom_group.ids, dtype=np.int64)
    except (AttributeError, mda.exceptions.NoDataError):
        return np.asarray(atom_group.indices, dtype=np.int64) + 1


def configure_warning_log(path):
    """
    Route Python warnings (incl. MDAnalysis') into a log file. Called from
    main() rather than at import time, so importing this module has no
    side effects.
    """
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    abspath = os.path.abspath(path)
    for existing in warnings_logger.handlers:
        if isinstance(existing, logging.FileHandler) and existing.baseFilename == abspath:
            return
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    warnings_logger.addHandler(handler)
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False


def heavy_atoms(ag, legacy=False):
    """
    Return `ag` without hydrogens. An atom is a hydrogen if its element is
    H/D (when the topology has elements) OR its name starts with H (after
    optional digits). Name matching is anchored at the start so that heavy
    atoms such as ARG NH1/NH2, TYR OH or TRP CH2 are kept.

    legacy=True reproduces the original GLYCO filter instead: drop every atom
    whose name contains an 'H' ANYWHERE (which also removes NH1/NH2, OH,
    CH2, ...). Used by --compat and --compat-v2.
    """
    names = np.asarray(ag.names, dtype=str)
    if legacy:
        is_h = np.fromiter(("H" in n for n in names), dtype=bool, count=len(names))
        return ag[~is_h]
    is_h = np.fromiter(
        (bool(_HYDROGEN_NAME.match(n)) for n in names),
        dtype=bool, count=len(names),
    )
    try:
        elements = np.char.upper(np.asarray(ag.elements, dtype=str))
        is_h |= np.isin(elements, ("H", "D"))
    except (AttributeError, mda.exceptions.NoDataError):
        pass
    return ag[~is_h]


def find_structures(indir, extensions):
    """
    Find independent structure files in `indir` matching `extensions`
    (a list of extensions without leading dots, e.g. ['pdb', 'gro']).
    Each match is treated as a standalone, single-file structure -- no
    trajectory pairing or chain reading involved.
    """
    files = []
    for ext in extensions:
        files.extend(sorted(glob.glob(os.path.join(indir, f'*.{ext}'))))
    return list(dict.fromkeys(files))


def validate_arguments(cfg, error, check_inputs=True):
    """
    Validate (and normalize) parameters on `cfg`, which can be an argparse
    Namespace or a GlycoAnalysis instance. Needs no Universe, so the CLI can
    run it BEFORE loading any files and fail with a clean usage message.

    `error` is a callable that reports a message and does not return
    (parser.error for the CLI, or a function raising ValueError).
    With check_inputs=False the input-file checks are skipped (library use,
    where a Universe is passed in directly).

    Sets cfg.structures (list of files for --indir, else []) and normalizes
    cfg.frame_list to a list of ints or None.
    """
    top_given = cfg.top is not None
    traj_given = bool(cfg.traj)
    indir_given = cfg.indir is not None
    cfg.structures = []

    if check_inputs:
        if not top_given and not indir_given:
            error("one of -a/--top(+-b/--traj) or --indir must be given")
        if indir_given and (top_given or traj_given):
            error("--indir is mutually exclusive with -a/--top and -b/--traj")
        if traj_given and not top_given:
            error("-b/--traj requires -a/--top")

        if top_given:
            if not os.path.isfile(cfg.top):
                error(f"-a/--top: no such file: {cfg.top!r}")
            for t in (cfg.traj or []):
                if not os.path.isfile(t):
                    error(f"-b/--traj: no such file: {t!r}")
        else:
            if not os.path.isdir(cfg.indir):
                error(f"--indir: no such directory: {cfg.indir!r}")
            extensions = [e.strip().lstrip('.') for e in cfg.in_types.split(',')
                          if e.strip()]
            structures = find_structures(cfg.indir, extensions)
            if not structures:
                error(
                    f"--indir: no files with extension(s) {extensions} "
                    f"found in {cfg.indir!r}"
                )
            cfg.structures = structures

    slice_given = any(v is not None
                      for v in (cfg.frame_start, cfg.frame_stop, cfg.frame_step))
    frames_given = bool(cfg.frame_list)

    if frames_given and slice_given:
        error("--frames is not allowed with --start/--stop/--step")

    if indir_given and (frames_given or slice_given):
        error(
            "--frames/--start/--stop/--step are not meaningful with "
            "--indir (each structure is a single, independent frame)"
        )

    for name in ('distance_cutoff', 'area_cutoff', 'radius', 'probe'):
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            error(f"--{name.replace('_', '-')} must be positive, got {value}")

    if cfg.frame_step is not None and cfg.frame_step == 0:
        error("--step cannot be 0")

    start, stop, step = cfg.frame_start, cfg.frame_stop, cfg.frame_step
    if (start is not None and stop is not None and start >= 0 and stop >= 0
            and (step is None or step > 0) and start >= stop):
        error(f"--start ({start}) must be less than --stop ({stop})")

    if frames_given:
        try:
            frame_ints = [int(f) for f in cfg.frame_list]
        except (TypeError, ValueError):
            error(f"--frames must be integers, got {cfg.frame_list!r}")
        if any(f < 0 for f in frame_ints):
            error("--frames must be non-negative")
        cfg.frame_list = frame_ints
    else:
        cfg.frame_list = None

    if getattr(cfg, 'compat', False) and getattr(cfg, 'compat_v2', False):
        error("--compat and --compat-v2 are mutually exclusive")

    if getattr(cfg, 'compat', False) and getattr(cfg, 'write_blockers', False):
        error("--write-blockers is not supported with --compat (the box "
              "test does not identify blocking atoms); --write-excluded "
              "is supported")

    if not cfg.outdir:
        cfg.outdir = default_outdir(cfg)


def default_outdir(cfg):
    """Pick an output directory name from the inputs when -o isn't given."""
    if cfg.indir is not None:
        return f"glyco_{Path(cfg.indir).resolve().name}"
    if cfg.top is not None:
        return f"glyco_{Path(cfg.top).stem}"
    return DEFAULT_OUTDIR_NAME  # library use: Universe passed directly


def planned_outputs(outdir, write_vmd=False, write_excluded=False,
                    write_blockers=False):
    """Paths of the files a run writes into `outdir`."""
    names = ['b_factors.pdb', 'b_factors.npy', 'frame_summary.csv',
             'residue_summary.csv']
    if write_excluded:
        names.append('excluded_glycan_atoms.csv')
    if write_blockers:
        names.append('blockers.csv')
    if write_vmd:
        names.append(VMD_SCRIPT_NAME)
    return [os.path.join(outdir, n) for n in names]


def confirm_overwrite(outdir, force=False, write_vmd=False,
                      write_excluded=False, write_blockers=False):
    """
    If any of the files a run would write already exist in `outdir`, prompt
    the user to overwrite them unless force=True. Returns True if execution
    should proceed, False if the user declined (or no prompt is possible).

    The check is on the specific output files, not on the directory, so the
    default outdir of '.' does not prompt every time.
    """
    existing = [os.path.basename(p)
                for p in planned_outputs(outdir, write_vmd,
                                         write_excluded, write_blockers)
                if os.path.exists(p)]
    if not existing:
        return True

    listing = ", ".join(existing)
    if force:
        print(f"[info] overwriting in {outdir!r}: {listing} (--force given).")
        return True

    if not sys.stdin.isatty():
        print(
            f"[error] {outdir!r} already contains: {listing}; no TTY is "
            f"available to prompt, pass --force to overwrite "
            f"non-interactively.",
            file=sys.stderr,
        )
        return False

    while True:
        try:
            answer = input(
                f"{outdir!r} already contains: {listing}. Overwrite? [y/N] "
            ).strip().lower()
        except EOFError:
            return False
        if answer in ('y', 'yes'):
            return True
        if answer in ('', 'n', 'no'):
            return False
        print("Please answer 'y' or 'n'.")


def load_universe(top, traj=None):
    """
    Load a Universe. Several trajectories go through the MDAnalysis chain
    reader and must all match the topology's atom count.
    """
    if not traj:
        return mda.Universe(top)
    return mda.Universe(top, traj if len(traj) > 1 else traj[0])


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

class GlycoAnalysis(AnalysisBase):
    """
    Run Glyco and other useful output functions.

    Semantics: --ps selects the residues the line-of-sight STARTING POINTS
    come from (CA atoms, further restricted to surface residues). Blocker
    candidates are chosen independently with --bs (default: all protein heavy
    atoms, not surface-limited), except atoms in the starting residue itself.

    Counting modes:
        default      CA starts, cylinder test (--cr), own residue excluded.
        --compat     GLYCO v1: all-heavy-atom starts with a per-residue union,
                     integer cutoff, legacy hydrogen filter, FreeSASA with the
                     rest of the structure present, and the original
                     integer-truncated +-1 A box blocking test.
        --compat-v2  GLYCO-2: same counting/filters as --compat but a float
                     cutoff and the cylinder blocking test, in which atoms of
                     the start residue CAN block and the radius is inclusive.
    """

    # Results are stored on the instance (not merged from workers) and the
    # nested FreeSASA run covers the whole slice, so this is serial-only.
    _analysis_algorithm_is_parallelizable = False

    @classmethod
    def get_supported_backends(cls):
        return ('serial',)

    def __init__(self, u, parser=None, verbose=False, args=None, **kwargs):
        super().__init__(u.trajectory, verbose=verbose)
        self.u = u
        self.parser = parser
        self.progress_file = os.environ.get("GLYCO_PROGRESS_FILE")
        self._run_slice = {}
        self._blocker_cache = None
        self._box_cache = None

        # Single source of truth for defaults: base_parser() itself, so
        # library usage (no parser passed) and CLI usage always agree.
        defaults = vars(base_parser().parse_args([]))

        # Only touch sys.argv (via parser.parse_args()) if a parser was
        # supplied WITHOUT an already-parsed `args`. Pure library usage
        # (GlycoAnalysis(u, prot_sel=..., gly_sel=...)) never parses
        # sys.argv, since it may contain unrelated arguments (pytest's own
        # CLI args, a notebook kernel's argv, ...).
        source = args if args is not None else parser
        self.apply_params(defaults=defaults, args=source, **kwargs)

        # deprecated spelling: compat_box=True is now simply compat=True
        if getattr(self, "compat_box", False):
            self.compat = True

        self._validate_args()
        # belt and braces: never let a None outdir reach a path/f-string
        if not self.outdir:
            self.outdir = default_outdir(self)
        self._validate_atom_selections()

    # -- configuration ------------------------------------------------------

    def _error(self, message):
        """
        Report a configuration error. Uses self.parser.error() (prints
        usage, exits status 2) when running from the CLI; raises
        ValueError instead when used as a library with no parser, since
        calling sys.exit() on an importing caller's process would be a
        surprising and unrecoverable side effect for a Python API.
        """
        if self.parser is not None:
            self.parser.error(message)
        raise ValueError(message)

    def _validate_args(self):
        # Library usage: a Universe was passed in directly and no input
        # files were named, so there is nothing about inputs to validate.
        library_mode = (self.parser is None and self.top is None
                        and self.indir is None and not self.traj)
        validate_arguments(self, self._error, check_inputs=not library_mode)

    def apply_params(self, args=None, defaults=None, **kwargs):
        """
        Merge parameters onto `self`, lowest to highest precedence:
        `defaults`, then `args` (an argparse Namespace, or an
        ArgumentParser, which is parsed from sys.argv), then kwargs.
        """
        values = dict(defaults or {})
        if args is not None:
            if isinstance(args, argparse.ArgumentParser):
                args = args.parse_args()
            values.update(vars(args))
        values.update(kwargs)

        for name, value in values.items():
            setattr(self, name, value)

    def run(self, start=None, stop=None, step=None, frames=None, **kwargs):
        """
        Like AnalysisBase.run, but falls back to the frame slice given on
        the command line / as parameters when none is passed explicitly. The
        slice is remembered so the nested FreeSASA run covers exactly the
        same frames as this one.
        """
        if all(v is None for v in (start, stop, step, frames)):
            start, stop, step = self.frame_start, self.frame_stop, self.frame_step
            frames = self.frame_list
        self._run_slice = {"start": start, "stop": stop, "step": step,
                           "frames": frames}
        return super().run(**self._run_slice, **kwargs)

    def _v2_pairs(self, start_idx):
        """(i, g) pairs exactly as c_worker selects them: float32 norm < float32 cutoff."""
        P32 = self.start_atoms.positions[start_idx].astype(np.float32)
        G32 = self.gly_atoms.positions.astype(np.float32)
        cut32 = np.float32(self.distance_cutoff)
        # generous float64 prefilter, then the exact float32 test
        pairs = cKDTree(P32.astype(np.float64)).sparse_distance_matrix(
            cKDTree(G32.astype(np.float64)), float(self.distance_cutoff) + 1e-3,
            output_type="ndarray")
        i = pairs["i"].astype(np.intp)
        g = pairs["j"].astype(np.intp)
        norm32 = np.linalg.norm(G32[g] - P32[i], axis=1)      # float32, as c_worker
        keep = norm32 < cut32
        return i[keep], g[keep], P32, G32

    def _validate_atom_selections(self):
        """
        Validate the protein, glycan and blocker selections against the
        Universe. Per-selection results are kept in
        self.selection_validation_results (see selection_report()); any
        failure is reported via self._error.
        """
        prot_sel = (self.prot_sel or "").strip()
        gly_sel = (self.gly_sel or "").strip()
        block_sel = (self.block_sel or "").strip()

        # --compat / --compat-v2: restrict start and blocker residues to the
        # residue names the original scripts used (ANDed with the user's
        # selection). The glycan selection is left untouched.
        if getattr(self, "compat", False) or getattr(self, "compat_v2", False):
            names = _V2_RESNAMES if getattr(self, "compat_v2", False) else _V1_RESNAMES
            resname_filter = f"resname {names}"
            if prot_sel:
                prot_sel = f"({prot_sel}) and ({resname_filter})"
            if block_sel:
                block_sel = f"({block_sel}) and ({resname_filter})"

        self.prot_sel = prot_sel
        self.gly_sel = gly_sel
        self.block_sel = block_sel

        results = {}
        for name, selection in (("protein", prot_sel), ("glycan", gly_sel),
                                ("blockers", block_sel)):
            if not selection:
                results[name] = {"valid": False,
                                 "message": "Selection cannot be empty."}
                continue
            try:
                atoms = self.u.select_atoms(f"({selection})")
            except Exception as e:  # noqa: BLE001
                results[name] = {
                    "valid": False,
                    "message": f"Invalid selection ('{selection}'): {e}",
                }
                continue
            if len(atoms) == 0:
                results[name] = {"valid": False,
                                 "message": f"0 atoms selected ('{selection}')"}
            else:
                results[name] = {"valid": True,
                                 "message": f"{len(atoms):,} atoms"}

        if results["protein"]["valid"]:
            n_ca = len(self.u.select_atoms(f"({prot_sel}) and name CA"))
            if n_ca == 0:
                results["protein"] = {
                    "valid": False,
                    "message": f"no CA atoms in protein selection ('{prot_sel}')",
                }

        self.selection_validation_results = results
        self.selections_valid = all(r["valid"] for r in results.values())

        if not self.selections_valid:
            problems = "; ".join(f"{name}: {r['message']}"
                                 for name, r in results.items() if not r["valid"])
            self._error(f"invalid atom selection(s): {problems}")

        self._validated_prot_sel = prot_sel
        self._validated_gly_sel = gly_sel
        self._validated_block_sel = block_sel

        if self.sasa_env:
            try:
                self.u.select_atoms(f"({self.sasa_env})")
            except Exception as e:  # noqa: BLE001
                self._error(f"invalid --sasa-env selection ('{self.sasa_env}'): {e}")

    def selection_report(self):
        """Human-readable summary of the validated selections."""
        lines = ["Atom selections:"]
        for name, r in self.selection_validation_results.items():
            flag = "ok" if r["valid"] else "INVALID"
            lines.append(f"  {name}: {flag} - {r['message']}")
        return "\n".join(lines)

    # -- run lifecycle -------------------------------------------------------

    def _prepare(self):
        self.frame_count = 0
        os.makedirs(self.outdir, exist_ok=True)
        self._write_progress(0, self.n_frames)
        if self._verbose:
            print(self.selection_report())

        prot_sel = self._validated_prot_sel
        gly_sel = self._validated_gly_sel
        block_sel = self._validated_block_sel

        # Counting modes:
        #   default      - modern: CA starts, cylinder test (own residue excluded)
        #   --compat     - GLYCO v1: legacy semantics + original box test
        #   --compat-v2  - GLYCO-2:  legacy semantics + cylinder test
        self.compat = bool(self.compat)
        self.compat_v2 = bool(self.compat_v2)
        self._box = self.compat                        # box blocking test
        self._legacy = self.compat or self.compat_v2   # shared legacy semantics
        # GLYCO-2 cylinder quirks: atoms of the start residue CAN block, and
        # the radius boundary is inclusive (distance <= r)
        self._v2_cyl = self.compat_v2

        # Legacy modes reproduce the original scripts' quirks: every heavy
        # atom of a surface residue is a line-of-sight start (union of
        # visible glycan atoms per residue), the 'H anywhere in the name'
        # filter for starts/blockers/glycan atoms, and FreeSASA run with the
        # rest of the structure present. Only v1 (--compat) also truncates
        # the distance cutoff to an integer; GLYCO-2 uses a float.
        self._cutoff = (float(int(self.distance_cutoff)) if self.compat
                        else float(self.distance_cutoff))
        self._legacy_box_cutoff = float(int(self.distance_cutoff))   # v1: int(args.cutoff)
        # if compat, inclusive query becomes strict "<"
        if (self.compat) or (self.compat_v2):
            self._cutoff = float(np.nextafter(self._cutoff, 0.0))

        # --ps selects the START residues only: their heavy atoms (self.selag)
        # define which residues are analysed/reported, and their CA atoms
        # (self.prot_atoms) are the output columns.
        self.selag = heavy_atoms(self.u.select_atoms(f"({prot_sel})"))
        self.prot_atoms = self.u.select_atoms(f"({prot_sel}) and name CA")

        # Blocker candidates come from --bs (default: all protein), NOT from
        # --ps, and are not limited to surface residues.
        self.blocker_atoms = heavy_atoms(self.u.select_atoms(f"({block_sel})"),
                                         legacy=self._legacy)
        self.gly_atoms = heavy_atoms(self.u.select_atoms(f"({gly_sel})"),
                                     legacy=self._legacy)

        self._residue_resindices = np.unique(self.selag.resindices)
        self._n_res_rows = len(self._residue_resindices)

        # FreeSASA runs on the start residues PLUS the blocker atoms (and any
        # --sasa-env atoms) so that residues outside --ps still occlude the
        # start residues; areas are reported for the start residues only.
        env = self.sasa_env or ("not water" if self._legacy else None)
        sasa_sel = f"({prot_sel}) or ({block_sel})"
        if env:
            sasa_sel += f" or ({env})"
        sasa_ag = heavy_atoms(self.u.select_atoms(sasa_sel))
        sasa_res = np.unique(sasa_ag.resindices)

        fsa = FreesasaAnalysis(sasa_ag, self.probe).run(
            verbose=self._verbose,
            progressbar_kwargs={'desc': 'Running FreeSASA Analysis'},
            **self._run_slice,
        )
        print(
        "[SASA diagnostic]",
        f"""mode={(
            'compat'
            if self.compat
            else 'compat-v2'
            if self.compat_v2
            else 'default'
        )}""",
        f"universe_atoms={len(self.u.atoms)}",
        f"sasa_atoms={len(sasa_ag)}",
        f"sasa_residues={sasa_ag.n_residues}",
        f"start_residues={len(self._residue_resindices)}",
        f"structure_valid={fsa.structure_valid}",
        flush=True,
        )

        if not fsa.structure_valid:
            msg = ("FreeSASA structure did not match the MDAnalysis residues; "
                "SASA used MDAnalysis vdW radii instead of ProtOr, so areas "
                "will not match the freesasa executable")
            if self._legacy:
                self._error(msg)       # --compat / --compat-v2 should reproduce the originals
            else:
                warnings.warn(msg, stacklevel=2)

        areas = np.asarray(
            fsa.results.residue_areas,
            dtype=float,
        )

        expected_shape = (
            self.n_frames,
            sasa_ag.n_residues,
        )

        if areas.shape != expected_shape:
            raise ValueError(
                "Unexpected FreeSASA residue-area shape: "
                f"got {areas.shape}, expected {expected_shape}"
            )

        # if areas.shape[1] != sasa_ag.n_residues:
        #     raise ValueError(
        #         f"FreeSASA returned {areas.shape[1]} residue areas, "
        #         f"but the selection contains {sasa_ag.n_residues} residues"
        #     )

        # FreesasaAnalysis now guarantees that residue_areas follows
        # sasa_ag.residues order.
        sasa_area_by_resindex = {
            int(residue.resindex): areas[:, i]
            for i, residue in enumerate(sasa_ag.residues)
        }

        missing_resindices = [
            int(resindex)
            for resindex in self._residue_resindices
            if int(resindex) not in sasa_area_by_resindex
        ]

        if missing_resindices:
            raise ValueError(
                "SASA is missing residues required by the protein selection: "
                f"{missing_resindices[:10]}"
            )

        # self.residue_areas = np.column_stack(
        #     [
        #         sasa_area_by_resindex[int(resindex)]
        #         for resindex in self._residue_resindices
        #     ]
        # ).T
        self.residue_areas = areas[
            :,
            np.searchsorted(
                sasa_res,
                self._residue_resindices,
            ),
        ]

        surface_count = int(
            np.count_nonzero(
                self.residue_areas[0] >= self.area_cutoff
            )
        )

        print(
            "[SASA diagnostic]",
            f"surface_residues={surface_count}",
            f"area_cutoff={self.area_cutoff}",
            flush=True,
        )

        # Blockers may belong to residues outside --ps, so same-residue
        # exclusion compares MDAnalysis resindices directly.
        self._blocker_resindex = np.asarray(self.blocker_atoms.resindices)

        if not np.isin(self.prot_atoms.resindices, self._residue_resindices).all():
            raise ValueError(
                "Some CA atoms belong to residues not present in the "
                "FreeSASA selection (prot_sel heavy-atom set); prot_sel "
                "selections for CA and heavy atoms must cover the same "
                "residues."
            )
        if len(np.unique(self.prot_atoms.resindices)) != self.prot_atoms.n_atoms:
            raise ValueError(
                "More than one 'CA' atom per residue in the protein "
                "selection; results are per residue and need exactly one."
            )
        self._prot_atom_res_row = np.searchsorted(self._residue_resindices,
                                                  self.prot_atoms.resindices)

        # Line-of-sight starting points: the CA atoms (default), or all
        # heavy atoms of the residue (legacy modes). Only those in SURFACE
        # residues are used each frame.
        if self._legacy:
            self.start_atoms = heavy_atoms(self.selag, legacy=True)
            self._start_res_row = np.searchsorted(self._residue_resindices,
                                                  self.start_atoms.resindices)
        else:
            self.start_atoms = self.prot_atoms
            self._start_res_row = self._prot_atom_res_row
        self._start_resindex = np.asarray(self.start_atoms.resindices)

        self._start_residue_keys = self._residue_keys(
            self.start_atoms
        )

        self._blocker_residue_keys = self._residue_keys(
            self.blocker_atoms
        )

        n_ca = self.prot_atoms.n_atoms
        self.all_counts = np.zeros([self.n_frames, n_ca])

        self.results = Results()
        self.results.residue_counts = self.all_counts
        self.results.n_surface_residues = np.zeros(self.n_frames, dtype=int)
        self.results.n_blocked_pairs = np.zeros(self.n_frames, dtype=int)
        self.results.mean_blocked_distance = np.full(self.n_frames, np.nan)

        # Blocker / box-visibility data for the frame used by the per-atom
        # CSV writers is cached during the run (see _single_frame).
        # Must come after all_counts exists: _resolve_frame_index uses it.
        self._blocker_cache = None
        self._box_cache = None
        self._output_k = self._resolve_frame_index(self.bfactor_frame)

        # throttle progress-file writes: ~100 updates over the whole run
        # instead of one fopen+json.dump+os.replace per frame
        self._progress_every = max(1, self.n_frames // 100)

        print(
              "[compat diagnostic]",
              f"mode={'compat' if self.compat else 'compat-v2' if self.compat_v2 else 'default'}",
              f"radius={self.radius}",
              f"legacy={self._legacy}",
              f"box={self._box}",
              f"start_atoms={len(self.start_atoms)}",
              f"blocker_atoms={len(self.blocker_atoms)}",
              f"glycan_atoms={len(self.gly_atoms)}",
              f"protein_CA={len(self.prot_atoms)}",
              f"surface_rows_possible={self._n_res_rows}",
              flush=True,
        )

    def _legacy_sasa_atoms(self, include_hetatm=False):
        """Atoms the freesasa CLI uses on the whole PDB with default options."""
        sel = "all" if include_hetatm else "record_type ATOM"
        return _first_altloc(heavy_atoms(self.u.select_atoms(sel)))

    def _single_frame(self):
        k = self._frame_index

        start_idx = self.surface_start_indices(k)
        print(
        "[compat diagnostic]",
        f"frame={k}",
        f"surface_start_atoms={len(start_idx)}",
        f"surface_residues={len(np.unique(self._start_res_row[start_idx]))}",
        flush=True,
        )

        self.results.n_surface_residues[k] = len(
            np.unique(self._start_res_row[start_idx])
        )

        if self._box:
            # GLYCO v1 box test: counts come from the box visibility
            # cand_keys, vis_keys, n_blocked, mean_d = self._box_visibility(start_idx)
            cand_keys, vis_keys, n_blocked, mean_d = (
                self._box_visibility(
                    start_idx,
                    workers=self.box_workers,
                )
            )

            self.all_counts[k] = self._counts_from_visible_keys(vis_keys)

            compat_counts = self.all_counts[k]

            print(
                "[glyco compat debug]",
                "nonzero residues:",
                int(np.count_nonzero(compat_counts)),
                "total visible glycan atoms:",
                int(compat_counts.sum()),
                "unique visible keys:",
                len(vis_keys),
                flush=True,
            )

            self.results.n_blocked_pairs[k] = n_blocked
            self.results.mean_blocked_distance[k] = mean_d

            # keep the output frame's visibility for the excluded-atom writer
            if k == self._output_k:
                self._box_cache = (k, start_idx, cand_keys, vis_keys)

            print(
            "[compat diagnostic]",
            f"candidate_keys={len(cand_keys)}",
            f"visible_keys={len(vis_keys)}",
            f"reported_nonzero={np.count_nonzero(self.all_counts[k])}",
            f"reported_total={int(self.all_counts[k].sum())}",
            flush=True,
            )
            print(
            "[compat debug]",
            "surface residues:",
            int(self.results.n_surface_residues[k]),
            "candidate keys:",
            len(cand_keys),
            "visible keys:",
            len(vis_keys),
            "nonzero residues:",
            int(np.count_nonzero(self.all_counts[k])),
            "total visible glycan atoms:",
            int(self.all_counts[k].sum()),
            flush=True,
        )
        else:
            blockers, dists, gtree = self.get_blockers(start_idx)
            self.all_counts[k] = self.count_glycan_neighbors(
                blockers, gtree, start_idx
            )

            # per-frame summary: unique blocked (start atom, glycan) pairs
            # and their mean distance
            if len(blockers):
                n_gly = len(self.gly_atoms)
                pair_key = (blockers[:, 0].astype(np.int64) * n_gly
                            + blockers[:, 2].astype(np.int64))
                _, first = np.unique(pair_key, return_index=True)
                self.results.n_blocked_pairs[k] = len(first)
                self.results.mean_blocked_distance[k] = float(dists[first].mean())

            # keep the output frame's blockers for the CSV writers
            if k == self._output_k:
                self._blocker_cache = (k, start_idx, blockers, dists)

        self.frame_count += 1
        if (self.frame_count % self._progress_every == 0
                or self.frame_count == self.n_frames):
            self._write_progress(self.frame_count, self.n_frames)

    def _conclude(self):
        os.makedirs(self.outdir, exist_ok=True)
        self.write_bfactors(frame_index=self.bfactor_frame)
        self.write_frame_summary()
        self.write_residue_csv(frame_index=self.bfactor_frame)

        if self.write_excluded:
            self.write_excluded_glycan_atoms_csv(frame_index=self.bfactor_frame)
        # blockers.csv needs per-atom blockers, which the box test lacks
        # (validate_arguments already rejects --compat with it)
        if self.write_blockers and not self._box:
            self.write_blockers_csv(frame_index=self.bfactor_frame)

        # release memory
        self._blocker_cache = None
        self._box_cache = None

        if self.write_vmd:
            self.write_vmd_user_script()

    # -- geometry ------------------------------------------------------------

    def outermost(self, frame_index=None):
        """
        Residue rows (columns of residue_areas) that are SURFACE residues in
        one analyzed frame, i.e. SASA >= area_cutoff.
        """
        if frame_index is None:
            frame_index = self._frame_index
        return np.flatnonzero(self.residue_areas[frame_index] >= self.area_cutoff)

    def surface_start_indices(self, frame_index=None):
        """Indices (into self.start_atoms) of start atoms in surface residues."""
        surf_res_rows = self.outermost(frame_index)
        return np.flatnonzero(np.isin(self._start_res_row, surf_res_rows))

    def _iter_pairs(self, P, gtree, ca_chunk):
        """
        Yield (i, j, dist) arrays of start/glycan atom pairs within the
        distance cutoff, computed for chunks of P at a time so the full
        pair list for a large glycoprotein never exists in memory at once.
        i indexes P, j indexes the glycan atoms.
        """
        for s in range(0, len(P), ca_chunk):
            sub = P[s:s + ca_chunk]
            pairs = cKDTree(sub).sparse_distance_matrix(
                gtree, self._cutoff, output_type="ndarray"
            )
            yield (pairs["i"].astype(np.intp) + s,
                   pairs["j"].astype(np.intp),
                   pairs["v"])

    def get_blockers(self, start_idx=None, chunk=20000, ca_chunk=2000):
        """
        For each (start atom, glycan atom) pair within the distance cutoff,
        find the protein heavy atoms lying between them that would block
        that line of sight (cylinder test of radius --cr).

        Segment endpoints: start atoms (CA, or all heavy atoms in the legacy
        modes) of SURFACE residues this frame (`start_idx`, indices into
        self.start_atoms; computed if None). Column 0 of the result is in
        the GLOBAL self.start_atoms index space, which is what
        count_glycan_neighbors expects.

        Candidate blockers: the --bs heavy atoms (self.blocker_atoms; by
        default all protein), independent of --ps and not just surface ones.
        By default, atoms belonging to the same residue as the segment's
        start atom are excluded (its own N, C, CB... sit within ~1.5 A of the
        CA and would otherwise "block" nearly every segment). With
        --compat-v2 this exclusion is NOT applied and the radius test is
        inclusive (d <= r), as in GLYCO-2's c_count_points. Column 1 holds
        the blocker's index into self.blocker_atoms and is informational only.

        Returns (blockers (n, 3), distances (n,), gtree or None).
        """
        if start_idx is None:
            start_idx = self.surface_start_indices()

        P = np.ascontiguousarray(self.start_atoms.positions[start_idx], dtype=np.float64)
        B = np.ascontiguousarray(self.blocker_atoms.positions, dtype=np.float64)
        G = np.ascontiguousarray(self.gly_atoms.positions, dtype=np.float64)

        gtree = cKDTree(G) if len(G) else None
        empty = (np.empty((0, 3), dtype=np.intp), np.empty(0, dtype=np.float64))
        if len(P) == 0 or len(B) == 0 or len(G) == 0:
            return (*empty, gtree)

        btree = cKDTree(B)
        start_resindices = self._start_resindex[start_idx]   # per row of P
        blocker_resindices = self._blocker_resindex          # per row of B
        r2 = self.radius ** 2

        out = []
        dist_out = []

        for ii, gg, distances in self._iter_pairs(P, gtree, ca_chunk):
            for s in range(0, len(ii), chunk):
                i_c = ii[s:s + chunk]
                g_c = gg[s:s + chunk]
                d_c = distances[s:s + chunk]

                b = P[i_c] - G[g_c]
                half = 0.5 * np.linalg.norm(b, axis=-1)
                mid = 0.5 * (P[i_c] + G[g_c])

                # 2. Candidate blockers near the segment midpoint
                cand = btree.query_ball_point(mid, half + self.radius)
                counts = np.fromiter(map(len, cand), dtype=np.intp, count=len(cand))
                if counts.sum() == 0:
                    continue

                jj = np.concatenate(cand).astype(np.intp, copy=False)
                rep = np.repeat(np.arange(len(i_c)), counts)

                # same-residue exclusion (GLYCO-2 does not do this; its
                # c_worker never compares residues)
                if not self._v2_cyl:
                    other_res = blocker_resindices[jj] != start_resindices[i_c[rep]]
                    jj = jj[other_res]
                    rep = rep[other_res]
                    if len(jj) == 0:
                        continue

                # 3. Exact blocker test (point-to-segment distance).
                # The start atom itself (if it is in B) has a = 0, hence
                # t = 0, and is rejected by t > 0 in every mode.
                bb = b[rep]
                a = P[i_c[rep]] - B[jj]

                seg_len2 = np.maximum(np.einsum("ij,ij->i", bb, bb), 1e-12)
                t = np.einsum("ij,ij->i", a, bb) / seg_len2
                perp = a - t[:, None] * bb
                d2 = np.einsum("ij,ij->i", perp, perp)

                in_radius = (d2 <= r2) if self._v2_cyl else (d2 < r2)
                keep = (t > 0) & (t < 1) & in_radius

                if keep.any():
                    out.append(
                        np.stack(
                            [
                                start_idx[i_c[rep][keep]],  # -> self.start_atoms index space
                                jj[keep],                   # index into self.blocker_atoms
                                g_c[rep][keep],             # -> self.gly_atoms index space
                            ],
                            axis=1,
                        )
                    )
                    dist_out.append(d_c[rep][keep])

        if not out:
            return (*empty, gtree)

        res = np.concatenate(out)
        dist_res = np.concatenate(dist_out)
        order = np.lexsort((res[:, 2], res[:, 1], res[:, 0]))

        return res[order], dist_res[order], gtree

    @staticmethod
    def _residue_keys(atom_group):
        try:
            chains = np.asarray(
                atom_group.chainIDs,
                dtype=str,
            )
        except (AttributeError, mda.exceptions.NoDataError):
            chains = np.full(
                len(atom_group),
                "",
                dtype=object,
            )

        return [
            (
                str(resname).strip(),
                str(chain).strip(),
                str(resid).strip(),
            )
            for resname, chain, resid in zip(
                atom_group.resnames,
                chains,
                atom_group.resids,
            )
        ]

    def count_glycan_neighbors(self, blockers, gtree=None, start_idx=None, chunk=20000):
        """
        Count non-blocked glycan neighbors, returned per CA atom (length
        len(self.prot_atoms)); residues that are not surface residues get 0.

        Default: for each surface CA, the number of glycan atoms within the
        cutoff whose line of sight from that CA is not blocked.

        --compat-v2 (and the cylinder part of any legacy mode): for each
        surface residue, the number of UNIQUE glycan atoms visible from at
        least one of the residue's heavy atoms (the original GLYCO counting).
        The value is broadcast to the residue's CA.

        blockers[:, 0] and blockers[:, 2] are in the index spaces of
        self.start_atoms and self.gly_atoms respectively.
        """
        P = self.start_atoms.positions
        n_start = len(P)
        n_gly = len(self.gly_atoms)

        counts = np.zeros(len(self.prot_atoms), dtype=np.intp)

        if start_idx is None:
            start_idx = self.surface_start_indices()
        if n_start == 0 or n_gly == 0 or len(start_idx) == 0:
            return counts

        if gtree is None:
            gtree = cKDTree(self.gly_atoms.positions)

        if len(blockers):
            excluded = np.unique(
                blockers[:, 0].astype(np.int64) * n_gly
                + blockers[:, 2].astype(np.int64)
            )
        else:
            excluded = np.empty(0, dtype=np.int64)

        per_start = np.zeros(n_start, dtype=np.intp)
        visible_keys = []

        for s in range(0, len(start_idx), chunk):
            idx = start_idx[s:s + chunk]

            neighbors = gtree.query_ball_point(P[idx], self._cutoff)
            sizes = np.fromiter(
                (len(x) for x in neighbors), dtype=np.intp, count=len(neighbors),
            )
            if sizes.sum() == 0:
                continue

            start_ids = np.repeat(idx.astype(np.int64), sizes)
            gly_ids = np.concatenate(neighbors).astype(np.int64, copy=False)

            if len(excluded):
                keep = ~np.isin(start_ids * n_gly + gly_ids, excluded)
                start_ids = start_ids[keep]
                gly_ids = gly_ids[keep]

            if self._legacy:
                visible_keys.append(np.unique(
                    self._start_res_row[start_ids].astype(np.int64) * n_gly + gly_ids
                ))
            else:
                per_start += np.bincount(start_ids, minlength=n_start)

        if not self._legacy:
            return per_start  # start atoms ARE the CA atoms

        if not visible_keys:
            return counts
        keys = np.unique(np.concatenate(visible_keys))
        per_res = np.bincount(keys // n_gly, minlength=self._n_res_rows)
        return per_res[self._prot_atom_res_row].astype(np.intp)

    # def _box_visibility(self, start_idx=None, pair_chunk=5000, ca_chunk=2000):
    #     """
    #     Original GLYCO v1 box blocking test (--compat); see
    #     count_glycan_neighbors_box for the geometry.

    #     Returns (cand_keys, vis_keys, n_blocked, mean_blocked_distance).
    #     Keys are sorted unique int64 values residue_row * n_gly + glycan_index:
    #       cand_keys - glycan atoms within the cutoff of a start atom of the row
    #       vis_keys  - the subset not blocked from at least one such start atom
    #                   (the residue's final union)
    #     """
    #     if start_idx is None:
    #         start_idx = self.surface_start_indices()

    #     empty = np.empty(0, dtype=np.int64)
    #     P = np.round(self.start_atoms.positions[start_idx].astype(np.float64), 3)
    #     B = np.round(self.blocker_atoms.positions.astype(np.float64), 3)
    #     G = np.round(self.gly_atoms.positions.astype(np.float64), 3)
    #     n_gly = len(G)
    #     if len(P) == 0 or len(B) == 0 or n_gly == 0:
    #         return empty, empty, 0, np.nan

    #     start_rows = self._start_res_row[start_idx]         # for counting
    #     start_resindices = self._start_resindex[start_idx]  # for self-exclusion
    #     blocker_resindices = self._blocker_resindex
    #     btree = cKDTree(B)
    #     gtree = cKDTree(G)

    #     cand_parts = []
    #     vis_parts = []
    #     n_blocked = 0
    #     dist_sum = 0.0

    #     for ii, gg, _ in self._iter_pairs(P, gtree, ca_chunk):
    #         for s in range(0, len(ii), pair_chunk):
    #             i_c = ii[s:s + pair_chunk]
    #             g_c = gg[s:s + pair_chunk]

    #             diff = G[g_c] - P[i_c]
    #             dist = np.linalg.norm(diff, axis=1)
    #             ok = (dist > 0) & (dist < self._cutoff)   # original: dist < cutoff
    #             i_c, g_c, diff, dist = i_c[ok], g_c[ok], diff[ok], dist[ok]
    #             if len(i_c) == 0:
    #                 continue

    #             # every pair that reaches here is a distance candidate
    #             cand_parts.append(np.unique(
    #                 start_rows[i_c].astype(np.int64) * n_gly + g_c
    #             ))

    #             # numpy.arange(0.0, 1.0, 1/dist): n = ceil(dist) samples at k*leng
    #             leng = 1.0 / dist
    #             n = np.ceil(1.0 / leng).astype(np.int64)
    #             total = int(n.sum())
    #             pair_of = np.repeat(np.arange(len(i_c)), n)
    #             k_in = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    #             t = k_in * leng[pair_of]
    #             pts = P[i_c][pair_of] + t[:, None] * diff[pair_of]
    #             cells = np.trunc(pts).astype(np.int64)          # int() truncation

    #             uniq, inv = np.unique(cells, axis=0, return_inverse=True)
    #             inv = inv.reshape(-1)
    #             near = btree.query_ball_point(uniq.astype(np.float64), r=1.0, p=np.inf)
    #             has = np.zeros(len(uniq), dtype=bool)
    #             rmin = np.full(len(uniq), -1, dtype=np.int64)
    #             rmax = np.full(len(uniq), -1, dtype=np.int64)
    #             for c, nb in enumerate(near):
    #                 if nb:
    #                     rows = blocker_resindices[nb]
    #                     has[c] = True
    #                     rmin[c] = rows.min()
    #                     rmax[c] = rows.max()

    #             # blocked if some box neighbour belongs to a DIFFERENT residue
    #             srow = start_resindices[i_c][pair_of]
    #             only_own = (rmin[inv] == srow) & (rmax[inv] == srow)
    #             hit = has[inv] & ~only_own
    #             blocked = np.bincount(pair_of, weights=hit, minlength=len(i_c)) > 0

    #             vis = ~blocked
    #             if vis.any():
    #                 vis_parts.append(np.unique(
    #                     start_rows[i_c][vis].astype(np.int64) * n_gly + g_c[vis]
    #                 ))
    #             n_blocked += int(blocked.sum())
    #             dist_sum += float(dist[blocked].sum())

    #     mean_d = dist_sum / n_blocked if n_blocked else np.nan
    #     cand_keys = np.unique(np.concatenate(cand_parts)) if cand_parts else empty
    #     vis_keys = np.unique(np.concatenate(vis_parts)) if vis_parts else empty
    #     return cand_keys, vis_keys, n_blocked, mean_d

    # def _box_visibility(self, start_idx=None, **kwargs):
    #     """
    #     Reference implementation of the original GLYCO box test.

    #     This intentionally uses explicit Python loops so that it follows
    #     glyco_mod.py/find_blocking_atom() directly. It is for validation,
    #     not performance.
    #     """
    #     if start_idx is None:
    #         start_idx = self.surface_start_indices()

    #     empty = np.empty(0, dtype=np.int64)
    #     n_gly = len(self.gly_atoms)

    #     if len(start_idx) == 0 or n_gly == 0:
    #         return empty, empty, 0, np.nan

    #     # Coordinates are rounded because the original GLYCO reads PDB
    #     # coordinates as decimal values and truncates sampled coordinates.
    #     P = np.round(
    #         np.asarray(self.start_atoms.positions[start_idx], dtype=float),
    #         3,
    #     )

    #     B = np.round(
    #         np.asarray(self.blocker_atoms.positions, dtype=float),
    #         3,
    #     )

    #     G = np.round(
    #         np.asarray(self.gly_atoms.positions, dtype=float),
    #         3,
    #     )

    #     start_rows = np.asarray(
    #         self._start_res_row[start_idx],
    #         dtype=np.int64,
    #     )

    #     start_residue_keys = self._start_residue_keys[start_idx]
    #     blocker_residue_keys = self._blocker_residue_keys

    #     cutoff = float(self._cutoff)

    #     candidate_keys = set()
    #     visible_keys = set()

    #     n_blocked = 0
    #     blocked_distance_sum = 0.0

    #     for local_start, start_position in enumerate(P):
    #         residue_row = int(start_rows[local_start])
    #         # start_resindex = int(start_resindices[local_start])
    #         start_residue_key = start_residue_keys[local_start]

    #         for gly_index, gly_position in enumerate(G):
    #             difference = gly_position - start_position
    #             distance = float(np.linalg.norm(difference))

    #             # Original GLYCO uses dist < cutoff.
    #             if distance <= 0.0 or distance >= cutoff:
    #                 continue

    #             candidate_keys.add(
    #                 residue_row * n_gly + gly_index
    #             )

    #             length_fraction = 1.0 / distance
    #             blocked = False

    #             # This reproduces:
    #             #
    #             # for t in numpy.arange(0.0, 1.0, 1 / dist)
    #             #
    #             for t in np.arange(
    #                 0.0,
    #                 1.0,
    #                 length_fraction,
    #             ):
    #                 point = start_position + t * difference

    #                 # Original GLYCO truncates toward zero.
    #                 x, y, z = np.trunc(point).astype(int)

    #                 for blocker_index, blocker_position in enumerate(B):
    #                     # blocker_resindex = int(
    #                     #     blocker_resindices[blocker_index]
    #                     # )
    #                     blocker_residue_key = blocker_residue_keys[
    #                         blocker_index
    #                     ]

    #                     if tuple(blocker_residue_key) == tuple(start_residue_key):
    #                         continue
    #                     # Original GLYCO ignores atoms in the starting
    #                     # residue as blockers.
    #                     # if blocker_resindex == start_resindex:
    #                         # continue

    #                     if (
    #                         abs(x - blocker_position[0]) <= 1.0
    #                         and abs(y - blocker_position[1]) <= 1.0
    #                         and abs(z - blocker_position[2]) <= 1.0
    #                     ):
    #                         blocked = True
    #                         break

    #                 if blocked:
    #                     break

    #             if blocked:
    #                 n_blocked += 1
    #                 blocked_distance_sum += distance
    #             else:
    #                 visible_keys.add(
    #                     residue_row * n_gly + gly_index
    #                 )

    #     candidate_keys = np.asarray(
    #         sorted(candidate_keys),
    #         dtype=np.int64,
    #     )

    #     visible_keys = np.asarray(
    #         sorted(visible_keys),
    #         dtype=np.int64,
    #     )

    #     mean_distance = (
    #         blocked_distance_sum / n_blocked
    #         if n_blocked
    #         else np.nan
    #     )

    #     return (
    #         candidate_keys,
    #         visible_keys,
    #         n_blocked,
    #         mean_distance,
    #     )

    def _box_visibility(
        self,
        start_idx=None,
        workers=None,
        starts_per_task=32,
    ):
        """
        Parallel implementation of the original GLYCO box test.

        --cr is intentionally ignored.
        """
        if start_idx is None:
            start_idx = self.surface_start_indices()

        empty = np.empty(0, dtype=np.int64)
        n_gly = len(self.gly_atoms)

        if len(start_idx) == 0 or n_gly == 0:
            return empty, empty, 0, np.nan

        start_idx = np.asarray(
            start_idx,
            dtype=np.int64,
        )

        start_positions = _legacy_coords(self.start_atoms.positions[start_idx])

        start_rows = np.asarray(
            self._start_res_row[start_idx],
            dtype=np.int64,
        )

        start_residue_keys = [
            self._start_residue_keys[i]
            for i in start_idx
        ]

        gly_positions = _legacy_coords(self.gly_atoms.positions)
        blocker_positions = _legacy_coords(self.blocker_atoms.positions)

        # Rebuild the integer-x index from THESE positions (current frame).
        # A bucket index cached in _prepare() would be stale on every frame
        # after the first, and in _frame_visibility().
        blocker_x_buckets = _build_x_buckets(blocker_positions)

        blocker_residue_keys = self._blocker_residue_keys

        # The worker uses local start-array indices, so make chunks over
        # range(len(start_positions)).
        n_starts = len(start_positions)

        ranges = [
            (lo, min(lo + starts_per_task, n_starts))
            for lo in range(0, n_starts, starts_per_task)
        ]

        if workers is None:
            workers = max(
                1,
                min(
                    mp.cpu_count(),
                    len(ranges),
                ),
            )

        context = mp.get_context("spawn")

        results = []

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_box_worker,
            initargs=(
                start_positions,
                start_rows,
                start_residue_keys,
                gly_positions,
                blocker_positions,
                blocker_residue_keys,
                blocker_x_buckets,
                self._cutoff,
                n_gly,
            ),
        ) as executor:
            for result in executor.map(_box_chunk, ranges):
                results.append(result.copy())

        candidate_parts = [
            result[0]
            for result in results
            if len(result[0])
        ]

        visible_parts = [
            result[1]
            for result in results
            if len(result[1])
        ]

        candidate_keys = (
            np.unique(
                np.concatenate(candidate_parts)
            )
            if candidate_parts
            else empty
        )

        visible_keys = (
            np.unique(
                np.concatenate(visible_parts)
            )
            if visible_parts
            else empty
        )

        n_blocked = sum(
            int(result[2])
            for result in results
        )

        distance_sum = sum(
            float(result[3])
            for result in results
        )

        mean_distance = (
            distance_sum / n_blocked
            if n_blocked
            else np.nan
        )

        return (
            candidate_keys,
            visible_keys,
            n_blocked,
            mean_distance,
        )

    def _counts_from_visible_keys(self, vis_keys):
        """Per-CA counts from visible (residue_row * n_gly + glycan) keys."""
        counts = np.zeros(len(self.prot_atoms), dtype=np.intp)
        if len(vis_keys):
            n_gly = len(self.gly_atoms)
            per_res = np.bincount(vis_keys // n_gly, minlength=self._n_res_rows)
            counts = per_res[self._prot_atom_res_row].astype(np.intp)
        return counts

    def count_glycan_neighbors_box(self, start_idx=None, pair_chunk=5000, ca_chunk=2000):
        """
        Reproduce the ORIGINAL GLYCO v1 blocking test exactly (--compat).

        For every (start atom, glycan atom) pair with distance < cutoff, the
        line is sampled at t = 0, 1/d, 2/d, ... (about one point per Angstrom,
        starting ON the start atom and stopping up to 1 A short of the glycan
        atom). Each sample point is truncated toward zero to integers
        (x, y, z), and the line is blocked if any protein heavy atom outside
        the start atom's residue has |x-ex| <= 1, |y-ey| <= 1 and
        |z-ez| <= 1 (an axis-aligned box on the truncated point, so it is
        biased by up to 1 A and NOT translation invariant). --cr is ignored.

        Coordinates are rounded to 0.001 A so int-truncation matches the text
        parsing of a PDB file. Counting: unique glycan atoms visible from at
        least one start atom of the residue.

        Returns (counts per CA, n blocked pairs, mean blocked distance).
        """
        _, vis_keys, n_blocked, mean_d = self._box_visibility(
            start_idx, pair_chunk, ca_chunk)
        return self._counts_from_visible_keys(vis_keys), n_blocked, mean_d

    # -- output --------------------------------------------------------------

    @staticmethod
    def _rows_for(residue_resindices, atom_resindices):
        """
        Row (position in `residue_resindices`) of each atom's residue. Does
        not assume `residue_resindices` is sorted.
        """
        residue_resindices = np.asarray(residue_resindices)
        order = np.argsort(residue_resindices)
        pos = np.searchsorted(residue_resindices, atom_resindices, sorter=order)
        return order[pos]

    @staticmethod
    def _atom_attr(atom_group, attribute, default=""):
        """Whole-group attribute as a string array (blank if unavailable)."""
        try:
            return np.asarray(getattr(atom_group, attribute)).astype(str)
        except (AttributeError, mda.exceptions.NoDataError):
            return np.full(len(atom_group), default, dtype=object)

    @staticmethod
    def _group_by_residue(keys, n_gly):
        """
        Split sorted unique keys (residue_row * n_gly + glycan_index) into
        {residue_row: array_of_glycan_indices}.
        """
        if len(keys) == 0:
            return {}
        rows = keys // n_gly
        gly = keys % n_gly
        uniq, first = np.unique(rows, return_index=True)
        return dict(zip(uniq.tolist(), np.split(gly, first[1:])))

    def _analyzed_frames(self):
        """Trajectory frame index of each analyzed frame (row of all_counts)."""
        frames = getattr(self, "frames", None)
        if frames is not None and len(frames) == self.n_frames:
            return np.asarray(frames, dtype=int)
        return np.arange(self.n_frames)

    def _resolve_frame_index(self, frame_index=None):
        """Normalize an index into the analyzed frames (None -> last)."""
        n = np.asarray(self.all_counts).shape[0]
        if frame_index is None:
            return n - 1
        k = frame_index + n if frame_index < 0 else frame_index
        if not 0 <= k < n:
            raise IndexError(
                f"frame index {frame_index} out of range for {n} analyzed frame(s)"
            )
        return k

    def _frame_blockers(self, k):
        """
        Return (start_idx, blockers, distances) for analyzed frame k
        (cylinder test). Uses the copy cached during the run when available,
        otherwise recomputes it with the Universe on the right trajectory
        frame.
        """
        cache = getattr(self, "_blocker_cache", None)
        if cache is not None and cache[0] == k:
            return cache[1:]

        trajectory_frame = int(self._analyzed_frames()[k])
        old_frame = int(self.u.trajectory.frame)
        try:
            self.u.trajectory[trajectory_frame]
            start_idx = self.surface_start_indices(k)
            blockers, distances, _ = self.get_blockers(start_idx)
        finally:
            self.u.trajectory[old_frame]

        return start_idx, blockers, distances

    def _frame_visibility(self, k, chunk=20000):
        """
        Return (start_idx, cand_keys, vis_keys) for analyzed frame k, where the
        keys are sorted unique residue_row * n_gly + glycan_index values:
        cand_keys = glycan atoms within the cutoff of a start atom of the
        residue, vis_keys = the subset visible from at least one of them
        (the residue's final union). Uses the box test with --compat and
        the cylinder test otherwise.
        """
        n_gly = len(self.gly_atoms)
        empty = np.empty(0, dtype=np.int64)
        trajectory_frame = int(self._analyzed_frames()[k])

        if self._box:
            cache = self._box_cache
            if cache is not None and cache[0] == k:
                return cache[1:]
            old_frame = int(self.u.trajectory.frame)
            try:
                self.u.trajectory[trajectory_frame]
                start_idx = self.surface_start_indices(k)
                cand_keys, vis_keys, _, _ = self._box_visibility(start_idx)
            finally:
                self.u.trajectory[old_frame]
            return start_idx, cand_keys, vis_keys

        # cylinder test: blocked pairs (cached from the run when possible)
        start_idx, blockers, _ = self._frame_blockers(k)

        # positions are read once, with the Universe on the right frame
        old_frame = int(self.u.trajectory.frame)
        try:
            self.u.trajectory[trajectory_frame]
            P = np.asarray(self.start_atoms.positions[start_idx], dtype=np.float64)
            G = np.asarray(self.gly_atoms.positions, dtype=np.float64)
        finally:
            self.u.trajectory[old_frame]

        cand_parts, vis_parts = [], []
        if len(start_idx) and n_gly:
            gtree = cKDTree(G)
            if len(blockers):
                blocked_keys = np.unique(
                    blockers[:, 0].astype(np.int64) * n_gly
                    + blockers[:, 2].astype(np.int64)
                )
            else:
                blocked_keys = empty

            for s in range(0, len(start_idx), chunk):
                idx = start_idx[s:s + chunk]
                neighbors = gtree.query_ball_point(P[s:s + chunk], self._cutoff)
                sizes = np.fromiter(
                    (len(x) for x in neighbors),
                    dtype=np.intp, count=len(neighbors),
                )
                if sizes.sum() == 0:
                    continue

                start_ids = np.repeat(idx.astype(np.int64), sizes)
                gly_ids = np.concatenate(neighbors).astype(np.int64, copy=False)
                res_rows = self._start_res_row[start_ids].astype(np.int64)

                cand_parts.append(np.unique(res_rows * n_gly + gly_ids))
                keep = ~np.isin(start_ids * n_gly + gly_ids, blocked_keys)
                vis_parts.append(np.unique(res_rows[keep] * n_gly + gly_ids[keep]))

        cand_keys = np.unique(np.concatenate(cand_parts)) if cand_parts else empty
        vis_keys = np.unique(np.concatenate(vis_parts)) if vis_parts else empty
        return start_idx, cand_keys, vis_keys

    def _ensure_tempfactors(self):
        try:
            self.u.atoms.tempfactors  # noqa: B018
        except (AttributeError, mda.exceptions.NoDataError):
            self.u.add_TopologyAttr('tempfactors')

    def write_blockers_csv(self, frame_index=None, output=None, chunk=100_000):
        """
        Write every blocked (start atom, blocking atom, glycan atom) triple
        for one analyzed frame. No coordinates are written.

        Atom indices are PDB serial numbers (1-based). pair_distance is the
        start-atom to glycan-atom distance. The file can be very large: one
        row per blocking atom per blocked start/glycan pair. Not available
        with --compat, whose box test does not identify blocking atoms.

        Parameters
        ----------
        frame_index : int, optional
            Index into the analyzed frames. None uses the last analyzed frame.
        output : str or pathlib.Path, optional
            Output filename. Defaults to blockers.csv in self.outdir.
        chunk : int
            Rows converted and written at a time (bounds memory use).
        """
        if self._box:
            raise ValueError(
                "blockers.csv is not available with --compat: the box "
                "test does not identify blocking atoms"
            )

        if output is None:
            output = os.path.join(self.outdir, "blockers.csv")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        k = self._resolve_frame_index(frame_index)
        trajectory_frame = int(self._analyzed_frames()[k])

        _, blockers, distances = self._frame_blockers(k)
        s_i, b_i, g_i = blockers.T          # unpack the three columns once
        n = len(blockers)
        distances = np.asarray(distances, dtype=np.float64)

        start, blocker, glycan = self.start_atoms, self.blocker_atoms, self.gly_atoms
        attr = self._atom_attr

        # Per-group arrays are built ONCE; rows are then indexed per chunk
        start_id, blocker_id, gly_id = (pdb_atom_ids(start),
                                        pdb_atom_ids(blocker),
                                        pdb_atom_ids(glycan))
        start_name, start_resname = attr(start, "names"), attr(start, "resnames")
        start_resid, start_chain = attr(start, "resids"), attr(start, "chainIDs")
        blk_name, blk_resname = attr(blocker, "names"), attr(blocker, "resnames")
        blk_resid, blk_chain = attr(blocker, "resids"), attr(blocker, "chainIDs")
        gly_name, gly_resname = attr(glycan, "names"), attr(glycan, "resnames")
        gly_resid, gly_chain = attr(glycan, "resids"), attr(glycan, "chainIDs")

        fieldnames = [
            "analyzed_frame", "trajectory_frame",
            "start_atom_index", "start_atom_name", "start_atom_resname",
            "start_atom_resid", "start_atom_chain",
            "blocker_atom_index", "blocker_atom_name", "blocker_atom_resname",
            "blocker_atom_resid", "blocker_atom_chain",
            "glycan_atom_index", "glycan_atom_name", "glycan_resname",
            "glycan_resid", "glycan_chain",
            "pair_distance",
        ]

        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)

            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                s, b, g = s_i[lo:hi], b_i[lo:hi], g_i[lo:hi]
                m = hi - lo
                columns = (
                    np.full(m, k, dtype=np.int64),
                    np.full(m, trajectory_frame, dtype=np.int64),
                    start_id[s], start_name[s], start_resname[s],
                    start_resid[s], start_chain[s],
                    blocker_id[b], blk_name[b], blk_resname[b],
                    blk_resid[b], blk_chain[b],
                    gly_id[g], gly_name[g], gly_resname[g],
                    gly_resid[g], gly_chain[g],
                    np.char.mod("%.6f", distances[lo:hi]),
                )
                writer.writerows(zip(*(c.tolist() for c in columns)))

        print(f"Blocker results written to: {output} ({n} rows)")
        return output

    def write_bfactors(self, frame_index=None):
        """
        Write b_factors.pdb: the WHOLE structure (every atom in the
        Universe, hydrogens and non-protein atoms included) for one analyzed
        frame (last by default), plus b_factors.npy with the counts for all
        analyzed frames.

        B-factor of each atom = the value of the CA atom in the same
        residue. Atoms in residues without a CA in prot_atoms (glycans,
        water, ions, ...) get 0.
        """
        os.makedirs(self.outdir, exist_ok=True)
        self._ensure_tempfactors()

        # put the Universe on the frame whose counts are written, so the
        # PDB coordinates and B-factors describe the same frame
        k = self._resolve_frame_index(frame_index)
        self.u.trajectory[int(self._analyzed_frames()[k])]

        self._populate_b_factors(k)
        self.u.atoms.write(os.path.join(self.outdir, 'b_factors.pdb'))
        np.save(os.path.join(self.outdir, 'b_factors'), np.asarray(self.all_counts))

    def _populate_b_factors(self, frame_index=None):
        """
        Set tempfactors on the whole Universe from self.all_counts: every
        atom of a residue receives the value of that residue's CA atom, and
        all other atoms are set to 0.

        Parameters
        ----------
        frame_index : int or None
            Which row of self.all_counts to use if it is 2D (shape
            (n_frames, n_ca)); negative counts from the end. Ignored if it
            is 1D. Defaults to the last frame.

        Notes
        -----
        Column i of self.all_counts corresponds to the i-th CA atom of
        self.prot_atoms (one CA per residue is enforced in _prepare).
        """
        data = np.asarray(self.all_counts)

        if data.ndim == 1:
            ca_values = data
        elif data.ndim == 2:
            ca_values = data[self._resolve_frame_index(frame_index)]
        else:
            raise ValueError(
                f"self.all_counts must be 1D or 2D, got shape {data.shape}"
            )

        ca_resindices = self.prot_atoms.resindices
        if len(ca_values) != len(ca_resindices):
            raise ValueError(
                f"self.all_counts has {len(ca_values)} values "
                f"but self.prot_atoms has {len(ca_resindices)} CA atoms"
            )

        if ca_values.size and ca_values.max() > 999.99:
            warnings.warn(
                f"max B-factor value {ca_values.max():g} exceeds the PDB "
                "B-factor column width (999.99); the PDB columns will be "
                "misaligned. Use b_factors.npy for the raw values.",
                stacklevel=2,
            )

        # every atom in a CA residue -> that residue's CA value
        res_atoms = self.prot_atoms.residues.atoms
        rows = self._rows_for(ca_resindices, res_atoms.resindices)

        self.u.atoms.tempfactors = np.zeros(self.u.atoms.n_atoms)
        res_atoms.tempfactors = ca_values[rows]

    def write_frame_summary(self):
        """Write frame_summary.csv: one row per analyzed frame."""
        r = self.results
        table = np.column_stack([
            self._analyzed_frames(), r.n_surface_residues, r.n_blocked_pairs,
            r.mean_blocked_distance,
        ])
        np.savetxt(
            os.path.join(self.outdir, 'frame_summary.csv'), table,
            delimiter=",", comments="", fmt=["%d", "%d", "%d", "%.6g"],
            header="frame,n_surface_residues,n_blocked_pairs,mean_blocked_distance",
        )

    def write_residue_csv(self, frame_index=None, output=None):
        """
        Write per-residue glycan density and SASA values.

        Output columns:
            Chain,ResNum,ResName,Glycan_density,SASA

        Parameters
        ----------
        frame_index : int, optional
            Index into the analyzed frames. None uses the last analyzed frame.
        output : str or pathlib.Path, optional
            Output filename. Defaults to residue_summary.csv in self.outdir.
        """
        if output is None:
            output = os.path.join(self.outdir, "residue_summary.csv")

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Normalize the requested analyzed-frame index.
        k = self._resolve_frame_index(frame_index)

        # Glycan density/counts. Each column corresponds to one CA residue.
        glycan_density = np.asarray(self.all_counts)[k]

        if len(glycan_density) != len(self.prot_atoms):
            raise ValueError(
                "The number of glycan-density values does not match "
                "the number of protein CA atoms."
            )

        # Map each CA residue to the corresponding SASA row.
        sasa_rows = np.searchsorted(
            self._residue_resindices,
            self.prot_atoms.resindices,
        )

        sasa_values = np.asarray(self.residue_areas)[k, sasa_rows]

        # Obtain residue-level identifiers from the CA atoms.
        resnums = np.asarray(self.prot_atoms.resids)
        resnames = np.asarray(self.prot_atoms.resnames)

        # chainIDs are normally available for PDB input. If they are absent,
        # write blank chain identifiers rather than failing.
        try:
            chains = np.asarray(self.prot_atoms.chainIDs)
        except (AttributeError, mda.exceptions.NoDataError):
            chains = np.full(len(self.prot_atoms), "", dtype=object)

        if not (
            len(resnums)
            == len(resnames)
            == len(chains)
            == len(glycan_density)
            == len(sasa_values)
        ):
            raise ValueError("Residue attributes have inconsistent lengths.")

        # Compatibility outputs should match the legacy programs, which omit
        # residues whose glycan count is zero. Modern/default output retains
        # all selected protein residues.
        nonzero_only = (
            getattr(self, "compat", False)
            or getattr(self, "compat_v2", False)
        )

        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)

            writer.writerow(
                [
                    "Chain",
                    "ResNum",
                    "ResName",
                    "Glycan_density",
                    "SASA",
                ]
            )

            n_written = 0

            for chain, resnum, resname, density, sasa in zip(
                chains,
                resnums,
                resnames,
                glycan_density,
                sasa_values,
            ):
                density = int(density)

                if nonzero_only and density == 0:
                    continue

                writer.writerow(
                    [
                        str(chain),
                        int(resnum),
                        str(resname),
                        density,
                        f"{float(sasa):.6g}",
                    ]
                )

                n_written += 1

        print(
            f"Residue summary written to: {output} "
            f"({n_written} residues)"
        )

        return output

    def _write_progress(self, current, total):
        """Write machine-readable progress for the Streamlit app."""
        if not self.progress_file:
            return

        temp_file = self.progress_file + ".tmp"

        with open(temp_file, "w") as f:
            json.dump({
                "current": current,
                "total": total,
                "percent": int(100 * current / total) if total else 0,
            }, f)

        os.replace(temp_file, self.progress_file)

    @staticmethod
    def _tcl_quote(value):
        """
        Quote a string as one Tcl word. Braces are used when safe (inside
        braces a backslash is literal, so Windows-style paths survive);
        otherwise fall back to double quotes with Tcl metacharacters escaped.
        """
        value = str(value)
        if "{" not in value and "}" not in value and not value.endswith("\\"):
            return "{" + value + "}"
        return '"' + re.sub(r'([\\"$\[\]])', r'\\\1', value) + '"'

    @staticmethod
    def _index_ranges(sorted_indices):
        """Compress sorted ints into VMD 'a to b' ranges."""
        if len(sorted_indices) == 0:
            return ""
        breaks = np.flatnonzero(np.diff(sorted_indices) != 1) + 1
        parts = []
        for run in np.split(sorted_indices, breaks):
            parts.append(str(run[0]) if len(run) == 1 else f"{run[0]} to {run[-1]}")
        return " ".join(parts)

    def write_vmd_user_script(self, output=None, frame_offset=0):
        """
        Generate a VMD Tcl script that loads the topology and trajectory
        from self.u, then assigns per-residue data to VMD's 'user'
        attribute for every ANALYZED frame.

        Creates one atom selection and moves it between frames, with one
        bulk 'user' assignment per analyzed frame. Defaults to writing into
        self.outdir.

        Note: VMD's 'index' selection is 0-based, so MDAnalysis indices are
        used here unchanged.
        """
        u = self.u

        topology = getattr(u, "filename", None)
        if not topology:
            raise ValueError("Could not determine topology filename from Universe")
        topology = Path(topology).resolve()
        if not topology.is_file():
            raise FileNotFoundError(topology)

        reader = u.trajectory
        trajectory_files = getattr(reader, "filenames", None)
        if trajectory_files is None:
            filename = getattr(reader, "filename", None)
            trajectory_files = [filename] if filename else []
        trajectory_files = [Path(p).resolve() for p in trajectory_files
                            if p is not None]
        if not trajectory_files:
            raise ValueError("Could not determine trajectory filename(s)")
        for path in trajectory_files:
            if not path.is_file():
                raise FileNotFoundError(path)

        # data: (n_residues, n_analyzed_frames), one row per CA residue
        data = np.asarray(self.all_counts).T
        residues = self.prot_atoms.residues
        if data.shape[0] != len(residues):
            raise ValueError(
                f"Data has {data.shape[0]} rows, but prot_atoms contains "
                f"{len(residues)} residues"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError("data contains NaN or infinite values")

        n_frames = data.shape[1]
        analyzed = self._analyzed_frames()
        n_traj_frames = int(reader.n_frames)

        # every atom of every CA residue, in ascending atom-index order
        # (the order VMD uses when assigning a list to a selection)
        atoms = residues.atoms
        atom_indices = np.asarray(atoms.indices, dtype=np.int64)
        atom_rows = self._rows_for(residues.resindices, atoms.resindices)
        order = np.argsort(atom_indices, kind="stable")
        atom_indices, atom_rows = atom_indices[order], atom_rows[order]

        data_max = float(np.max(data)) if data.size else 0.0
        if data_max <= 0:
            data_max = 1.0

        lines = [
            "# Auto-generated VMD Tcl script",
            "# Bulk per-frame assignment of residue values",
            "",
            f"set molid [mol new {self._tcl_quote(topology.as_posix())} waitfor all]",
            "",
        ]
        for traj in trajectory_files:
            lines.append(
                f"mol addfile {self._tcl_quote(traj.as_posix())} waitfor all molid $molid"
            )
        lines += [
            "",
            "mol modselect 0 $molid protein",
            "mol modstyle 0 $molid NewCartoon",
            "mol modcolor 0 $molid User",
            "",
            f"set n_traj_frames {n_traj_frames}",
            "set n_vmd_frames [molinfo $molid get numframes]",
            "",
            "if {$n_vmd_frames == $n_traj_frames + 1} {",
            "    animate delete beg 0 end 0 $molid",
            '    puts "Deleted duplicate first frame."',
            "} elseif {$n_vmd_frames == $n_traj_frames} {",
            '    puts "Frame counts match; no frame deleted."',
            "} else {",
            '    error "Frame mismatch: VMD has $n_vmd_frames frames, but the '/
            'trajectory has $n_traj_frames. Expected equal counts or exactly '/
            'one extra VMD frame."',
            "}",
            "",
            "# Set User color scale",
            f"mol scaleminmax $molid 0 0 {data_max:.10g}",
            "",
            f"set sel [atomselect $molid {{index {self._index_ranges(atom_indices)}}}]",
            "",
        ]

        for k in range(n_frames):
            vmd_frame = frame_offset + int(analyzed[k])
            values = " ".join(f"{v:.10g}" for v in data[atom_rows, k])
            lines += [
                f"# Data frame {k} -> VMD frame {vmd_frame}",
                f"$sel frame {vmd_frame}",
                f"$sel set user {{{values}}}",
                "",
            ]

        lines += [
            "$sel delete",
            'puts "Finished assigning per-residue user values."',
            "",
        ]

        if output is None:
            output = os.path.join(self.outdir, VMD_SCRIPT_NAME)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines))

        print(f"VMD script written to: {output}")
        return output

    def write_excluded_glycan_atoms_csv(self, frame_index=None, output=None):
        """
        Write one row per glycan atom absent from the final residue-level
        visibility union, for every SURFACE residue with a nonzero final
        count (zero-count residues are omitted).

        A glycan atom is in the union if it lies within the distance cutoff
        of at least one of the residue's start atoms (all heavy atoms in the
        legacy modes, the CA otherwise) and its line of sight from that
        start atom is not blocked. The blocking test matches the one used for
        the counts: cylinder by default and with --compat-v2, the original
        box test with --compat.

        reason:
            outside_distance_cutoff     - not within the cutoff of any start atom
            blocked_for_all_start_atoms - within the cutoff of some start atom,
                                          but blocked for every such start atom

        glycan_atom_index is the PDB serial number (1-based).
        """
        if output is None:
            output = os.path.join(self.outdir, "excluded_glycan_atoms.csv")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        k = self._resolve_frame_index(frame_index)
        trajectory_frame = int(self._analyzed_frames()[k])
        n_gly = len(self.gly_atoms)
        empty = np.empty(0, dtype=np.int64)

        start_idx, cand_keys, vis_keys = self._frame_visibility(k)
        candidates = self._group_by_residue(cand_keys, n_gly)
        visible = self._group_by_residue(vis_keys, n_gly)

        # per-atom / per-residue metadata, fetched once
        gly_ids_pdb = pdb_atom_ids(self.gly_atoms)
        gly_names = self._atom_attr(self.gly_atoms, "names")
        gly_resnames = self._atom_attr(self.gly_atoms, "resnames")
        gly_chains = self._atom_attr(self.gly_atoms, "chainIDs")
        gly_resids = self._atom_attr(self.gly_atoms, "resids")

        ca_resnames = self._atom_attr(self.prot_atoms, "resnames")
        ca_chains = self._atom_attr(self.prot_atoms, "chainIDs")
        ca_resids = self._atom_attr(self.prot_atoms, "resids")
        ca_of_row = {int(r): i for i, r in enumerate(self._prot_atom_res_row)}

        # Surface residues only: buried residues have no start atoms
        surface_rows = (
            np.unique(self._start_res_row[start_idx]) if len(start_idx) else empty
        )

        fieldnames = [
            "analyzed_frame", "trajectory_frame",
            "protein_resname", "protein_chain", "protein_resid",
            "final_glycan_count",
            "glycan_atom_index", "glycan_atom_name", "glycan_resname",
            "glycan_chain", "glycan_resid",
            "reason",
        ]

        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)

            for row in surface_rows.tolist():
                ca = ca_of_row.get(row)
                if ca is None:
                    continue  # residue without a CA: no count to report

                vis = visible.get(row, empty)
                cand = candidates.get(row, empty)

                is_visible = np.zeros(n_gly, dtype=bool)
                is_visible[vis] = True
                is_candidate = np.zeros(n_gly, dtype=bool)
                is_candidate[cand] = True

                final_count = int(self.all_counts[k, ca])
                if len(vis) != final_count:
                    warnings.warn(
                        f"Residue {ca_chains[ca]}:{ca_resnames[ca]}{ca_resids[ca]}"
                        f": union size {len(vis)} != reported count {final_count}",
                        stacklevel=2,
                    )

                # Zero-count residues are not written (matches GLYCO-2 and
                # the nonzero-only residue_summary.csv of the compat modes).
                if final_count == 0:
                    continue

                excluded = np.flatnonzero(~is_visible)
                if len(excluded) == 0:
                    continue

                reasons = np.where(
                    is_candidate[excluded],
                    "blocked_for_all_start_atoms",
                    "outside_distance_cutoff",
                )

                writer.writerows(zip(
                    repeat(k),
                    repeat(trajectory_frame),
                    repeat(ca_resnames[ca]),
                    repeat(ca_chains[ca]),
                    repeat(ca_resids[ca]),
                    repeat(final_count),
                    gly_ids_pdb[excluded].tolist(),
                    gly_names[excluded].tolist(),
                    gly_resnames[excluded].tolist(),
                    gly_chains[excluded].tolist(),
                    gly_resids[excluded].tolist(),
                    reasons.tolist(),
                ))

        print(f"Excluded glycan atoms written to: {output}")
        return output

    def _atom_residue_keys(self, atom_group):
        """
        Return GLYCO-style residue identities:
            (resname, chain, resid)
        """
        try:
            chains = np.asarray(
                atom_group.chainIDs,
                dtype=str,
            )
        except (AttributeError, mda.exceptions.NoDataError):
            chains = np.full(
                len(atom_group),
                "",
                dtype=str,
            )

        return np.asarray(
            [
                (
                    str(resname).strip(),
                    str(chain).strip(),
                    str(resid).strip(),
                )
                for resname, chain, resid in zip(
                    atom_group.resnames,
                    chains,
                    atom_group.resids,
                )
            ],
            dtype=object,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_job(args, parser, top, traj, outdir, verbose):
    """Load one structure/trajectory, run the analysis, write outputs."""
    u = load_universe(top, traj)
    analysis = GlycoAnalysis(
        u, parser=parser, args=args,
        top=top, traj=traj, indir=None, outdir=outdir,
    )
    analysis.run(
        verbose=verbose,
        progressbar_kwargs={'desc': 'Running GlyCo Analysis'},
    )


def _structure_outdirs(structures, base_outdir):
    """One output subfolder per structure, disambiguating clashing stems."""
    stems = [Path(p).stem for p in structures]
    dirs = []
    for path, stem in zip(structures, stems):
        name = stem if stems.count(stem) == 1 else Path(path).name.replace('.', '_')
        dirs.append(os.path.join(base_outdir, name))
    return dirs


def main(argv=None):
    parser = base_parser()
    args = parser.parse_args(argv)

    # Validate everything that does not need a Universe BEFORE loading any
    # files, so bad paths/options give a clean usage error, not a traceback.
    validate_arguments(args, parser.error)

    is_streamlit = os.environ.get("GLYCO_STREAMLIT") == "1"
    verbose = False if is_streamlit else args.verbose

    os.makedirs(args.outdir, exist_ok=True)
    configure_warning_log(os.path.join(args.outdir, "mdanalysis_warnings.log"))

    if args.indir is None:
        # single topology (+ optional trajectories): errors exit via parser
        if not confirm_overwrite(args.outdir, args.force, args.write_vmd,
                                 args.write_excluded, args.write_blockers):
            return 1
        try:
            u = load_universe(args.top, args.traj)
        except Exception as e:  # noqa: BLE001
            parser.error(
                "could not load -a/--top with -b/--traj (do all trajectory "
                f"files match the topology's atom count?): {e}"
            )
        GlycoAnalysis(u, parser=parser, args=args).run(
            verbose=verbose,
            progressbar_kwargs={'desc': 'Running GlyCo Analysis'},
        )
        return 0

    # --indir: each structure is analyzed independently into its own
    # subfolder; one bad structure does not abort the rest.
    outdirs = _structure_outdirs(args.structures, args.outdir)
    failures = []
    for n, (path, outdir) in enumerate(zip(args.structures, outdirs), start=1):
        print(f"[{n}/{len(args.structures)}] {path}")
        if not confirm_overwrite(outdir, args.force, args.write_vmd,
                                 args.write_excluded, args.write_blockers):
            failures.append((path, "output exists and overwrite declined"))
            continue
        try:
            # parser=None -> configuration errors raise ValueError here
            # instead of exiting the whole batch
            _run_job(args, None, path, None, outdir, verbose)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {path}: {e}", file=sys.stderr)
            failures.append((path, str(e)))

    if failures:
        print(f"{len(failures)} of {len(args.structures)} structure(s) failed:",
              file=sys.stderr)
        for path, reason in failures:
            print(f"  {path}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())