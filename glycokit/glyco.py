#!/usr/bin/env python3
# ruff: noqa: EXE001
"""
glyco: glycan accessibility of protein surface residues (glycokit / GLYCO v3).

Counting modes
    default      CA starts, cylinder test (--cr), own residue excluded.
    --compat     reproduces GLYCO v1 (glyco_mod.py).
    --compat-v2  reproduces GLYCO-2 (glycomod.py / core_glyco_mod.py).
"""
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

# Per-process state of the GLYCO v1 box workers
_BOX_STATE = {}

# Safety margin of the vectorized distance prefilter in _box_chunk. The
# exact GLYCO v1 test (numpy.linalg.norm of one pair, '<' cutoff) is applied
# afterwards, so this only has to exceed float64 rounding differences.
_BOX_PREFILTER_EPS = 1e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resid_labels(ag):
    """
    PDB-style residue numbers including the insertion code ('52', '52A'),
    i.e. what glyco_mod.py / core_glyco_mod.py read from line[22:28].strip().
    """
    resids = np.asarray(ag.resids)
    try:
        icodes = [str(i).strip() for i in ag.icodes]
    except (AttributeError, mda.exceptions.NoDataError):
        icodes = [""] * len(resids)
    return np.asarray([f"{int(r)}{i}" for r, i in zip(resids, icodes)],
                      dtype=object)


def _v2_cylinder_blocked(P32, G32, B64, i, g, j, r):
    """Vectorized c_count_points: True where blocker j lies in the (i, g) cylinder."""
    vec = P32[i] - G32[g]                                   # float32 (pt2 - pt1)
    const = np.float32(r) * np.linalg.norm(vec, axis=1)     # float32
    q_pt1 = B64[j] - G32[g]                                 # float64
    q_pt2 = B64[j] - P32[i]                                 # float64
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
    into `positions`.
    """
    positions = np.asarray(positions, dtype=np.float64)
    buckets = {}
    if positions.size:
        for index, x in enumerate(positions[:, 0].tolist()):
            buckets.setdefault(int(x), []).append(index)
    return {key: tuple(value) for key, value in buckets.items()}


def _init_box_worker(start_positions, start_rows, start_residue_keys,
                     gly_positions, blocker_positions, blocker_residue_keys,
                     blocker_x_buckets, cutoff, n_gly):
    """Initialize one process with plain NumPy/Python data (no MDAnalysis objects)."""
    _BOX_STATE.clear()
    _BOX_STATE.update(
        start_positions=np.asarray(start_positions, dtype=np.float64),
        start_rows=np.asarray(start_rows, dtype=np.int64),
        start_residue_keys=[tuple(k) for k in start_residue_keys],
        gly_positions=np.asarray(gly_positions, dtype=np.float64),
        blocker_positions=np.asarray(blocker_positions, dtype=np.float64),
        blocker_residue_keys=[tuple(k) for k in blocker_residue_keys],
        blocker_x_buckets=blocker_x_buckets,
        cutoff=float(cutoff),
        n_gly=int(n_gly),
    )


def _box_chunk(start_range):
    """
    GLYCO v1 check_protein() for a range of (local) start atoms.

    Returns (candidate keys, visible keys, n blocked pairs, sum of blocked
    distances); keys are residue_row * n_gly + glycan_index.
    """
    s = _BOX_STATE
    P, rows, skeys = s["start_positions"], s["start_rows"], s["start_residue_keys"]
    G, B, bkeys = s["gly_positions"], s["blocker_positions"], s["blocker_residue_keys"]
    buckets, cutoff, n_gly = s["blocker_x_buckets"], s["cutoff"], s["n_gly"]

    lo, hi = start_range
    candidate_keys, visible_keys = set(), set()
    n_blocked = 0
    blocked_distance_sum = 0.0

    for si in range(lo, hi):
        p = P[si]
        row = int(rows[si])
        own = skeys[si]

        # vectorized prefilter; the exact v1 test follows per pair
        d = G - p
        approx = np.sqrt(np.einsum("ij,ij->i", d, d))
        for gi in np.flatnonzero(approx < cutoff + _BOX_PREFILTER_EPS).tolist():
            difference = G[gi] - p
            distance = float(np.linalg.norm(difference))

            # original GLYCO condition: distance < cutoff
            if distance >= cutoff:
                continue

            key = row * n_gly + gi
            candidate_keys.add(key)

            # check_protein(): dist == 0 -> visible
            if distance == 0.0:
                visible_keys.add(key)
                continue

            blocked = False
            # numpy.arange(0.0, 1.0, 1 / dist), int() truncation toward zero
            for t in np.arange(0.0, 1.0, 1.0 / distance):
                point = p + t * difference
                x, y, z = int(point[0]), int(point[1]), int(point[2])
                # pro_dict.get(x) + pro_dict.get(x-1) + pro_dict.get(x+1)
                for bi in chain(buckets.get(x, ()), buckets.get(x - 1, ()),
                                buckets.get(x + 1, ())):
                    bx, by, bz = B[bi]
                    if (abs(y - by) <= 1.0 and abs(z - bz) <= 1.0
                            and abs(x - bx) <= 1.0
                            and bkeys[bi] != own):     # own residue never blocks
                        blocked = True
                        break
                if blocked:
                    break

            if blocked:
                n_blocked += 1
                blocked_distance_sum += distance
            else:
                visible_keys.add(key)

    return (np.asarray(sorted(candidate_keys), dtype=np.int64),
            np.asarray(sorted(visible_keys), dtype=np.int64),
            n_blocked,
            blocked_distance_sum)


def atom_identity(atom):
    try:
        chain_id = str(atom.chainID).strip()
    except (AttributeError, mda.exceptions.NoDataError):
        chain_id = ""
    return (
        int(atom.id),
        str(atom.name).strip(),
        str(atom.resname).strip(),
        chain_id,
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
    """Route Python warnings (incl. MDAnalysis') into a log file."""
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
    optional digits).

    legacy=True reproduces the original GLYCO filter instead: drop every atom
    whose name contains an 'H' ANYWHERE (also NH1/NH2, OH, CH2, ...).
    """
    names = np.asarray(ag.names, dtype=str)
    if legacy:
        is_h = np.fromiter(("H" in n for n in names), dtype=bool, count=len(names))
        return ag[~is_h]
    is_h = np.fromiter((bool(_HYDROGEN_NAME.match(n)) for n in names),
                       dtype=bool, count=len(names))
    try:
        elements = np.char.upper(np.asarray(ag.elements, dtype=str))
        is_h |= np.isin(elements, ("H", "D"))
    except (AttributeError, mda.exceptions.NoDataError):
        pass
    return ag[~is_h]


def find_structures(indir, extensions):
    """Independent structure files in `indir` matching `extensions`."""
    files = []
    for ext in extensions:
        files.extend(sorted(glob.glob(os.path.join(indir, f'*.{ext}'))))
    return list(dict.fromkeys(files))


def validate_arguments(cfg, error, check_inputs=True):
    """
    Validate (and normalize) parameters on `cfg` (argparse Namespace or a
    GlycoAnalysis instance). Sets cfg.structures and normalizes
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
                error(f"--indir: no files with extension(s) {extensions} "
                      f"found in {cfg.indir!r}")
            cfg.structures = structures

    slice_given = any(v is not None
                      for v in (cfg.frame_start, cfg.frame_stop, cfg.frame_step))
    frames_given = bool(cfg.frame_list)

    if frames_given and slice_given:
        error("--frames is not allowed with --start/--stop/--step")
    if indir_given and (frames_given or slice_given):
        error("--frames/--start/--stop/--step are not meaningful with "
              "--indir (each structure is a single, independent frame)")

    for name in ('distance_cutoff', 'area_cutoff', 'radius', 'probe'):
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            error(f"--{name.replace('_', '-')} must be positive, got {value}")

    if getattr(cfg, 'box_workers', 1) is not None and getattr(cfg, 'box_workers', 1) < 1:
        error("--box-workers must be >= 1")

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
    return DEFAULT_OUTDIR_NAME


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
    """Prompt before overwriting existing output files unless force=True."""
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
        print(f"[error] {outdir!r} already contains: {listing}; no TTY is "
              f"available to prompt, pass --force to overwrite "
              f"non-interactively.", file=sys.stderr)
        return False

    while True:
        try:
            answer = input(f"{outdir!r} already contains: {listing}. "
                           f"Overwrite? [y/N] ").strip().lower()
        except EOFError:
            return False
        if answer in ('y', 'yes'):
            return True
        if answer in ('', 'n', 'no'):
            return False
        print("Please answer 'y' or 'n'.")


def load_universe(top, traj=None):
    """Load a Universe; several trajectories go through the chain reader."""
    if not traj:
        return mda.Universe(top)
    return mda.Universe(top, traj if len(traj) > 1 else traj[0])

# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

class GlycoAnalysis(AnalysisBase):
    """
    Run Glyco and other useful output functions.

    Counting modes:
      default      CA starts, cylinder test (--cr), own residue excluded.
      --compat     GLYCO v1: all-heavy-atom starts with a per-residue union,
                   integer cutoff (strict '<'), legacy hydrogen filter,
                   blockers from ATOM records only, FreeSASA on the atoms the
                   freesasa executable uses, and the original integer-truncated
                   +-1 A box blocking test.
      --compat-v2  GLYCO-2: same counting/filters/FreeSASA set, float32 pair
                   selection (norm < cutoff) and the float32 cylinder test of
                   c_count_points (start residue CAN block, radius inclusive).
    """

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

        # Single source of truth for defaults: base_parser() itself.
        defaults = vars(base_parser().parse_args([]))

        # Only touch sys.argv if a parser was supplied WITHOUT parsed args.
        source = args if args is not None else parser
        self.apply_params(defaults=defaults, args=source, **kwargs)

        # deprecated spelling: compat_box=True is now simply compat=True
        if getattr(self, "compat_box", False):
            self.compat = True

        self._validate_args()
        if not self.outdir:
            self.outdir = default_outdir(self)
        self._validate_atom_selections()

    # -- configuration ------------------------------------------------------

    def _error(self, message):
        """parser.error() from the CLI, ValueError for library use."""
        if self.parser is not None:
            self.parser.error(message)
        raise ValueError(message)

    def _diag(self, *parts):
        """Diagnostic output, only with -v/--verbose."""
        if self._verbose:
            print(*parts, flush=True)

    def _validate_args(self):
        # Library usage: a Universe was passed in directly and no input
        # files were named, so there is nothing about inputs to validate.
        library_mode = (self.parser is None and self.top is None
                        and self.indir is None and not self.traj)
        validate_arguments(self, self._error, check_inputs=not library_mode)

    def apply_params(self, args=None, defaults=None, **kwargs):
        """Merge defaults < args (Namespace or parser) < kwargs onto self."""
        values = dict(defaults or {})
        if args is not None:
            if isinstance(args, argparse.ArgumentParser):
                args = args.parse_args()
            values.update(vars(args))
        values.update(kwargs)
        for name, value in values.items():
            setattr(self, name, value)

    def run(self, start=None, stop=None, step=None, frames=None, **kwargs):
        """AnalysisBase.run, falling back to the CLI frame slice."""
        if all(v is None for v in (start, stop, step, frames)):
            start, stop, step = self.frame_start, self.frame_stop, self.frame_step
            frames = self.frame_list
        self._run_slice = {"start": start, "stop": stop, "step": step,
                           "frames": frames}
        return super().run(**self._run_slice, **kwargs)

    def _validate_atom_selections(self):
        """Validate protein, glycan and blocker selections against the Universe."""
        prot_sel = (self.prot_sel or "").strip()
        gly_sel = (self.gly_sel or "").strip()
        block_sel = (self.block_sel or "").strip()

        # --compat / --compat-v2: restrict start and blocker residues to the
        # residue names of the original scripts. Glycans are untouched.
        if getattr(self, "compat", False) or getattr(self, "compat_v2", False):
            names = _V2_RESNAMES if getattr(self, "compat_v2", False) else _V1_RESNAMES
            resname_filter = f"resname {names}"
            if prot_sel:
                prot_sel = f"({prot_sel}) and ({resname_filter})"
            if block_sel:
                block_sel = f"({block_sel}) and ({resname_filter})"
                # GLYCO v1 gen_pro_dict(): blockers from ATOM records only.
                # GLYCO-2 (get_atoms_data) has no record-type restriction, and
                # neither script restricts the START atoms by record type.
                if getattr(self, "compat", False):
                    candidate = f"({block_sel}) and record_type ATOM"
                    try:
                        self.u.select_atoms(candidate)
                        block_sel = candidate
                    except Exception:  # noqa: BLE001
                        warnings.warn(
                            "record_type is unavailable for this topology; "
                            "--compat blockers are NOT restricted to ATOM records",
                            stacklevel=2,
                        )

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
                results[name] = {"valid": False,
                                 "message": f"Invalid selection ('{selection}'): {e}"}
                continue
            if len(atoms) == 0:
                results[name] = {"valid": False,
                                 "message": f"0 atoms selected ('{selection}')"}
            else:
                results[name] = {"valid": True, "message": f"{len(atoms):,} atoms"}

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

    def _mode_name(self):
        return 'compat' if self.compat else 'compat-v2' if self.compat_v2 else 'default'

    def _legacy_sasa_atoms(self, include_hetatm=False):
        """Atoms the freesasa executable uses on the whole PDB (default options)."""
        sel = "all" if include_hetatm else "record_type ATOM"
        return _first_altloc(heavy_atoms(self.u.select_atoms(sel)))

    def _prepare(self):
        self.frame_count = 0
        os.makedirs(self.outdir, exist_ok=True)
        self._write_progress(0, self.n_frames)
        if self._verbose:
            print(self.selection_report())

        prot_sel = self._validated_prot_sel
        gly_sel = self._validated_gly_sel
        block_sel = self._validated_block_sel

        self.compat = bool(self.compat)
        self.compat_v2 = bool(self.compat_v2)
        self._box = self.compat                        # box blocking test
        self._legacy = self.compat or self.compat_v2   # shared legacy semantics
        self._v2_cyl = self.compat_v2                  # GLYCO-2 cylinder quirks

        # Distance tests:
        #   default     inclusive float64 KD-tree query with --dc (self._cutoff)
        #   --compat    int(cutoff), strict '<' in the box workers
        #   --compat-v2 float32 'norm < cutoff' in _v2_pairs
        self._cutoff = float(self.distance_cutoff)
        self._legacy_box_cutoff = float(int(self.distance_cutoff))

        # --ps: START residues (heavy atoms) and their CA output columns
        self.selag = heavy_atoms(self.u.select_atoms(f"({prot_sel})"))
        self.prot_atoms = self.u.select_atoms(f"({prot_sel}) and name CA")

        # Blockers from --bs (not surface-limited); glycans from --gs
        self.blocker_atoms = heavy_atoms(self.u.select_atoms(f"({block_sel})"),
                                         legacy=self._legacy)
        self.gly_atoms = heavy_atoms(self.u.select_atoms(f"({gly_sel})"),
                                     legacy=self._legacy)

        self._residue_resindices = np.unique(self.selag.resindices)
        self._n_res_rows = len(self._residue_resindices)

        # ---------------- FreeSASA ----------------
        #   --compat / --compat-v2: mirror the freesasa executable on the whole
        #     PDB, as both legacy scripts run it (no --hetatm): heavy atoms of
        #     ATOM records, first altloc only. --sasa-hetatm adds HETATM.
        #     --sasa-env replaces the legacy set.
        #   default: start residues + blockers (+ --sasa-env).
        if self._legacy and not self.sasa_env:
            try:
                sasa_ag = self._legacy_sasa_atoms(include_hetatm=bool(self.sasa_hetatm))
            except Exception as e:  # noqa: BLE001
                self._error("could not build the legacy FreeSASA atom set "
                            f"(needs PDB record types; use --sasa-env): {e}")
        else:
            sasa_sel = f"({prot_sel}) or ({block_sel})"
            if self.sasa_env:
                sasa_sel += f" or ({self.sasa_env})"
            sasa_ag = heavy_atoms(self.u.select_atoms(sasa_sel))
        if len(sasa_ag) == 0:
            self._error("FreeSASA atom set is empty")
        sasa_res = np.unique(sasa_ag.resindices)

        fsa = FreesasaAnalysis(sasa_ag, self.probe).run(
            verbose=self._verbose,
            progressbar_kwargs={'desc': 'Running FreeSASA Analysis'},
            **self._run_slice,
        )
        self._diag("[SASA diagnostic]", f"mode={self._mode_name()}",
                   f"universe_atoms={len(self.u.atoms)}",
                   f"sasa_atoms={len(sasa_ag)}",
                   f"sasa_residues={len(sasa_res)}",
                   f"start_residues={self._n_res_rows}",
                   f"structure_valid={fsa.structure_valid}")

        if not fsa.structure_valid:
<<<<<<< HEAD
            #msg = fsa.structure_message
            msg = ("FreeSASA structure did not match the MDAnalysis residues; "
                   "SASA used MDAnalysis vdW radii instead of ProtOr, so areas "
                   "will not match the freesasa executable")
=======
            # msg = ("FreeSASA structure did not match the MDAnalysis residues; "
            #        "SASA used MDAnalysis vdW radii instead of ProtOr, so areas "
            #        "will not match the freesasa executable")
            msg = fsa.structure_message
>>>>>>> b59cecf (fixed failing tests)
            if self._legacy:
                self._error(msg)
            else:
                warnings.warn(msg, stacklevel=2)

        areas = np.asarray(fsa.results.residue_areas, dtype=float)
        if areas.shape != (self.n_frames, len(sasa_res)):
            raise ValueError("Unexpected FreeSASA residue-area shape: got "
                             f"{areas.shape}, expected {(self.n_frames, len(sasa_res))}")

        # FreesasaAnalysis columns follow np.unique(ag.resindices)
        column_of = {int(r): i for i, r in enumerate(sasa_res)}
        missing = [int(r) for r in self._residue_resindices if int(r) not in column_of]
        if missing:
            if self._legacy:
                # e.g. protein residues stored as HETATM: absent from the
                # legacy .rsa, so never surface residues in GLYCO v1/GLYCO-2.
                warnings.warn(f"{len(missing)} start residue(s) are not in the "
                              "FreeSASA set and are treated as buried (SASA 0): "
                              f"{missing[:10]}", stacklevel=2)
            else:
                raise ValueError("SASA is missing residues required by the "
                                 f"protein selection: {missing[:10]}")

        self.residue_areas = np.zeros((self.n_frames, self._n_res_rows))
        for row, resindex in enumerate(self._residue_resindices):
            col = column_of.get(int(resindex))
            if col is not None:
                self.residue_areas[:, row] = areas[:, col]

        self._diag("[SASA diagnostic]",
                   f"surface_residues={int(np.count_nonzero(self.residue_areas[0] >= self.area_cutoff))}",
                   f"area_cutoff={self.area_cutoff}")

        # ---------------- residue bookkeeping ----------------
        self._blocker_resindex = np.asarray(self.blocker_atoms.resindices)

        if not np.isin(self.prot_atoms.resindices, self._residue_resindices).all():
            raise ValueError("Some CA atoms belong to residues not present in the "
                             "prot_sel heavy-atom set.")
        if len(np.unique(self.prot_atoms.resindices)) != self.prot_atoms.n_atoms:
            raise ValueError("More than one 'CA' atom per residue in the protein "
                             "selection; results are per residue and need exactly one.")
        self._prot_atom_res_row = np.searchsorted(self._residue_resindices,
                                                  self.prot_atoms.resindices)

        # Line-of-sight starts: CA (default) or all legacy-heavy atoms
        if self._legacy:
            self.start_atoms = heavy_atoms(self.selag, legacy=True)
            self._start_res_row = np.searchsorted(self._residue_resindices,
                                                  self.start_atoms.resindices)
        else:
            self.start_atoms = self.prot_atoms
            self._start_res_row = self._prot_atom_res_row
        self._start_resindex = np.asarray(self.start_atoms.resindices)

        # (resname, chain, resid+icode): GLYCO v1 same-residue identity
        self._start_residue_keys = self._residue_keys(self.start_atoms)
        self._blocker_residue_keys = self._residue_keys(self.blocker_atoms)

        n_ca = self.prot_atoms.n_atoms
        self.all_counts = np.zeros([self.n_frames, n_ca])

        self.results = Results()
        self.results.residue_counts = self.all_counts
        self.results.n_surface_residues = np.zeros(self.n_frames, dtype=int)
        self.results.n_blocked_pairs = np.zeros(self.n_frames, dtype=int)
        self.results.mean_blocked_distance = np.full(self.n_frames, np.nan)

        # caches for the per-atom CSV writers (after all_counts exists)
        self._blocker_cache = None
        self._box_cache = None
        self._output_k = self._resolve_frame_index(self.bfactor_frame)

        self._progress_every = max(1, self.n_frames // 100)

        self._diag("[compat diagnostic]", f"mode={self._mode_name()}",
                   f"radius={self.radius}", f"legacy={self._legacy}",
                   f"box={self._box}", f"box_workers={self.box_workers}",
                   f"start_atoms={len(self.start_atoms)}",
                   f"blocker_atoms={len(self.blocker_atoms)}",
                   f"glycan_atoms={len(self.gly_atoms)}",
                   f"protein_CA={len(self.prot_atoms)}")

    def _single_frame(self):
        k = self._frame_index
        start_idx = self.surface_start_indices(k)
        self.results.n_surface_residues[k] = len(np.unique(self._start_res_row[start_idx]))

        if self._box:
            # GLYCO v1 box test
            cand_keys, vis_keys, n_blocked, mean_d = self._box_visibility(
                start_idx, workers=self.box_workers)
            self.all_counts[k] = self._counts_from_visible_keys(vis_keys)
            self.results.n_blocked_pairs[k] = n_blocked
            self.results.mean_blocked_distance[k] = mean_d
            if k == self._output_k:
                self._box_cache = (k, start_idx, cand_keys, vis_keys)
            self._diag("[compat diagnostic]", f"frame={k}",
                       f"surface_start_atoms={len(start_idx)}",
                       f"candidate_keys={len(cand_keys)}",
                       f"visible_keys={len(vis_keys)}",
                       f"nonzero={np.count_nonzero(self.all_counts[k])}",
                       f"total={int(self.all_counts[k].sum())}")
        else:
            # cylinder test (default, or GLYCO-2 float32 with --compat-v2)
            blockers, dists, gtree = self.get_blockers(start_idx)
            self.all_counts[k] = self.count_glycan_neighbors(blockers, gtree, start_idx)

            if len(blockers):
                n_gly = len(self.gly_atoms)
                pair_key = (blockers[:, 0].astype(np.int64) * n_gly
                            + blockers[:, 2].astype(np.int64))
                _, first = np.unique(pair_key, return_index=True)
                self.results.n_blocked_pairs[k] = len(first)
                self.results.mean_blocked_distance[k] = float(dists[first].mean())

            if k == self._output_k:
                self._blocker_cache = (k, start_idx, blockers, dists)
            self._diag("[compat diagnostic]", f"frame={k}",
                       f"surface_start_atoms={len(start_idx)}",
                       f"blocker_rows={len(blockers)}",
                       f"nonzero={np.count_nonzero(self.all_counts[k])}",
                       f"total={int(self.all_counts[k].sum())}")

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
        if self.write_blockers and not self._box:
            self.write_blockers_csv(frame_index=self.bfactor_frame)

        self._blocker_cache = None
        self._box_cache = None

        if self.write_vmd:
            self.write_vmd_user_script()

    # -- geometry ------------------------------------------------------------

    def outermost(self, frame_index=None):
        """Residue rows that are SURFACE residues (SASA >= area_cutoff)."""
        if frame_index is None:
            frame_index = self._frame_index
        return np.flatnonzero(self.residue_areas[frame_index] >= self.area_cutoff)

    def surface_start_indices(self, frame_index=None):
        """Indices (into self.start_atoms) of start atoms in surface residues."""
        surf_res_rows = self.outermost(frame_index)
        return np.flatnonzero(np.isin(self._start_res_row, surf_res_rows))

    def _v2_pairs(self, start_idx):
        """(i, g) pairs exactly as c_worker selects them: float32 norm < float32 cutoff.
        i indexes start_idx (local), g indexes gly_atoms."""
        start_idx = np.asarray(start_idx, dtype=np.intp)
        P32 = self.start_atoms.positions[start_idx].astype(np.float32)
        G32 = self.gly_atoms.positions.astype(np.float32)
        empty = np.empty(0, dtype=np.intp)
        if len(P32) == 0 or len(G32) == 0:
            return empty, empty, P32, G32
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

    def _candidate_pairs(self, start_idx, chunk=20000):
        """
        Yield (start_ids, gly_ids) int64 arrays (global start_atoms / gly_atoms
        indices) of pairs within the cutoff, using the distance test of the
        active cylinder mode: GLYCO-2's float32 'norm < cutoff' with
        --compat-v2, an inclusive float64 KD-tree query otherwise.
        Shared by counts and the excluded-atom writer so both agree.
        """
        start_idx = np.asarray(start_idx, dtype=np.intp)
        if len(start_idx) == 0 or len(self.gly_atoms) == 0:
            return
        if self._v2_cyl:
            i, g, _, _ = self._v2_pairs(start_idx)
            if len(i):
                yield start_idx[i].astype(np.int64), g.astype(np.int64)
            return
        P = np.asarray(self.start_atoms.positions, dtype=np.float64)
        gtree = cKDTree(np.asarray(self.gly_atoms.positions, dtype=np.float64))
        for s in range(0, len(start_idx), chunk):
            idx = start_idx[s:s + chunk]
            neighbors = gtree.query_ball_point(P[idx], self._cutoff)
            sizes = np.fromiter((len(x) for x in neighbors), dtype=np.intp,
                                count=len(neighbors))
            if sizes.sum() == 0:
                continue
            yield (np.repeat(idx.astype(np.int64), sizes),
                   np.concatenate(neighbors).astype(np.int64, copy=False))

    def _iter_pairs(self, P, gtree, ca_chunk):
        """Yield (i, j, dist) start/glycan pairs within self._cutoff (default mode)."""
        for s in range(0, len(P), ca_chunk):
            sub = P[s:s + ca_chunk]
            pairs = cKDTree(sub).sparse_distance_matrix(
                gtree, self._cutoff, output_type="ndarray")
            yield (pairs["i"].astype(np.intp) + s,
                   pairs["j"].astype(np.intp),
                   pairs["v"])

    def get_blockers(self, start_idx=None, chunk=20000, ca_chunk=2000):
        """
        Blocking protein atoms of every (start atom, glycan atom) pair within
        the cutoff (cylinder test). Default: own residue excluded, strict
        radius. --compat-v2: GLYCO-2's float32 c_count_points semantics.

        Returns (blockers (n, 3) [start_atoms idx, blocker idx, gly idx],
                 distances (n,), gtree or None).
        """
        if start_idx is None:
            start_idx = self.surface_start_indices()
        start_idx = np.asarray(start_idx, dtype=np.intp)

        if self._v2_cyl:
            return self._get_blockers_v2(start_idx, chunk)

        P = np.ascontiguousarray(self.start_atoms.positions[start_idx], dtype=np.float64)
        B = np.ascontiguousarray(self.blocker_atoms.positions, dtype=np.float64)
        G = np.ascontiguousarray(self.gly_atoms.positions, dtype=np.float64)

        gtree = cKDTree(G) if len(G) else None
        empty = (np.empty((0, 3), dtype=np.intp), np.empty(0, dtype=np.float64))
        if len(P) == 0 or len(B) == 0 or len(G) == 0:
            return (*empty, gtree)

        btree = cKDTree(B)
        start_resindices = self._start_resindex[start_idx]
        blocker_resindices = self._blocker_resindex
        r2 = self.radius ** 2

        out, dist_out = [], []
        for ii, gg, distances in self._iter_pairs(P, gtree, ca_chunk):
            for s in range(0, len(ii), chunk):
                i_c, g_c, d_c = ii[s:s + chunk], gg[s:s + chunk], distances[s:s + chunk]

                b = P[i_c] - G[g_c]
                half = 0.5 * np.linalg.norm(b, axis=-1)
                mid = 0.5 * (P[i_c] + G[g_c])

                cand = btree.query_ball_point(mid, half + self.radius)
                counts = np.fromiter(map(len, cand), dtype=np.intp, count=len(cand))
                if counts.sum() == 0:
                    continue
                jj = np.concatenate(cand).astype(np.intp, copy=False)
                rep = np.repeat(np.arange(len(i_c)), counts)

                # same-residue exclusion
                other_res = blocker_resindices[jj] != start_resindices[i_c[rep]]
                jj, rep = jj[other_res], rep[other_res]
                if len(jj) == 0:
                    continue

                # exact point-to-segment test; t > 0 rejects the start atom
                bb = b[rep]
                a = P[i_c[rep]] - B[jj]
                seg_len2 = np.maximum(np.einsum("ij,ij->i", bb, bb), 1e-12)
                t = np.einsum("ij,ij->i", a, bb) / seg_len2
                perp = a - t[:, None] * bb
                d2 = np.einsum("ij,ij->i", perp, perp)
                keep = (t > 0) & (t < 1) & (d2 < r2)

                if keep.any():
                    out.append(np.stack([start_idx[i_c[rep][keep]],
                                         jj[keep],
                                         g_c[rep][keep]], axis=1))
                    dist_out.append(d_c[rep][keep])

        if not out:
            return (*empty, gtree)
        res = np.concatenate(out)
        dist_res = np.concatenate(dist_out)
        order = np.lexsort((res[:, 2], res[:, 1], res[:, 0]))
        return res[order], dist_res[order], gtree

    def _get_blockers_v2(self, start_idx, chunk=20000):
        """GLYCO-2 cylinder test (c_worker / c_count_points semantics, float32)."""
        empty = (np.empty((0, 3), dtype=np.intp), np.empty(0, dtype=np.float64))
        start_idx = np.asarray(start_idx, dtype=np.intp)
        if len(start_idx) == 0 or len(self.blocker_atoms) == 0 or len(self.gly_atoms) == 0:
            return (*empty, None)

        i, g, P32, G32 = self._v2_pairs(start_idx)
        if len(i) == 0:
            return (*empty, None)

        # GLYCO-2 parses float32 coordinates and stacks them into float64
        B64 = np.ascontiguousarray(
            self.blocker_atoms.positions.astype(np.float32).astype(np.float64))
        btree = cKDTree(B64)
        P64, G64 = P32.astype(np.float64), G32.astype(np.float64)

        out, dist_out = [], []
        for s in range(0, len(i), chunk):
            ic, gc = i[s:s + chunk], g[s:s + chunk]
            mid = 0.5 * (P64[ic] + G64[gc])
            half = 0.5 * np.linalg.norm(P64[ic] - G64[gc], axis=1)
            # every point of the cylinder lies within half + r of the midpoint
            cand = btree.query_ball_point(mid, half + self.radius + 1e-3)
            counts = np.fromiter(map(len, cand), dtype=np.intp, count=len(cand))
            if counts.sum() == 0:
                continue
            jj = np.concatenate(cand).astype(np.intp, copy=False)
            rep = np.repeat(np.arange(len(ic)), counts)
            ii, gg = ic[rep], gc[rep]
            # no same-residue exclusion, inclusive radius (as in GLYCO-2)
            hit = _v2_cylinder_blocked(P32, G32, B64, ii, gg, jj, self.radius)
            if hit.any():
                ii, gg, jh = ii[hit], gg[hit], jj[hit]
                out.append(np.stack([start_idx[ii], jh, gg], axis=1))
                dist_out.append(
                    np.linalg.norm(G32[gg] - P32[ii], axis=1).astype(np.float64))

        if not out:
            return (*empty, None)
        res, dres = np.concatenate(out), np.concatenate(dist_out)
        order = np.lexsort((res[:, 2], res[:, 1], res[:, 0]))
        return res[order], dres[order], None

    @staticmethod
    def _residue_keys(atom_group):
        """(resname, chain, resid+icode) per atom, as the legacy scripts key residues."""
        try:
            chains = [str(c).strip() for c in atom_group.chainIDs]
        except (AttributeError, mda.exceptions.NoDataError):
            chains = [""] * len(atom_group)
        return [
            (str(resname).strip(), chain_id, str(label))
            for resname, chain_id, label in zip(
                atom_group.resnames, chains, _resid_labels(atom_group))
        ]

    def count_glycan_neighbors(self, blockers, gtree=None, start_idx=None, chunk=20000):
        """
        Non-blocked glycan neighbours per CA (length len(self.prot_atoms)).
        Default: per surface CA. Legacy: UNIQUE glycan atoms visible from at
        least one heavy atom of the residue, broadcast to its CA.
        gtree is kept for API compatibility.
        """
        n_start, n_gly = len(self.start_atoms), len(self.gly_atoms)
        counts = np.zeros(len(self.prot_atoms), dtype=np.intp)
        if start_idx is None:
            start_idx = self.surface_start_indices()
        if n_start == 0 or n_gly == 0 or len(start_idx) == 0:
            return counts

        excluded = (np.unique(blockers[:, 0].astype(np.int64) * n_gly
                              + blockers[:, 2].astype(np.int64))
                    if len(blockers) else np.empty(0, dtype=np.int64))

        per_start = np.zeros(n_start, dtype=np.intp)
        visible_keys = []
        for start_ids, gly_ids in self._candidate_pairs(start_idx, chunk):
            if len(excluded):
                keep = ~np.isin(start_ids * n_gly + gly_ids, excluded)
                start_ids, gly_ids = start_ids[keep], gly_ids[keep]
            if self._legacy:
                visible_keys.append(np.unique(
                    self._start_res_row[start_ids].astype(np.int64) * n_gly + gly_ids))
            else:
                per_start += np.bincount(start_ids, minlength=n_start)

        if not self._legacy:
            return per_start              # start atoms ARE the CA atoms
        if not visible_keys:
            return counts
        keys = np.unique(np.concatenate(visible_keys))
        per_res = np.bincount(keys // n_gly, minlength=self._n_res_rows)
        return per_res[self._prot_atom_res_row].astype(np.intp)

    def _box_visibility(self, start_idx=None, workers=None, starts_per_task=32):
        """
        Original GLYCO v1 box test (--compat); --cr is ignored.

        Returns (cand_keys, vis_keys, n_blocked, mean_blocked_distance), keys
        being sorted unique residue_row * n_gly + glycan_index values.
        workers=1 runs serially in this process.
        """
        if start_idx is None:
            start_idx = self.surface_start_indices()
        empty = np.empty(0, dtype=np.int64)
        n_gly = len(self.gly_atoms)
        if len(start_idx) == 0 or n_gly == 0:
            return empty, empty, 0, np.nan

        start_idx = np.asarray(start_idx, dtype=np.int64)
        blocker_positions = _legacy_coords(self.blocker_atoms.positions)
        initargs = (
            _legacy_coords(self.start_atoms.positions[start_idx]),
            np.asarray(self._start_res_row[start_idx], dtype=np.int64),
            [self._start_residue_keys[i] for i in start_idx],
            _legacy_coords(self.gly_atoms.positions),
            blocker_positions,
            self._blocker_residue_keys,
            _build_x_buckets(blocker_positions),   # rebuilt for THIS frame
            self._legacy_box_cutoff,               # v1: int(cutoff), strict '<'
            n_gly,
        )

        n_starts = len(start_idx)
        ranges = [(lo, min(lo + starts_per_task, n_starts))
                  for lo in range(0, n_starts, starts_per_task)]
        if workers is None:
            workers = getattr(self, "box_workers", 1) or 1
        workers = max(1, min(int(workers), len(ranges)))

        if workers == 1:
            _init_box_worker(*initargs)
            try:
                results = [_box_chunk(r) for r in ranges]
            finally:
                _BOX_STATE.clear()
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_box_worker,
                initargs=initargs,
            ) as executor:
                results = list(executor.map(_box_chunk, ranges))

        cand = [r[0] for r in results if len(r[0])]
        vis = [r[1] for r in results if len(r[1])]
        cand_keys = np.unique(np.concatenate(cand)) if cand else empty
        vis_keys = np.unique(np.concatenate(vis)) if vis else empty
        n_blocked = sum(int(r[2]) for r in results)
        dist_sum = sum(float(r[3]) for r in results)
        mean_d = dist_sum / n_blocked if n_blocked else np.nan
        return cand_keys, vis_keys, n_blocked, mean_d

    def _counts_from_visible_keys(self, vis_keys):
        """Per-CA counts from visible (residue_row * n_gly + glycan) keys."""
        counts = np.zeros(len(self.prot_atoms), dtype=np.intp)
        if len(vis_keys):
            n_gly = len(self.gly_atoms)
            per_res = np.bincount(vis_keys // n_gly, minlength=self._n_res_rows)
            counts = per_res[self._prot_atom_res_row].astype(np.intp)
        return counts

    def count_glycan_neighbors_box(self, start_idx=None, workers=None, **_ignored):
        """
        GLYCO v1 counts (--compat): lines sampled at t = 0, 1/d, 2/d, ...,
        int-truncated sample points, blocked by any heavy atom outside the
        start residue within a +-1 A axis-aligned box.

        Returns (counts per CA, n blocked pairs, mean blocked distance).
        """
        if workers is None:
            workers = self.box_workers
        _, vis_keys, n_blocked, mean_d = self._box_visibility(start_idx, workers=workers)
        return self._counts_from_visible_keys(vis_keys), n_blocked, mean_d

    # -- output --------------------------------------------------------------

    @staticmethod
    def _rows_for(residue_resindices, atom_resindices):
        """Row (position in residue_resindices) of each atom's residue."""
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
        """Sorted unique keys -> {residue_row: array_of_glycan_indices}."""
        if len(keys) == 0:
            return {}
        rows = keys // n_gly
        gly = keys % n_gly
        uniq, first = np.unique(rows, return_index=True)
        return dict(zip(uniq.tolist(), np.split(gly, first[1:])))

    def _analyzed_frames(self):
        """Trajectory frame index of each analyzed frame."""
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
                f"frame index {frame_index} out of range for {n} analyzed frame(s)")
        return k

    def _frame_blockers(self, k):
        """(start_idx, blockers, distances) for analyzed frame k (cylinder test)."""
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
        (start_idx, cand_keys, vis_keys) for analyzed frame k:
        cand_keys = glycan atoms within the cutoff of a start atom of the
        residue, vis_keys = those visible from at least one of them. Uses the
        same pair/blocking tests as the counts.
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
                cand_keys, vis_keys, _, _ = self._box_visibility(
                    start_idx, workers=self.box_workers)
            finally:
                self.u.trajectory[old_frame]
            return start_idx, cand_keys, vis_keys

        # cylinder test: blocked pairs (cached from the run when possible)
        start_idx, blockers, _ = self._frame_blockers(k)
        blocked_keys = (np.unique(blockers[:, 0].astype(np.int64) * n_gly
                                  + blockers[:, 2].astype(np.int64))
                        if len(blockers) else empty)

        cand_parts, vis_parts = [], []
        old_frame = int(self.u.trajectory.frame)
        try:
            self.u.trajectory[trajectory_frame]
            for start_ids, gly_ids in self._candidate_pairs(start_idx, chunk):
                res_rows = self._start_res_row[start_ids].astype(np.int64)
                cand_parts.append(np.unique(res_rows * n_gly + gly_ids))
                keep = ~np.isin(start_ids * n_gly + gly_ids, blocked_keys)
                vis_parts.append(np.unique(res_rows[keep] * n_gly + gly_ids[keep]))
        finally:
            self.u.trajectory[old_frame]

        cand_keys = np.unique(np.concatenate(cand_parts)) if cand_parts else empty
        vis_keys = np.unique(np.concatenate(vis_parts)) if vis_parts else empty
        return start_idx, cand_keys, vis_keys

    def _ensure_tempfactors(self):
        try:
            self.u.atoms.tempfactors  # noqa: B018
        except (AttributeError, mda.exceptions.NoDataError):
            self.u.add_TopologyAttr('tempfactors')

    def write_blockers_csv(self, frame_index=None, output=None, chunk=100_000):
        """One row per (start atom, blocking atom, glycan atom) for one frame."""
        if self._box:
            raise ValueError("blockers.csv is not available with --compat: the box "
                             "test does not identify blocking atoms")

        if output is None:
            output = os.path.join(self.outdir, "blockers.csv")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        k = self._resolve_frame_index(frame_index)
        trajectory_frame = int(self._analyzed_frames()[k])

        _, blockers, distances = self._frame_blockers(k)
        s_i, b_i, g_i = blockers.T
        n = len(blockers)
        distances = np.asarray(distances, dtype=np.float64)

        start, blocker, glycan = self.start_atoms, self.blocker_atoms, self.gly_atoms
        attr = self._atom_attr

        start_id, blocker_id, gly_id = (pdb_atom_ids(start), pdb_atom_ids(blocker),
                                        pdb_atom_ids(glycan))
        start_name, start_resname = attr(start, "names"), attr(start, "resnames")
        start_resid, start_chain = _resid_labels(start), attr(start, "chainIDs")
        blk_name, blk_resname = attr(blocker, "names"), attr(blocker, "resnames")
        blk_resid, blk_chain = _resid_labels(blocker), attr(blocker, "chainIDs")
        gly_name, gly_resname = attr(glycan, "names"), attr(glycan, "resnames")
        gly_resid, gly_chain = _resid_labels(glycan), attr(glycan, "chainIDs")

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
        """b_factors.pdb (whole structure, one frame) and b_factors.npy (all frames)."""
        os.makedirs(self.outdir, exist_ok=True)
        self._ensure_tempfactors()

        k = self._resolve_frame_index(frame_index)
        self.u.trajectory[int(self._analyzed_frames()[k])]

        self._populate_b_factors(k)
        self.u.atoms.write(os.path.join(self.outdir, 'b_factors.pdb'))
        np.save(os.path.join(self.outdir, 'b_factors'), np.asarray(self.all_counts))

    def _populate_b_factors(self, frame_index=None):
        """Every atom of a CA residue gets that residue's value; others 0."""
        data = np.asarray(self.all_counts)
        if data.ndim == 1:
            ca_values = data
        elif data.ndim == 2:
            ca_values = data[self._resolve_frame_index(frame_index)]
        else:
            raise ValueError(f"self.all_counts must be 1D or 2D, got shape {data.shape}")

        ca_resindices = self.prot_atoms.resindices
        if len(ca_values) != len(ca_resindices):
            raise ValueError(f"self.all_counts has {len(ca_values)} values "
                             f"but self.prot_atoms has {len(ca_resindices)} CA atoms")

        if ca_values.size and ca_values.max() > 999.99:
            warnings.warn(
                f"max B-factor value {ca_values.max():g} exceeds the PDB "
                "B-factor column width (999.99); the PDB columns will be "
                "misaligned. Use b_factors.npy for the raw values.",
                stacklevel=2,
            )

        res_atoms = self.prot_atoms.residues.atoms
        rows = self._rows_for(ca_resindices, res_atoms.resindices)
        self.u.atoms.tempfactors = np.zeros(self.u.atoms.n_atoms)
        res_atoms.tempfactors = ca_values[rows]

    def write_frame_summary(self):
        """frame_summary.csv: one row per analyzed frame."""
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
        Per-residue glycan density and SASA: Chain,ResNum,ResName,Glycan_density,SASA.
        ResNum includes the insertion code. Compat modes write nonzero-count
        residues only (as GLYCO-2 and compare.py expect).
        """
        if output is None:
            output = os.path.join(self.outdir, "residue_summary.csv")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        k = self._resolve_frame_index(frame_index)
        glycan_density = np.asarray(self.all_counts)[k]
        if len(glycan_density) != len(self.prot_atoms):
            raise ValueError("The number of glycan-density values does not match "
                             "the number of protein CA atoms.")

        sasa_rows = np.searchsorted(self._residue_resindices, self.prot_atoms.resindices)
        sasa_values = np.asarray(self.residue_areas)[k, sasa_rows]

        resnums = _resid_labels(self.prot_atoms)
        resnames = np.asarray(self.prot_atoms.resnames)
        try:
            chains = np.asarray(self.prot_atoms.chainIDs)
        except (AttributeError, mda.exceptions.NoDataError):
            chains = np.full(len(self.prot_atoms), "", dtype=object)

        if not (len(resnums) == len(resnames) == len(chains)
                == len(glycan_density) == len(sasa_values)):
            raise ValueError("Residue attributes have inconsistent lengths.")

        # Compatibility outputs omit zero-count residues, like the legacy
        # programs; default output retains all selected protein residues.
        nonzero_only = bool(getattr(self, "compat", False)
                            or getattr(self, "compat_v2", False))

        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Chain", "ResNum", "ResName", "Glycan_density", "SASA"])
            n_written = 0
            for chain_id, resnum, resname, density, sasa in zip(
                    chains, resnums, resnames, glycan_density, sasa_values):
                density = int(density)
                if nonzero_only and density == 0:
                    continue
                writer.writerow([str(chain_id).strip(), str(resnum),
                                 str(resname).strip(), density,
                                 f"{float(sasa):.6g}"])
                n_written += 1

        print(f"Residue summary written to: {output} ({n_written} residues)")
        return output

    def _write_progress(self, current, total):
        """Machine-readable progress for the Streamlit app."""
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
        """Quote a string as one Tcl word."""
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
        """VMD Tcl script assigning per-residue data to 'user' for every analyzed frame."""
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
        trajectory_files = [Path(p).resolve() for p in trajectory_files if p is not None]
        if not trajectory_files:
            raise ValueError("Could not determine trajectory filename(s)")
        for path in trajectory_files:
            if not path.is_file():
                raise FileNotFoundError(path)

        data = np.asarray(self.all_counts).T
        residues = self.prot_atoms.residues
        if data.shape[0] != len(residues):
            raise ValueError(f"Data has {data.shape[0]} rows, but prot_atoms contains "
                             f"{len(residues)} residues")
        if not np.all(np.isfinite(data)):
            raise ValueError("data contains NaN or infinite values")

        n_frames = data.shape[1]
        analyzed = self._analyzed_frames()
        n_traj_frames = int(reader.n_frames)

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
                f"mol addfile {self._tcl_quote(traj.as_posix())} waitfor all molid $molid")
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
            ('    error "Frame mismatch: VMD has $n_vmd_frames frames, but the '
             'trajectory has $n_traj_frames. Expected equal counts or exactly '
             'one extra VMD frame."'),
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

        lines += ["$sel delete", 'puts "Finished assigning per-residue user values."', ""]

        if output is None:
            output = os.path.join(self.outdir, VMD_SCRIPT_NAME)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines))

        print(f"VMD script written to: {output}")
        return output

    def write_excluded_glycan_atoms_csv(self, frame_index=None, output=None):
        """
        One row per glycan atom absent from the final residue-level visibility
        union, for every SURFACE residue with a nonzero count.

        reason: outside_distance_cutoff | blocked_for_all_start_atoms
        glycan_atom_index is the PDB serial number.
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

        gly_ids_pdb = pdb_atom_ids(self.gly_atoms)
        gly_names = self._atom_attr(self.gly_atoms, "names")
        gly_resnames = self._atom_attr(self.gly_atoms, "resnames")
        gly_chains = self._atom_attr(self.gly_atoms, "chainIDs")
        gly_resids = _resid_labels(self.gly_atoms)

        ca_resnames = self._atom_attr(self.prot_atoms, "resnames")
        ca_chains = self._atom_attr(self.prot_atoms, "chainIDs")
        ca_resids = _resid_labels(self.prot_atoms)
        ca_of_row = {int(r): i for i, r in enumerate(self._prot_atom_res_row)}

        surface_rows = (np.unique(self._start_res_row[start_idx])
                        if len(start_idx) else empty)

        fieldnames = [
            "analyzed_frame", "trajectory_frame",
            "protein_resname", "protein_chain", "protein_resid",
            "final_glycan_count",
            "glycan_atom_index", "glycan_atom_name", "glycan_resname",
            "glycan_chain", "glycan_resid",
            "reason",
        ]

        n_rows = 0
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)

            for row in surface_rows.tolist():
                ca = ca_of_row.get(row)
                if ca is None:
                    continue

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
                if final_count == 0:
                    continue

                excluded = np.flatnonzero(~is_visible)
                if len(excluded) == 0:
                    continue

                reasons = np.where(is_candidate[excluded],
                                   "blocked_for_all_start_atoms",
                                   "outside_distance_cutoff")

                writer.writerows(zip(
                    repeat(k), repeat(trajectory_frame),
                    repeat(str(ca_resnames[ca]).strip()),
                    repeat(str(ca_chains[ca]).strip()),
                    repeat(str(ca_resids[ca])),
                    repeat(final_count),
                    gly_ids_pdb[excluded].tolist(),
                    gly_names[excluded].tolist(),
                    gly_resnames[excluded].tolist(),
                    gly_chains[excluded].tolist(),
                    gly_resids[excluded].tolist(),
                    reasons.tolist(),
                ))
                n_rows += len(excluded)

        print(f"Excluded glycan atoms written to: {output} ({n_rows} rows)")
        return output

    def _atom_residue_keys(self, atom_group):
        """GLYCO-style residue identities (resname, chain, resid+icode)."""
        return np.asarray(self._residue_keys(atom_group), dtype=object)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_job(args, parser, top, traj, outdir, verbose):
    """Load one structure/trajectory, run the analysis, write outputs."""
    u = load_universe(top, traj)
    analysis = GlycoAnalysis(u, parser=parser, args=args,
                             top=top, traj=traj, indir=None, outdir=outdir)
    analysis.run(verbose=verbose,
                 progressbar_kwargs={'desc': 'Running GlyCo Analysis'})


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

    # Validate everything that does not need a Universe BEFORE loading files.
    validate_arguments(args, parser.error)

    is_streamlit = os.environ.get("GLYCO_STREAMLIT") == "1"
    verbose = False if is_streamlit else args.verbose

    os.makedirs(args.outdir, exist_ok=True)
    configure_warning_log(os.path.join(args.outdir, "mdanalysis_warnings.log"))

    if args.indir is None:
        if not confirm_overwrite(args.outdir, args.force, args.write_vmd,
                                 args.write_excluded, args.write_blockers):
            return 1
        try:
            u = load_universe(args.top, args.traj)
        except Exception as e:  # noqa: BLE001
            parser.error("could not load -a/--top with -b/--traj (do all trajectory "
                         f"files match the topology's atom count?): {e}")
        GlycoAnalysis(u, parser=parser, args=args).run(
            verbose=verbose,
            progressbar_kwargs={'desc': 'Running GlyCo Analysis'},
        )
        return 0

    # --indir: each structure analyzed independently into its own subfolder
    outdirs = _structure_outdirs(args.structures, args.outdir)
    failures = []
    for n, (path, outdir) in enumerate(zip(args.structures, outdirs), start=1):
        print(f"[{n}/{len(args.structures)}] {path}")
        if not confirm_overwrite(outdir, args.force, args.write_vmd,
                                 args.write_excluded, args.write_blockers):
            failures.append((path, "output exists and overwrite declined"))
            continue
        try:
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
