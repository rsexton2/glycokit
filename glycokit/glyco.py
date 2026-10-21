#!/usr/bin/env python3
# ruff: noqa: EXE001
"""
glyco: Glycan shielding of protein surface residues (glycokit / GLYCO v3).

Counting modes
    default      heavy-atom starts (per-residue union), float64 pair test
                 (norm < --dc), van der Waals sphere blocking test
                 (--vdw-scale, --vdw-probe), own residue excluded.
    --cylinder   default, but with the cylinder blocking test (--cr, inclusive).
    --compat     reproduces GLYCO v1.
    --compat-v2  reproduces GLYCO-2.
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
from MDAnalysis.guesser.default_guesser import DefaultGuesser
from MDAnalysis.guesser.tables import vdwradii as VDW_RADII
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

# Padding of the KD-tree prefilters (pair search and the cylinder / vdW
# candidate searches). The exact tests follow, so this only guards against
# rounding.
_CYLINDER_PAD = 1e-3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resid_labels(ag):
    """
    PDB-style residue numbers including the insertion code ('52', '52A').

    This is what glyco_mod.py / core_glyco_mod.py read from
    ``line[22:28].strip()``.

    :param ag: Atoms to label.
    :type ag: MDAnalysis.core.groups.AtomGroup
    :return: One label per atom.
    :rtype: numpy.ndarray of str (dtype object)
    """
    resids = np.asarray(ag.resids)
    try:
        icodes = [str(i).strip() for i in ag.icodes]
    except (AttributeError, mda.exceptions.NoDataError):
        icodes = [""] * len(resids)
    return np.asarray([f"{int(r)}{i}" for r, i in zip(resids, icodes)],
                      dtype=object)


def _cylinder_blocked(P, G, B, i, g, j, r):
    """
    GLYCO-2 ``c_count_points`` cylinder test.

    True where blocker ``B[j]`` lies inside the cylinder of radius ``r``
    around the segment ``G[g] -> P[i]``. End caps are strict (the start and
    glycan atoms never block themselves); the radius is inclusive.
    Arithmetic follows the input dtypes: float32 ``P``/``G`` reproduce
    GLYCO-2 exactly, float64 ``P``/``G`` give the default test.

    :param numpy.ndarray P: Start-atom coordinates, shape (n, 3).
    :param numpy.ndarray G: Glycan-atom coordinates, shape (m, 3).
    :param numpy.ndarray B: Blocker coordinates, shape (k, 3).
    :param numpy.ndarray i: Row indices into ``P``.
    :param numpy.ndarray g: Row indices into ``G`` (same length as ``i``).
    :param numpy.ndarray j: Row indices into ``B`` (same length as ``i``).
    :param float r: Cylinder radius in Å.
    :return: Boolean mask, one entry per (i, g, j) triple.
    :rtype: numpy.ndarray
    """
    vec = P[i] - G[g]                                           # pt2 - pt1
    const = np.asarray(r, dtype=vec.dtype) * np.linalg.norm(vec, axis=1)
    q_pt1 = B[j] - G[g]
    q_pt2 = B[j] - P[i]
    return ((np.einsum("ij,ij->i", q_pt1, vec) > 0)
            & (np.einsum("ij,ij->i", q_pt2, vec) < 0)
            & (np.linalg.norm(np.cross(q_pt1, vec), axis=1) <= const))


def _first_altloc(ag):
    """
    Keep blank altlocs plus the first altloc letter seen in each residue.

    :param ag: Atoms to filter.
    :type ag: MDAnalysis.core.groups.AtomGroup
    :return: ``ag`` unchanged if the topology has no altLocs, else the
        filtered group.
    :rtype: MDAnalysis.core.groups.AtomGroup
    """
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
    Convert MDAnalysis positions to the float64 values GLYCO v1 parses.

    GLYCO v1 reads ``float(line[30:38])``; for standard 3-decimal PDB
    coordinates that equals rounding to 3 decimals.

    :param positions: Coordinates, shape (n, 3).
    :type positions: numpy.ndarray
    :return: float64 coordinates rounded to 3 decimals.
    :rtype: numpy.ndarray
    """
    return np.round(np.asarray(positions, dtype=np.float64), 3)


def _build_x_buckets(positions):
    """
    Reproduce glyco_mod.py's integer-x index (``gen_pro_dict``'s outdict2).

    Keys are ``int(x)`` (truncation toward zero), matching
    ``int(float(line[30:38]))`` in GLYCO v1.

    :param positions: Coordinates, shape (n, 3).
    :type positions: numpy.ndarray
    :return: Mapping ``int(x) -> tuple of row indices into positions``.
    :rtype: dict[int, tuple[int, ...]]
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
    """
    Initialize one box-test process with plain NumPy/Python data.

    No MDAnalysis objects are passed, so the state pickles cheaply for
    ``spawn`` workers.

    :param numpy.ndarray start_positions: Start coordinates, shape (n, 3).
    :param numpy.ndarray start_rows: Residue row of each start atom.
    :param list start_residue_keys: (resname, chain, resid+icode) per start.
    :param numpy.ndarray gly_positions: Glycan coordinates, shape (m, 3).
    :param numpy.ndarray blocker_positions: Blocker coordinates, shape (k, 3).
    :param list blocker_residue_keys: Residue key per blocker atom.
    :param dict blocker_x_buckets: Output of :func:`_build_x_buckets`.
    :param float cutoff: Distance cutoff (GLYCO v1: ``int(cutoff)``).
    :param int n_gly: Number of glycan atoms.
    """
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
    GLYCO v1 ``check_protein()`` for a range of (local) start atoms.

    Keys are ``residue_row * n_gly + glycan_index``. Requires
    :func:`_init_box_worker` to have been called in this process.

    :param start_range: Half-open range ``(lo, hi)`` of local start indices.
    :type start_range: tuple[int, int]
    :return: (candidate keys, visible keys, number of blocked pairs, sum of
        blocked pair distances).
    :rtype: tuple[numpy.ndarray, numpy.ndarray, int, float]
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


def pdb_atom_ids(atom_group):
    """
    PDB-style (1-based) serial numbers for every atom in an AtomGroup.

    Uses the topology ``ids`` attribute when present, otherwise falls back
    to ``index + 1`` (valid only for sequential PDB-like input).

    :param atom_group: Atoms to number.
    :type atom_group: MDAnalysis.core.groups.AtomGroup
    :return: One serial number per atom.
    :rtype: numpy.ndarray of int64
    """
    try:
        return np.asarray(atom_group.ids, dtype=np.int64)
    except (AttributeError, mda.exceptions.NoDataError):
        return np.asarray(atom_group.indices, dtype=np.int64) + 1


def configure_warning_log(path):
    """
    Route Python warnings (including MDAnalysis') into a log file.

    Calling it twice with the same path does not add a second handler.

    :param str path: Log file path.
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
    Return ``ag`` without hydrogens.

    Default: an atom is a hydrogen if its element is H/D (when the topology
    has elements) OR its name starts with H after optional digits.
    ``legacy=True`` reproduces the original GLYCO filter instead: drop every
    atom whose name contains an 'H' ANYWHERE (also NH1/NH2, OH, CH2, ...).

    :param ag: Atoms to filter.
    :type ag: MDAnalysis.core.groups.AtomGroup
    :param bool legacy: Use the original GLYCO name filter.
    :return: Heavy atoms of ``ag``.
    :rtype: MDAnalysis.core.groups.AtomGroup
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
    """
    Independent structure files in ``indir`` matching ``extensions``.

    :param str indir: Directory to search (non-recursive).
    :param extensions: Extensions without the leading dot.
    :type extensions: list[str]
    :return: Sorted paths per extension, duplicates removed, order kept.
    :rtype: list[str]
    """
    files = []
    for ext in extensions:
        files.extend(sorted(glob.glob(os.path.join(indir, f'*.{ext}'))))
    return list(dict.fromkeys(files))


def validate_arguments(cfg, error, check_inputs=True):
    """
    Validate and normalize run parameters.

    Sets ``cfg.structures``, normalizes ``cfg.frame_list`` to a list of ints
    or None, and fills ``cfg.outdir`` if empty. ``cfg`` must carry every
    :func:`base_parser` destination (a parsed Namespace, or a
    :class:`GlycoAnalysis` with the parser defaults applied).

    :param cfg: Parameters to validate (modified in place).
    :type cfg: argparse.Namespace or GlycoAnalysis
    :param error: Called with a message on invalid input; must not return
        (e.g. ``parser.error`` or a function raising ValueError).
    :type error: callable
    :param bool check_inputs: Also validate input files / directories.
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

    positive = {'distance_cutoff': '--dc', 'area_cutoff': '--ac',
                'radius': '--cr', 'probe': '--pr', 'vdw_scale': '--vdw-scale'}
    for name, flag in positive.items():
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            error(f"{flag} must be positive, got {value}")

    if cfg.vdw_probe is not None and cfg.vdw_probe < 0:
        error(f"--vdw-probe must be non-negative, got {cfg.vdw_probe}")

    try:
        n_workers = int(cfg.n_workers)
    except (TypeError, ValueError):
        error(f"--n-workers must be an integer, got {cfg.n_workers!r}")
    if n_workers != -1 and n_workers < 1:
        error(f"--n-workers must be -1 or >= 1, got {n_workers}")

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

    if cfg.compat and cfg.compat_v2:
        error("--compat and --compat-v2 are mutually exclusive")

    if cfg.compat and cfg.write_blockers:
        error("--write-blockers is not supported with --compat (the box "
              "test does not identify blocking atoms); --write-excluded "
              "is supported")

    if cfg.compat and cfg.cylinder:
        error("--cylinder has no effect with --compat (GLYCO v1 uses the box test)")

    if not cfg.outdir:
        cfg.outdir = default_outdir(cfg)


def default_outdir(cfg):
    """
    Pick an output directory name from the inputs when -o isn't given.

    :param cfg: Run parameters (``indir`` / ``top`` are read).
    :type cfg: argparse.Namespace or GlycoAnalysis
    :return: ``glyco_<indir name>``, ``glyco_<top stem>`` or the default.
    :rtype: str
    """
    if cfg.indir is not None:
        return f"glyco_{Path(cfg.indir).resolve().name}"
    if cfg.top is not None:
        return f"glyco_{Path(cfg.top).stem}"
    return DEFAULT_OUTDIR_NAME


def planned_outputs(outdir, write_vmd=False, write_excluded=False,
                    write_blockers=False):
    """
    Paths of the files a run writes into ``outdir``.

    :param str outdir: Output directory.
    :param bool write_vmd: Include the VMD script.
    :param bool write_excluded: Include excluded_glycan_atoms.csv.
    :param bool write_blockers: Include blockers.csv.
    :return: Output file paths.
    :rtype: list[str]
    """
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
    Prompt before overwriting existing output files unless ``force`` is set.

    Without a TTY the run is refused (unless forced) instead of prompting.

    :param str outdir: Output directory.
    :param bool force: Overwrite without asking.
    :param bool write_vmd: The run writes the VMD script.
    :param bool write_excluded: The run writes excluded_glycan_atoms.csv.
    :param bool write_blockers: The run writes blockers.csv.
    :return: True if the run may proceed.
    :rtype: bool
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
    """
    Load a Universe; several trajectories go through the chain reader.

    :param str top: Topology / structure file.
    :param traj: Trajectory files, or None for a single structure.
    :type traj: list[str] or None
    :return: The loaded Universe.
    :rtype: MDAnalysis.Universe
    """
    if not traj:
        return mda.Universe(top)
    return mda.Universe(top, traj if len(traj) > 1 else traj[0])

# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

class GlycoAnalysis(AnalysisBase):
    """
    Glycan shielding of protein surface residues.

    Counting modes:

    default
        Heavy-atom starts with a per-residue union, float64 pair selection
        (norm < cutoff), van der Waals sphere blocking test: a blocker blocks
        if the closed start->glycan segment touches its sphere of radius
        ``vdw_scale * r_vdW + vdw_probe``. Own residue and blockers whose
        sphere contains the start atom are excluded.
    default + --cylinder
        As default, but with the GLYCO-2 cylinder test in float64 (radius
        --cr, inclusive).
    --compat (v1)
        GLYCO v1: per-residue union, integer cutoff (strict '<'), legacy
        hydrogen filter, blockers from ATOM records only, FreeSASA on the
        atoms the freesasa executable uses, and the original
        integer-truncated +-1 A box blocking test.
    --compat-v2 (v2)
        GLYCO-2: float32 pair and cylinder arithmetic (always the cylinder
        test), start residue CAN block; legacy hydrogen filter, resname
        filter and FreeSASA set.

    Methods taking ``v1`` / ``v2`` / ``cylinder`` / ``legacy`` flags follow
    the run mode when the flags are None. Per-call overrides only change the
    visibility test, not the atom groups or SASA built in :meth:`_prepare`.

    :param u: Universe to analyze.
    :type u: MDAnalysis.Universe
    :param parser: CLI parser; errors go through ``parser.error`` if given.
    :type parser: argparse.ArgumentParser or None
    :param bool verbose: Diagnostic output and progress bars.
    :param args: Parsed arguments (defaults < args < kwargs).
    :type args: argparse.Namespace or None
    :param kwargs: Parameter overrides (any ``base_parser`` destination).
    :raises ValueError: On invalid parameters or selections (library use).
    """

    _analysis_algorithm_is_parallelizable = False

    @classmethod
    def get_supported_backends(cls):
        """
        Backends usable with :meth:`run` (serial only).

        :return: ``('serial',)``
        :rtype: tuple[str]
        """
        return ('serial',)

    def __init__(self, u, parser=None, verbose=False, args=None, **kwargs):
        super().__init__(u.trajectory, verbose=verbose)
        self.u = u
        self.parser = parser
        self.progress_file = os.environ.get("GLYCO_PROGRESS_FILE")
        self._run_slice = {}

        # internal caches (reset again in _prepare)
        self._frame_cache = None
        self._blocker_radii = None
        self._start_residue_keys = None
        self._blocker_residue_keys = None

        # Single source of truth for defaults: base_parser() itself.
        defaults = vars(base_parser().parse_args([]))

        # Only touch sys.argv if a parser was supplied WITHOUT parsed args.
        source = args if args is not None else parser
        self.apply_params(defaults=defaults, args=source, **kwargs)

        self._validate_args()
        if not self.outdir:
            self.outdir = default_outdir(self)
        self._validate_atom_selections()

    # -- mode resolution ----------------------------------------------------

    def _use_v1(self, v1=None):
        """
        Resolve a per-call v1 (GLYCO v1 box test) flag.

        :param v1: Explicit flag, or None for the run mode (--compat).
        :type v1: bool or None
        :rtype: bool
        """
        return bool(self.compat) if v1 is None else bool(v1)

    def _use_v2(self, v2=None):
        """
        Resolve a per-call v2 (GLYCO-2 float32 semantics) flag.

        :param v2: Explicit flag, or None for the run mode (--compat-v2).
        :type v2: bool or None
        :rtype: bool
        """
        return bool(self.compat_v2) if v2 is None else bool(v2)

    def _use_cylinder(self, cylinder=None, v2=False):
        """
        Resolve the cylinder-vs-vdW blocking test.

        GLYCO-2 (v2) is defined by the cylinder test, so v2 always uses it.

        :param cylinder: Explicit flag, or None for the run mode (--cylinder).
        :type cylinder: bool or None
        :param bool v2: The resolved v2 flag.
        :rtype: bool
        :raises ValueError: If ``cylinder=False`` is requested together with v2.
        """
        if v2:
            if cylinder is False:
                raise ValueError("v2 (GLYCO-2) requires the cylinder test")
            return True
        if cylinder is None:
            return bool(self.cylinder)
        return bool(cylinder)

    def _use_legacy(self, legacy=None):
        """
        Resolve a legacy (v1 or v2 shared setup) flag.

        :param legacy: Explicit flag, or None for --compat or --compat-v2.
        :type legacy: bool or None
        :rtype: bool
        """
        if legacy is None:
            return bool(self.compat) or bool(self.compat_v2)
        return bool(legacy)

    def _resolve_mode(self, v1=None, v2=None):
        """
        Resolve the visibility mode.

        With both flags None the run mode is used; otherwise the given flags
        select the mode and an unspecified flag counts as False. The blocking
        test (vdW vs cylinder) is resolved separately by
        :meth:`_use_cylinder`.

        :param v1: GLYCO v1 box test.
        :type v1: bool or None
        :param v2: GLYCO-2 float32 cylinder test.
        :type v2: bool or None
        :return: ``(v1, v2)``
        :rtype: tuple[bool, bool]
        :raises ValueError: If both resolve to True.
        """
        if v1 is None and v2 is None:
            v1, v2 = self._use_v1(), self._use_v2()
        else:
            v1, v2 = bool(v1), bool(v2)
        if v1 and v2:
            raise ValueError("v1 and v2 are mutually exclusive")
        return v1, v2

    def _resolve_test(self, v1=None, v2=None, cylinder=None):
        """
        Fully resolve the visibility test.

        :param v1: See :meth:`_resolve_mode`.
        :type v1: bool or None
        :param v2: See :meth:`_resolve_mode`.
        :type v2: bool or None
        :param cylinder: See :meth:`_use_cylinder`.
        :type cylinder: bool or None
        :return: ``(v1, v2, cylinder)``; cylinder is always False with v1.
        :rtype: tuple[bool, bool, bool]
        :raises ValueError: If v1 and v2 are both requested, or
            ``cylinder=False`` is combined with v2.
        """
        v1, v2 = self._resolve_mode(v1, v2)
        return v1, v2, (False if v1 else self._use_cylinder(cylinder, v2))

    def _blocking_name(self, v1=None, v2=None, cylinder=None):
        """
        Name of the blocking test for diagnostics.

        :param v1: See :meth:`_resolve_test`.
        :type v1: bool or None
        :param v2: See :meth:`_resolve_test`.
        :type v2: bool or None
        :param cylinder: See :meth:`_resolve_test`.
        :type cylinder: bool or None
        :return: ``'box'``, ``'cylinder'`` or ``'vdw'``.
        :rtype: str
        """
        v1, _, cyl = self._resolve_test(v1, v2, cylinder)
        return 'box' if v1 else 'cylinder' if cyl else 'vdw'

    def _mode_name(self):
        """
        Name of the run mode for diagnostics.

        :return: ``'compat'``, ``'compat-v2'``, ``'default-cylinder'`` or
            ``'default-vdw'``.
        :rtype: str
        """
        v1, v2 = self._resolve_mode()
        if v1:
            return 'compat'
        if v2:
            return 'compat-v2'
        return f"default-{self._blocking_name()}"

    def _n_workers(self, processes=False):
        """
        Resolve ``--n-workers``.

        :param bool processes: Resolve for the box-test process pool, which
            needs an explicit count: -1 becomes the number of available CPUs.
            For SciPy KD-tree queries -1 is passed through (all processors).
        :return: Worker count (>= 1, or -1 for KD-tree queries).
        :rtype: int
        """
        n = int(self.n_workers)
        if n == -1 and processes:
            try:
                return max(1, len(os.sched_getaffinity(0)))   # Linux: honours CPU limits
            except AttributeError:
                return max(1, os.cpu_count() or 1)
        return n

    def _kd_query_kwargs(self):
        """
        Keyword arguments shared by every KD-tree ``query_ball_point`` call.

        ``workers`` follows ``--n-workers``. Results are never sorted per
        query, because every caller deduplicates or lexsorts its output.

        :rtype: dict
        """
        return {"workers": self._n_workers(), "return_sorted": False}

    # -- configuration ------------------------------------------------------

    def _error(self, message):
        """
        Report an error: ``parser.error()`` from the CLI, ValueError otherwise.

        :param str message: Error message.
        :raises ValueError: In library use (no parser).
        :raises SystemExit: From ``parser.error`` in CLI use.
        """
        if self.parser is not None:
            self.parser.error(message)
        raise ValueError(message)

    def _diag(self, *parts):
        """
        Diagnostic output, only with -v/--verbose.

        :param parts: Items passed to :func:`print`.
        """
        if self._verbose:
            print(*parts, flush=True)

    def _validate_args(self):
        """
        Validate parameters via :func:`validate_arguments`.

        In library use (a Universe passed in directly, no input files named)
        the input-file checks are skipped.
        """
        library_mode = (self.parser is None and self.top is None
                        and self.indir is None and not self.traj)
        validate_arguments(self, self._error, check_inputs=not library_mode)

    def apply_params(self, args=None, defaults=None, **kwargs):
        """
        Merge ``defaults < args < kwargs`` onto ``self`` as attributes.

        :param args: Parsed arguments, or a parser to parse ``sys.argv`` with.
        :type args: argparse.Namespace or argparse.ArgumentParser or None
        :param defaults: Default values.
        :type defaults: dict or None
        :param kwargs: Highest-priority overrides.
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
        :meth:`AnalysisBase.run`, falling back to the CLI frame slice.

        :param start: First frame.
        :type start: int or None
        :param stop: Stop frame (exclusive).
        :type stop: int or None
        :param step: Frame step.
        :type step: int or None
        :param frames: Explicit frame indices.
        :type frames: list[int] or None
        :param kwargs: Passed to :meth:`AnalysisBase.run` (verbose,
            progressbar_kwargs, ...).
        :return: ``self``
        :rtype: GlycoAnalysis
        """
        if all(v is None for v in (start, stop, step, frames)):
            start, stop, step = self.frame_start, self.frame_stop, self.frame_step
            frames = self.frame_list
        self._run_slice = {"start": start, "stop": stop, "step": step,
                           "frames": frames}
        return super().run(**self._run_slice, **kwargs)

    def _compat_selection(self, sel, blockers=False, legacy=None, v1=None):
        """
        Restrict a selection to the legacy residue names.

        v1 uses ``_V1_RESNAMES`` and (for blockers) ATOM records only, as
        GLYCO v1's ``gen_pro_dict()``; v2 uses ``_V2_RESNAMES`` without a
        record-type restriction. Glycan selections are never passed here.

        :param str sel: Selection string (empty strings are returned as is).
        :param bool blockers: The selection is the blocker selection.
        :param legacy: Apply the filter; None follows the run mode.
        :type legacy: bool or None
        :param v1: GLYCO v1 variant; None follows the run mode.
        :type v1: bool or None
        :return: The (possibly) restricted selection.
        :rtype: str
        """
        if not sel or not self._use_legacy(legacy):
            return sel
        v1 = self._use_v1(v1)
        names = _V1_RESNAMES if v1 else _V2_RESNAMES
        sel = f"({sel}) and (resname {names})"
        if blockers and v1:
            candidate = f"({sel}) and record_type ATOM"
            try:
                self.u.select_atoms(candidate)
                return candidate
            except Exception:  # noqa: BLE001
                warnings.warn(
                    "record_type is unavailable for this topology; "
                    "--compat blockers are NOT restricted to ATOM records",
                    stacklevel=2,
                )
        return sel

    def _validate_atom_selections(self):
        """
        Validate protein, glycan and blocker selections against the Universe.

        Applies :meth:`_compat_selection`, checks that each selection is
        non-empty and parses, that the protein selection has CA atoms, and
        that ``--sasa-env`` parses. Results are stored in
        ``selection_validation_results`` and ``_validated_*_sel``.

        :raises ValueError: On invalid selections (library use).
        """
        prot_sel = self._compat_selection((self.prot_sel or "").strip())
        gly_sel = (self.gly_sel or "").strip()
        block_sel = self._compat_selection((self.block_sel or "").strip(),
                                           blockers=True)

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
        """
        Human-readable summary of the validated selections.

        :rtype: str
        """
        lines = ["Atom selections:"]
        for name, r in self.selection_validation_results.items():
            flag = "ok" if r["valid"] else "INVALID"
            lines.append(f"  {name}: {flag} - {r['message']}")
        return "\n".join(lines)

    # -- FreeSASA setup -------------------------------------------------------

    def _sasa_atoms(self, legacy=None):
        """
        FreeSASA atom set.

        Legacy (without ``--sasa-env``) mirrors the freesasa executable on
        the whole PDB with default options: heavy atoms of ATOM records,
        first altloc only; ``--sasa-hetatm`` adds HETATM. Otherwise: start
        residues + blockers (+ ``--sasa-env``) heavy atoms.

        :param legacy: Use the legacy set; None follows the run mode.
        :type legacy: bool or None
        :rtype: MDAnalysis.core.groups.AtomGroup
        :raises ValueError: If the legacy set cannot be built (library use).
        """
        if self._use_legacy(legacy) and not self.sasa_env:
            sel = "all" if self.sasa_hetatm else "record_type ATOM"
            try:
                return _first_altloc(heavy_atoms(self.u.select_atoms(sel)))
            except Exception as e:  # noqa: BLE001
                self._error("could not build the legacy FreeSASA atom set "
                            f"(needs PDB record types; use --sasa-env): {e}")
        sel = f"({self._validated_prot_sel}) or ({self._validated_block_sel})"
        if self.sasa_env:
            sel += f" or ({self.sasa_env})"
        return heavy_atoms(self.u.select_atoms(sel))

    def _check_sasa_structure(self, fsa, legacy=None):
        """
        Handle a FreeSASA structure that did not match the MDAnalysis residues.

        Legacy runs must reproduce the freesasa executable, so this is an
        error there and a warning otherwise.

        :param fsa: Completed FreeSASA analysis.
        :type fsa: FreesasaAnalysis
        :param legacy: None follows the run mode.
        :type legacy: bool or None
        :raises ValueError: Legacy mode, invalid structure (library use).
        """
        if fsa.structure_valid:
            return
        msg = ("FreeSASA structure did not match the MDAnalysis residues; "
               "SASA used MDAnalysis vdW radii instead of ProtOr, so areas "
               "will not match the freesasa executable")
        if self._use_legacy(legacy):
            self._error(msg)
        else:
            warnings.warn(msg, stacklevel=2)

    def _check_sasa_coverage(self, missing, legacy=None):
        """
        Handle start residues absent from the FreeSASA set.

        Legacy: warn and treat them as buried (e.g. protein residues stored
        as HETATM, which the legacy .rsa lacks). Default: error.

        :param missing: Resindices of the missing start residues.
        :type missing: list[int]
        :param legacy: None follows the run mode.
        :type legacy: bool or None
        :raises ValueError: Default mode with missing residues.
        """
        if not missing:
            return
        if self._use_legacy(legacy):
            warnings.warn(f"{len(missing)} start residue(s) are not in the "
                          "FreeSASA set and are treated as buried (SASA 0): "
                          f"{missing[:10]}", stacklevel=2)
        else:
            raise ValueError("SASA is missing residues required by the "
                             f"protein selection: {missing[:10]}")

    # -- run lifecycle -------------------------------------------------------

    def _prepare(self):
        """
        Build atom groups, run FreeSASA and allocate results (AnalysisBase hook).

        :raises ValueError: On empty SASA sets, unexpected SASA shapes or
            inconsistent CA atoms.
        """
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
        self.cylinder = bool(self.cylinder)
        self._legacy = self._use_legacy()

        # Distance tests:
        #   default     float64 'norm < cutoff' (_cutoff_pairs)
        #   --compat-v2 float32 'norm < cutoff' (_cutoff_pairs)
        #   --compat    int(cutoff), strict '<' in the box workers

        # --ps: START residues (heavy atoms) and their CA output columns
        self.selag = heavy_atoms(self.u.select_atoms(f"({prot_sel})"))
        self.prot_atoms = self.u.select_atoms(f"({prot_sel}) and name CA")

        # Blockers from --bs (not surface-limited); glycans from --gs
        self.blocker_atoms = heavy_atoms(self.u.select_atoms(f"({block_sel})"),
                                         legacy=self._legacy)
        self.gly_atoms = heavy_atoms(self.u.select_atoms(f"({gly_sel})"),
                                     legacy=self._legacy)

        # vdW radii computed up front, so missing-radius warnings appear
        # before the trajectory loop (not needed for the box / cylinder tests)
        self._blocker_radii = (self._blocker_vdw_radii()
                               if self._blocking_name() == 'vdw' else None)

        self._residue_resindices = np.unique(self.selag.resindices)
        self._n_res_rows = len(self._residue_resindices)

        # ---------------- FreeSASA ----------------
        sasa_ag = self._sasa_atoms()
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
        self._check_sasa_structure(fsa)

        areas = np.asarray(fsa.results.residue_areas, dtype=float)
        if areas.shape != (self.n_frames, len(sasa_res)):
            raise ValueError("Unexpected FreeSASA residue-area shape: got "
                             f"{areas.shape}, expected {(self.n_frames, len(sasa_res))}")

        # FreesasaAnalysis columns follow np.unique(ag.resindices)
        column_of = {int(r): i for i, r in enumerate(sasa_res)}
        missing = [int(r) for r in self._residue_resindices if int(r) not in column_of]
        self._check_sasa_coverage(missing)

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

        # Line-of-sight starts: every heavy atom of each --ps residue, with a
        # per-residue union of visible glycan atoms. Legacy modes apply the
        # legacy hydrogen filter on top.
        self.start_atoms = (heavy_atoms(self.selag, legacy=True)
                            if self._legacy else self.selag)
        self._start_res_row = np.searchsorted(self._residue_resindices,
                                              self.start_atoms.resindices)
        self._start_resindex = np.asarray(self.start_atoms.resindices)

        # (resname, chain, resid+icode) keys for the GLYCO v1 box test; built
        # on first use in _box_visibility
        self._start_residue_keys = None
        self._blocker_residue_keys = None

        n_ca = self.prot_atoms.n_atoms
        self.all_counts = np.zeros([self.n_frames, n_ca])

        self.results = Results()
        self.results.residue_counts = self.all_counts
        self.results.n_surface_residues = np.zeros(self.n_frames, dtype=int)
        self.results.n_blocked_pairs = np.zeros(self.n_frames, dtype=int)
        self.results.mean_blocked_distance = np.full(self.n_frames, np.nan)

        # cache for the per-atom CSV writers (after all_counts exists)
        self._frame_cache = None
        self._output_k = self._resolve_frame_index(self.bfactor_frame)

        self._progress_every = max(1, self.n_frames // 100)

        blocking = self._blocking_name()
        geometry = {
            'box': f"box_cutoff={int(self.distance_cutoff)}",
            'cylinder': f"radius={self.radius}",
            'vdw': f"vdw_scale={self.vdw_scale} vdw_probe={self.vdw_probe}",
        }[blocking]
        self._diag("[diagnostic]", f"mode={self._mode_name()}",
                   f"blocking={blocking}", geometry,
                   f"legacy={self._legacy}",
                   f"n_workers={self._n_workers()}",
                   f"start_atoms={len(self.start_atoms)}",
                   f"blocker_atoms={len(self.blocker_atoms)}",
                   f"glycan_atoms={len(self.gly_atoms)}",
                   f"protein_CA={len(self.prot_atoms)}")

    def _single_frame(self):
        """
        Count visible glycan atoms for the current frame (AnalysisBase hook).

        Fills ``all_counts[k]`` and the per-frame summary results. For the
        ``--bfactor-frame`` frame, caches
        ``(k, v1, v2, cylinder, start_idx, cand, vis, blockers, dists)``
        for the CSV writers (see :meth:`_frame_result`).
        """
        k = self._frame_index
        v1, v2, cyl = self._resolve_test()

        start_idx = self.surface_start_indices(k)
        self.results.n_surface_residues[k] = len(np.unique(self._start_res_row[start_idx]))

        cand, vis, n_blocked, mean_d, blockers, dists = self._visibility(
            start_idx, v1, v2, cyl)
        self.all_counts[k] = self._counts_from_visible_keys(vis)
        self.results.n_blocked_pairs[k] = n_blocked
        self.results.mean_blocked_distance[k] = mean_d

        if k == self._output_k:
            self._frame_cache = (k, v1, v2, cyl, start_idx, cand, vis, blockers, dists)

        self._diag("[diagnostic]", f"frame={k}",
                   f"surface_start_atoms={len(start_idx)}",
                   f"candidate_keys={len(cand)}", f"visible_keys={len(vis)}",
                   f"blocker_rows={'n/a' if blockers is None else len(blockers)}",
                   f"nonzero={np.count_nonzero(self.all_counts[k])}",
                   f"total={int(self.all_counts[k].sum())}")

        self.frame_count += 1
        if (self.frame_count % self._progress_every == 0
                or self.frame_count == self.n_frames):
            self._write_progress(self.frame_count, self.n_frames)

    def _conclude(self):
        """Write all requested output files (AnalysisBase hook)."""
        os.makedirs(self.outdir, exist_ok=True)
        self.write_bfactors(frame_index=self.bfactor_frame)
        self.write_frame_summary()
        self.write_residue_csv(frame_index=self.bfactor_frame)

        if self.write_excluded:
            self.write_excluded_glycan_atoms_csv(frame_index=self.bfactor_frame)

        # blockers.csv needs per-atom blockers, which the box test lacks
        if self.write_blockers and not self._use_v1():
            self.write_blockers_csv(frame_index=self.bfactor_frame)

        self._frame_cache = None

        if self.write_vmd:
            self.write_vmd_user_script()

    # -- geometry ------------------------------------------------------------

    def outermost(self, frame_index=None):
        """
        Residue rows that are SURFACE residues (SASA >= area_cutoff).

        :param frame_index: Analyzed-frame index; None uses the current one.
        :type frame_index: int or None
        :rtype: numpy.ndarray of int
        """
        if frame_index is None:
            frame_index = self._frame_index
        return np.flatnonzero(self.residue_areas[frame_index] >= self.area_cutoff)

    def surface_start_indices(self, frame_index=None):
        """
        Indices (into ``start_atoms``) of start atoms in surface residues.

        :param frame_index: Analyzed-frame index; None uses the current one.
        :type frame_index: int or None
        :rtype: numpy.ndarray of int
        """
        surf_res_rows = self.outermost(frame_index)
        return np.flatnonzero(np.isin(self._start_res_row, surf_res_rows))

    def _cutoff_pairs(self, start_idx, v2=None, G=None, gtree=None):
        """
        (start, glycan) pairs with ``norm < cutoff``.

        float64 by default; float32 with v2 (GLYCO-2's ``c_worker``). A
        padded inclusive float64 KD-tree prefilter (``query_ball_point``,
        ``--n-workers`` threads) precedes the exact test in the active dtype.

        :param start_idx: Indices into ``start_atoms``.
        :type start_idx: numpy.ndarray
        :param v2: float32 arithmetic; None follows the run mode.
        :type v2: bool or None
        :param G: Glycan coordinates in the active dtype; None builds them.
        :type G: numpy.ndarray or None
        :param gtree: float64 KD-tree of the glycan atoms; None builds it.
        :type gtree: scipy.spatial.cKDTree or None
        :return: ``(i, g, dist, P, G)``: ``i`` indexes ``start_idx`` (local),
            ``g`` indexes ``gly_atoms``, ``dist`` is the pair distance, and
            ``P`` / ``G`` are the start / glycan coordinates in the active
            dtype.
        :rtype: tuple[numpy.ndarray, ...]
        """
        dtype = np.float32 if self._use_v2(v2) else np.float64
        start_idx = np.asarray(start_idx, dtype=np.intp)
        P = self.start_atoms.positions[start_idx].astype(dtype)
        if G is None:
            G = self.gly_atoms.positions.astype(dtype)
        empty = np.empty(0, dtype=np.intp)
        if len(P) == 0 or len(G) == 0:
            return empty, empty, np.empty(0, dtype=dtype), P, G
        if gtree is None:
            gtree = cKDTree(G.astype(np.float64))

        cutoff = float(self.distance_cutoff)
        neigh = gtree.query_ball_point(P.astype(np.float64), cutoff + _CYLINDER_PAD,
                                       **self._kd_query_kwargs())
        sizes = np.fromiter(map(len, neigh), dtype=np.intp, count=len(neigh))
        if sizes.sum() == 0:
            return empty, empty, np.empty(0, dtype=dtype), P, G
        i = np.repeat(np.arange(len(P), dtype=np.intp), sizes)
        g = np.concatenate(neigh).astype(np.intp, copy=False)
        d = np.linalg.norm(G[g] - P[i], axis=1)                # in `dtype`
        keep = d < dtype(cutoff)
        return i[keep], g[keep], d[keep], P, G

    def _blocker_vdw_radii(self, default=1.7, overrides=None):
        """
        Per-atom van der Waals radii (Å) for ``blocker_atoms``.

        Radii are looked up by element in MDAnalysis' ``vdwradii`` table.
        Missing elements are guessed from atom names with MDAnalysis'
        element guesser; names are never looked up in the table directly
        ('CA' there would be calcium). Unknown elements get ``default``.

        :param float default: Radius for elements missing from the table.
        :param overrides: Extra or replacement ``{element: radius}`` entries.
        :type overrides: dict or None
        :return: One radius per blocker atom.
        :rtype: numpy.ndarray of float64
        """
        table = {str(k).upper(): float(v) for k, v in VDW_RADII.items()}
        if overrides:
            table.update({str(k).upper(): float(v) for k, v in overrides.items()})

        atoms = self.blocker_atoms
        try:
            elements = np.asarray(atoms.elements, dtype=object)
        except (AttributeError, mda.exceptions.NoDataError):
            elements = np.full(len(atoms), "", dtype=object)

        missing = np.array([not str(e).strip() for e in elements], dtype=bool)
        if missing.any():
            guess = DefaultGuesser(self.u).guess_atom_element
            names = atoms.names
            for k in np.flatnonzero(missing):
                elements[k] = guess(names[k])

        radii = np.array([table.get(str(e).strip().upper(), np.nan)
                          for e in elements], dtype=np.float64)
        unknown = np.isnan(radii)
        if unknown.any():
            bad = sorted({str(e) for e in elements[unknown]})
            warnings.warn(f"No vdW radius for element(s) {bad}; using {default} Å",
                          stacklevel=2)
            radii[unknown] = default
        return radii

    def _blocking_context(self, v2, cylinder=None, exclude_own_residue=None,
                          exclude_start_overlap=True, radii=None, scale=None,
                          probe=None):
        """
        Per-frame arrays and KD-trees shared by every start chunk.

        :param bool v2: Resolved GLYCO-2 flag.
        :param cylinder: See :meth:`_use_cylinder`.
        :type cylinder: bool or None
        :param exclude_own_residue: None means ``not v2`` for the cylinder
            test and True for the vdW test.
        :type exclude_own_residue: bool or None
        :param bool exclude_start_overlap: vdW only; see :meth:`_vdw_hits`.
        :param radii: vdW radii per blocker; None uses the cached radii.
        :type radii: numpy.ndarray or None
        :param scale: None uses ``--vdw-scale``.
        :type scale: float or None
        :param probe: None uses ``--vdw-probe``.
        :type probe: float or None
        :rtype: dict
        :raises ValueError: If ``radii`` has the wrong shape, or
            ``cylinder=False`` is combined with v2.
        """
        cyl = self._use_cylinder(cylinder, v2)
        if exclude_own_residue is None:
            exclude_own_residue = (not v2) if cyl else True

        B = np.ascontiguousarray(self.blocker_atoms.positions, dtype=np.float64)
        G = self.gly_atoms.positions.astype(np.float32 if v2 else np.float64)
        G64 = G.astype(np.float64, copy=False)
        ctx = {
            "v2": v2, "cylinder": cyl,
            "exclude_own_residue": bool(exclude_own_residue),
            "exclude_start_overlap": bool(exclude_start_overlap),
            "B": B, "btree": cKDTree(B) if len(B) else None,
            "G": G, "G64": G64, "gtree": cKDTree(G64) if len(G64) else None,
        }
        if not cyl:
            if radii is None:
                if self._blocker_radii is None:
                    self._blocker_radii = self._blocker_vdw_radii()
                radii = self._blocker_radii
            scale = self.vdw_scale if scale is None else scale
            probe = self.vdw_probe if probe is None else probe
            R = float(scale) * np.asarray(radii, dtype=np.float64) + float(probe)
            if R.shape != (len(B),):
                raise ValueError(f"radii must have shape ({len(B)},), got {R.shape}")
            ctx["R2"] = R * R
            ctx["r_max"] = float(R.max()) if len(R) else 0.0
        return ctx

    def _cylinder_hits(self, sub, i, g, dist, P, ctx, chunk=20000):
        """
        Cylinder test (GLYCO-2 geometry, radius ``--cr``, inclusive) on
        precomputed cutoff pairs.

        :param numpy.ndarray sub: Start-atom indices of this chunk.
        :param numpy.ndarray i: Pair start indices (local to ``sub``).
        :param numpy.ndarray g: Pair glycan indices.
        :param numpy.ndarray dist: Pair distances.
        :param numpy.ndarray P: Start coordinates of ``sub`` (active dtype).
        :param dict ctx: Output of :meth:`_blocking_context`.
        :param int chunk: Pairs per batch.
        :return: Lists of ``(n, 3)`` blocker rows and matching float64 distances.
        :rtype: tuple[list, list]
        """
        B, btree, G, G64 = ctx["B"], ctx["btree"], ctx["G"], ctx["G64"]
        r = float(self.radius)
        kw = self._kd_query_kwargs()
        P64 = P.astype(np.float64, copy=False)
        sub_res = self._start_resindex[sub]
        out, dout = [], []
        for s in range(0, len(i), chunk):
            ic, gc, dc = i[s:s + chunk], g[s:s + chunk], dist[s:s + chunk]

            # every point of the cylinder lies within half + r of the midpoint
            mid = 0.5 * (P64[ic] + G64[gc])
            half = 0.5 * np.linalg.norm(P64[ic] - G64[gc], axis=1)
            cand = btree.query_ball_point(mid, half + r + _CYLINDER_PAD, **kw)
            counts = np.fromiter(map(len, cand), dtype=np.intp, count=len(cand))
            if counts.sum() == 0:
                continue
            jj = np.concatenate(cand).astype(np.intp, copy=False)
            rep = np.repeat(np.arange(len(ic)), counts)

            if ctx["exclude_own_residue"]:
                other = self._blocker_resindex[jj] != sub_res[ic[rep]]
                jj, rep = jj[other], rep[other]
                if len(jj) == 0:
                    continue

            i_r, g_r = ic[rep], gc[rep]
            hit = _cylinder_blocked(P, G, B, i_r, g_r, jj, r)   # float32 P/G with v2
            if hit.any():
                out.append(np.stack([sub[i_r[hit]], jj[hit], g_r[hit]], axis=1))
                dout.append(dc[rep][hit].astype(np.float64))
        return out, dout

    def _vdw_hits(self, sub, i, g, dist, P, ctx, chunk=20000):
        """
        Van der Waals sphere test on precomputed cutoff pairs.

        A blocker blocks a pair if the closed segment start -> glycan touches
        its sphere of radius ``vdw_scale * r_vdW + vdw_probe``. The start atom
        itself never blocks. Its residue is ignored if
        ``ctx['exclude_own_residue']``, and blockers whose sphere contains the
        start atom are ignored if ``ctx['exclude_start_overlap']``.
        ``--cr`` is not used.

        Parameters and return value as in :meth:`_cylinder_hits`.
        """
        B, btree, G = ctx["B"], ctx["btree"], ctx["G64"]
        R2, r_max = ctx["R2"], ctx["r_max"]
        kw = self._kd_query_kwargs()
        blocker_ix = self.blocker_atoms.ix              # global atom indices
        P = P.astype(np.float64, copy=False)
        sub_res = self._start_resindex[sub]
        sub_ix = self.start_atoms.ix[sub]
        out, dout = [], []
        for s in range(0, len(i), chunk):
            ic, gc, dc = i[s:s + chunk], g[s:s + chunk], dist[s:s + chunk]
            Pi, Gg = P[ic], G[gc]
            seg = Gg - Pi                                       # P -> G
            mid = 0.5 * (Pi + Gg)
            half = 0.5 * np.linalg.norm(seg, axis=1)

            # Broad phase: any point within R_j of the segment lies within
            # half + R_j <= half + r_max of the midpoint.
            cand = btree.query_ball_point(mid, half + r_max + _CYLINDER_PAD, **kw)
            counts = np.fromiter(map(len, cand), dtype=np.intp, count=len(cand))
            if counts.sum() == 0:
                continue
            jj = np.concatenate(cand).astype(np.intp, copy=False)
            rep = np.repeat(np.arange(len(ic)), counts)

            # the start atom itself never blocks; optionally its residue
            mask = blocker_ix[jj] != sub_ix[ic[rep]]
            if ctx["exclude_own_residue"]:
                mask &= self._blocker_resindex[jj] != sub_res[ic[rep]]
            jj, rep = jj[mask], rep[mask]
            if len(jj) == 0:
                continue

            # exact closed-segment / closed-sphere test
            p, d, c = Pi[rep], seg[rep], B[jj]
            cp = c - p
            seg_len2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
            t = np.clip(np.einsum("ij,ij->i", cp, d) / seg_len2, 0.0, 1.0)
            diff = cp - t[:, None] * d
            hit = np.einsum("ij,ij->i", diff, diff) <= R2[jj]
            if ctx["exclude_start_overlap"]:
                hit &= np.einsum("ij,ij->i", cp, cp) > R2[jj]

            if hit.any():
                out.append(np.stack([sub[ic[rep][hit]], jj[hit], gc[rep][hit]], axis=1))
                dout.append(dc[rep][hit].astype(np.float64))
        return out, dout

    def _iter_blocked_chunks(self, start_idx, ctx, start_chunk=2000, chunk=20000):
        """
        Yield cutoff pairs and their blockers per start chunk.

        The pair search runs exactly once per chunk.

        :param numpy.ndarray start_idx: Indices into ``start_atoms``.
        :param dict ctx: Output of :meth:`_blocking_context`.
        :param int start_chunk: Start atoms per pair search.
        :param int chunk: Pairs per blocking-test batch.
        :return: Generator of ``(start_ids, gly_ids, rows, dists)``: global
            int64 pair indices, ``(n, 3)`` blocker rows of this chunk and
            their float64 pair distances.
        :rtype: collections.abc.Iterator[tuple]
        """
        hits = self._cylinder_hits if ctx["cylinder"] else self._vdw_hits
        no_rows = np.empty((0, 3), dtype=np.intp)
        no_dist = np.empty(0, dtype=np.float64)
        for s0 in range(0, len(start_idx), start_chunk):
            sub = start_idx[s0:s0 + start_chunk]
            i, g, dist, P, _ = self._cutoff_pairs(sub, ctx["v2"],
                                                  G=ctx["G"], gtree=ctx["gtree"])
            if len(i) == 0:
                continue
            rows, dists = no_rows, no_dist
            if ctx["btree"] is not None:
                out, dout = hits(sub, i, g, dist, P, ctx, chunk)
                if out:
                    rows, dists = np.concatenate(out), np.concatenate(dout)
            yield sub[i].astype(np.int64), g.astype(np.int64), rows, dists

    @staticmethod
    def _sort_blockers(row_parts, dist_parts):
        """
        Concatenate per-chunk blockers and lexsort them by (start, blocker, glycan).

        :param list row_parts: ``(n, 3)`` blocker-row arrays.
        :param list dist_parts: Matching distance arrays.
        :return: ``(blockers, distances)``
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        if not row_parts:
            return np.empty((0, 3), dtype=np.intp), np.empty(0, dtype=np.float64)
        res, dres = np.concatenate(row_parts), np.concatenate(dist_parts)
        order = np.lexsort((res[:, 2], res[:, 1], res[:, 0]))
        return res[order], dres[order]

    def get_blockers(self, start_idx=None, v2=None, cylinder=None,
                     exclude_own_residue=None, exclude_start_overlap=True,
                     start_chunk=2000, chunk=20000):
        """
        Blocking protein atoms of every (start atom, glycan atom) cutoff pair.

        Default: van der Waals sphere test (:meth:`_vdw_hits`).
        ``cylinder=True`` or v2: cylinder test (:meth:`_cylinder_hits`).

        :param start_idx: Indices into ``start_atoms``; None uses the surface
            start atoms of the current frame.
        :type start_idx: numpy.ndarray or None
        :param v2: GLYCO-2 float32 semantics; None follows the run mode.
        :type v2: bool or None
        :param cylinder: Cylinder instead of vdW test; None follows --cylinder.
        :type cylinder: bool or None
        :param exclude_own_residue: Ignore blockers in the start atom's
            residue. None means True, except ``not v2`` for the cylinder test.
        :type exclude_own_residue: bool or None
        :param bool exclude_start_overlap: vdW only: ignore blockers whose
            sphere already contains the start atom.
        :param int start_chunk: Start atoms per pair search.
        :param int chunk: Pairs per batch.
        :return: ``(blockers, distances)``: an (n, 3) array of
            ``[start_atoms idx, blocker idx, gly idx]`` sorted lexicographically,
            and the float64 pair distances.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        if start_idx is None:
            start_idx = self.surface_start_indices()
        start_idx = np.asarray(start_idx, dtype=np.intp)
        v2 = self._use_v2(v2)
        if (len(start_idx) == 0 or len(self.blocker_atoms) == 0
                or len(self.gly_atoms) == 0):
            return self._sort_blockers([], [])

        ctx = self._blocking_context(v2, cylinder, exclude_own_residue,
                                     exclude_start_overlap)
        rows, dists = [], []
        for _, _, r, d in self._iter_blocked_chunks(start_idx, ctx, start_chunk, chunk):
            if len(r):
                rows.append(r)
                dists.append(d)
        return self._sort_blockers(rows, dists)

    @staticmethod
    def _residue_keys(atom_group):
        """
        (resname, chain, resid+icode) per atom, as the legacy scripts key residues.

        :param atom_group: Atoms to key.
        :type atom_group: MDAnalysis.core.groups.AtomGroup
        :rtype: list[tuple[str, str, str]]
        """
        try:
            chains = [str(c).strip() for c in atom_group.chainIDs]
        except (AttributeError, mda.exceptions.NoDataError):
            chains = [""] * len(atom_group)
        return [
            (str(resname).strip(), chain_id, str(label))
            for resname, chain_id, label in zip(
                atom_group.resnames, chains, _resid_labels(atom_group))
        ]

    # -- visibility (all modes) ---------------------------------------------

    def _visibility(self, start_idx, v1=None, v2=None, cylinder=None):
        """
        Residue-level visibility for the current trajectory frame.

        In the vdW and cylinder modes the pair search runs once per start
        chunk and feeds both the blocking test and the residue keys.

        :param start_idx: Indices into ``start_atoms``.
        :type start_idx: numpy.ndarray
        :param v1: GLYCO v1 box test; see :meth:`_resolve_mode`.
        :type v1: bool or None
        :param v2: GLYCO-2 float32 cylinder test; see :meth:`_resolve_mode`.
        :type v2: bool or None
        :param cylinder: Cylinder instead of the vdW blocking test; None
            follows ``--cylinder``. Always True with v2, ignored with v1.
        :type cylinder: bool or None
        :return: ``(cand_keys, vis_keys, n_blocked, mean_blocked_distance,
            blockers, distances)``; keys are sorted unique
            ``residue_row * n_gly + glycan_index`` values; blockers and
            distances are None with v1 (the box test does not identify
            blocking atoms).
        :rtype: tuple
        """
        v1, v2 = self._resolve_mode(v1, v2)
        if v1:
            cand, vis, n_blocked, mean_d = self._box_visibility(start_idx)
            return cand, vis, n_blocked, mean_d, None, None

        start_idx = np.asarray(start_idx, dtype=np.intp)
        n_gly = len(self.gly_atoms)
        no_keys = np.empty(0, dtype=np.int64)
        if len(start_idx) == 0 or n_gly == 0:
            blockers, dists = self._sort_blockers([], [])
            return no_keys, no_keys, 0, np.nan, blockers, dists

        ctx = self._blocking_context(v2, cylinder)
        cand_parts, vis_parts, row_parts, dist_parts = [], [], [], []
        for start_ids, gly_ids, rows, d in self._iter_blocked_chunks(start_idx, ctx):
            pair_key = start_ids * n_gly + gly_ids
            res_rows = self._start_res_row[start_ids].astype(np.int64)
            cand_parts.append(np.unique(res_rows * n_gly + gly_ids))
            if len(rows):
                # rows only contain start atoms of this chunk, so a per-chunk
                # blocked set is complete (and smaller than a per-frame one)
                blocked = np.unique(rows[:, 0].astype(np.int64) * n_gly
                                    + rows[:, 2].astype(np.int64))
                keep = ~np.isin(pair_key, blocked)
                res_rows, vis_gly = res_rows[keep], gly_ids[keep]
                row_parts.append(rows)
                dist_parts.append(d)
            else:
                vis_gly = gly_ids
            vis_parts.append(np.unique(res_rows * n_gly + vis_gly))

        cand = np.unique(np.concatenate(cand_parts)) if cand_parts else no_keys
        vis = np.unique(np.concatenate(vis_parts)) if vis_parts else no_keys
        blockers, dists = self._sort_blockers(row_parts, dist_parts)
        n_blocked, mean_d = self._blocked_stats(blockers, dists)
        return cand, vis, n_blocked, mean_d, blockers, dists

    def _blocked_stats(self, blockers, dists):
        """
        Number of unique blocked (start, glycan) pairs and their mean distance.

        Matches the box test's per-pair bookkeeping.

        :param numpy.ndarray blockers: Output of :meth:`get_blockers`.
        :param numpy.ndarray dists: Matching distances.
        :return: ``(n_blocked, mean_distance)``; NaN mean if nothing is blocked.
        :rtype: tuple[int, float]
        """
        if len(blockers) == 0:
            return 0, np.nan
        n_gly = len(self.gly_atoms)
        key = blockers[:, 0].astype(np.int64) * n_gly + blockers[:, 2].astype(np.int64)
        _, first = np.unique(key, return_index=True)
        return len(first), float(dists[first].mean())

    def _box_visibility(self, start_idx=None, workers=None, starts_per_task=32):
        """
        Original GLYCO v1 box test; ``--cr``, ``--cylinder``, ``--vdw-scale``
        and ``--vdw-probe`` do not apply.

        The cutoff is ``int(distance_cutoff)`` with a strict '<', as GLYCO v1.
        The (resname, chain, resid+icode) residue keys are built on first use.

        :param start_idx: Indices into ``start_atoms``; None uses the surface
            start atoms of the current frame.
        :type start_idx: numpy.ndarray or None
        :param workers: Worker processes; 1 runs serially in this process,
            None uses ``--n-workers`` (-1 = all available CPUs).
        :type workers: int or None
        :param int starts_per_task: Start atoms per worker task.
        :return: ``(cand_keys, vis_keys, n_blocked, mean_blocked_distance)``;
            keys are sorted unique ``residue_row * n_gly + glycan_index``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, int, float]
        """
        if start_idx is None:
            start_idx = self.surface_start_indices()
        empty = np.empty(0, dtype=np.int64)
        n_gly = len(self.gly_atoms)
        if len(start_idx) == 0 or n_gly == 0:
            return empty, empty, 0, np.nan

        if self._start_residue_keys is None:
            self._start_residue_keys = self._residue_keys(self.start_atoms)
            self._blocker_residue_keys = self._residue_keys(self.blocker_atoms)

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
            float(int(self.distance_cutoff)),      # v1: int(cutoff), strict '<'
            n_gly,
        )

        n_starts = len(start_idx)
        ranges = [(lo, min(lo + starts_per_task, n_starts))
                  for lo in range(0, n_starts, starts_per_task)]

        if workers is None:
            workers = self._n_workers(processes=True)
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
        """
        Per-CA counts from visible ``residue_row * n_gly + glycan`` keys.

        :param numpy.ndarray vis_keys: Sorted unique visible keys.
        :return: One count per atom of ``prot_atoms``.
        :rtype: numpy.ndarray of int
        """
        counts = np.zeros(len(self.prot_atoms), dtype=np.intp)
        if len(vis_keys):
            n_gly = len(self.gly_atoms)
            per_res = np.bincount(vis_keys // n_gly, minlength=self._n_res_rows)
            counts = per_res[self._prot_atom_res_row].astype(np.intp)
        return counts

    # -- output --------------------------------------------------------------

    @staticmethod
    def _rows_for(residue_resindices, atom_resindices):
        """
        Row (position in ``residue_resindices``) of each atom's residue.

        :param residue_resindices: Residue resindices (any order).
        :type residue_resindices: numpy.ndarray
        :param atom_resindices: Resindex of each atom.
        :type atom_resindices: numpy.ndarray
        :rtype: numpy.ndarray of int
        """
        residue_resindices = np.asarray(residue_resindices)
        order = np.argsort(residue_resindices)
        pos = np.searchsorted(residue_resindices, atom_resindices, sorter=order)
        return order[pos]

    @staticmethod
    def _atom_attr(atom_group, attribute, default=""):
        """
        Whole-group attribute as a string array.

        :param atom_group: Source atoms.
        :type atom_group: MDAnalysis.core.groups.AtomGroup
        :param str attribute: Attribute name (e.g. ``'names'``).
        :param str default: Value used when the attribute is unavailable.
        :rtype: numpy.ndarray of str
        """
        try:
            return np.asarray(getattr(atom_group, attribute)).astype(str)
        except (AttributeError, mda.exceptions.NoDataError):
            return np.full(len(atom_group), default, dtype=object)

    @staticmethod
    def _group_by_residue(keys, n_gly):
        """
        Split sorted unique keys by residue row.

        :param numpy.ndarray keys: ``residue_row * n_gly + glycan`` keys.
        :param int n_gly: Number of glycan atoms.
        :return: ``{residue_row: array_of_glycan_indices}``
        :rtype: dict[int, numpy.ndarray]
        """
        if len(keys) == 0:
            return {}
        rows = keys // n_gly
        gly = keys % n_gly
        uniq, first = np.unique(rows, return_index=True)
        return dict(zip(uniq.tolist(), np.split(gly, first[1:])))

    def _analyzed_frames(self):
        """
        Trajectory frame index of each analyzed frame.

        :rtype: numpy.ndarray of int
        """
        frames = getattr(self, "frames", None)
        if frames is not None and len(frames) == self.n_frames:
            return np.asarray(frames, dtype=int)
        return np.arange(self.n_frames)

    def _resolve_frame_index(self, frame_index=None):
        """
        Normalize an index into the analyzed frames.

        :param frame_index: Index (negative counts from the end); None means
            the last analyzed frame.
        :type frame_index: int or None
        :rtype: int
        :raises IndexError: If the index is out of range.
        """
        n = np.asarray(self.all_counts).shape[0]
        if frame_index is None:
            return n - 1
        k = frame_index + n if frame_index < 0 else frame_index
        if not 0 <= k < n:
            raise IndexError(
                f"frame index {frame_index} out of range for {n} analyzed frame(s)")
        return k

    def _frame_result(self, k, v1=None, v2=None, cylinder=None):
        """
        Visibility of analyzed frame ``k``, from the run cache when possible.

        The cache is filled by :meth:`_single_frame` for ``--bfactor-frame``.
        It is used only if the frame, the mode (v1 / v2) and the blocking
        test (vdW / cylinder) all match. Otherwise the trajectory is moved to
        the frame, visibility is recomputed with :meth:`_visibility`, and the
        previous trajectory position is restored.

        :param int k: Analyzed-frame index, already resolved with
            :meth:`_resolve_frame_index`.
        :param v1: GLYCO v1 box test; see :meth:`_resolve_mode`.
        :type v1: bool or None
        :param v2: GLYCO-2 float32 cylinder test; see :meth:`_resolve_mode`.
        :type v2: bool or None
        :param cylinder: Cylinder instead of the vdW blocking test; None
            follows ``--cylinder``. Always True with v2, ignored with v1.
        :type cylinder: bool or None
        :return: ``(start_idx, cand_keys, vis_keys, blockers, distances)``.
            Keys are sorted unique ``residue_row * n_gly + glycan_index``
            values; blockers and distances are None with v1.
        :rtype: tuple
        :raises ValueError: If v1 and v2 are both requested, or
            ``cylinder=False`` is combined with v2.
        """
        v1, v2, cyl = self._resolve_test(v1, v2, cylinder)
        cache = self._frame_cache
        if cache is not None and cache[:4] == (k, v1, v2, cyl):
            return cache[4:]
        old = int(self.u.trajectory.frame)
        try:
            self.u.trajectory[int(self._analyzed_frames()[k])]
            start_idx = self.surface_start_indices(k)
            cand, vis, _, _, blockers, dists = self._visibility(start_idx, v1, v2, cyl)
        finally:
            self.u.trajectory[old]
        return start_idx, cand, vis, blockers, dists

    def _frame_visibility(self, k, v1=None, v2=None, cylinder=None):
        """
        Candidate and visible residue keys of analyzed frame ``k``.

        ``cand_keys`` are glycan atoms within the cutoff of at least one start
        atom of the residue. ``vis_keys`` are those visible from at least one
        of them under the selected blocking test.

        :param int k: Analyzed-frame index (already resolved).
        :param v1: See :meth:`_resolve_mode`.
        :type v1: bool or None
        :param v2: See :meth:`_resolve_mode`.
        :type v2: bool or None
        :param cylinder: See :meth:`_frame_result`.
        :type cylinder: bool or None
        :return: ``(start_idx, cand_keys, vis_keys)``
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        start_idx, cand, vis, _, _ = self._frame_result(k, v1, v2, cylinder)
        return start_idx, cand, vis

    def _frame_blockers(self, k, v2=None, cylinder=None):
        """
        Per-atom blockers of analyzed frame ``k`` (vdW or cylinder test).

        Never uses the box test, which does not identify blocking atoms, so
        this also works in a --compat run with the atom groups of that run
        (legacy filters applied).

        :param int k: Analyzed-frame index (already resolved).
        :param v2: GLYCO-2 float32 cylinder test; None follows the run mode.
        :type v2: bool or None
        :param cylinder: See :meth:`_frame_result`.
        :type cylinder: bool or None
        :return: ``(start_idx, blockers, distances)``: blockers is an (n, 3)
            array of ``[start_atoms idx, blocker idx, gly idx]``; distances
            are the float64 start–glycan pair distances.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        v2 = self._use_v2(v2)
        start_idx, _, _, blockers, dists = self._frame_result(k, False, v2, cylinder)
        return start_idx, blockers, dists

    def _ensure_tempfactors(self):
        """Add a ``tempfactors`` topology attribute if the Universe lacks one."""
        try:
            self.u.atoms.tempfactors  # noqa: B018
        except (AttributeError, mda.exceptions.NoDataError):
            self.u.add_TopologyAttr('tempfactors')

    def write_blockers_csv(self, frame_index=None, output=None, chunk=100_000,
                           v1=None, v2=None, cylinder=None):
        """
        One row per (start atom, blocking atom, glycan atom) for one frame.

        :param frame_index: Analyzed-frame index; None means the last.
        :type frame_index: int or None
        :param output: Output path; default ``<outdir>/blockers.csv``.
        :type output: str or pathlib.Path or None
        :param int chunk: Rows written per batch.
        :param v1: See :meth:`_resolve_mode`.
        :type v1: bool or None
        :param v2: See :meth:`_resolve_mode`.
        :type v2: bool or None
        :param cylinder: Cylinder instead of the vdW blocking test; see
            :meth:`_frame_result`.
        :type cylinder: bool or None
        :return: Path of the written file.
        :rtype: pathlib.Path
        :raises ValueError: With v1 (the box test has no blocking atoms).
        """
        v1, v2 = self._resolve_mode(v1, v2)
        if v1:
            raise ValueError("blockers.csv is not available with --compat: the box "
                             "test does not identify blocking atoms")

        if output is None:
            output = os.path.join(self.outdir, "blockers.csv")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        k = self._resolve_frame_index(frame_index)
        trajectory_frame = int(self._analyzed_frames()[k])

        _, blockers, distances = self._frame_blockers(k, v2, cylinder)
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
        """
        Write b_factors.pdb (whole structure, one frame) and b_factors.npy (all frames).

        Leaves the trajectory at the written frame.

        :param frame_index: Analyzed-frame index; None means the last.
        :type frame_index: int or None
        """
        os.makedirs(self.outdir, exist_ok=True)
        self._ensure_tempfactors()

        k = self._resolve_frame_index(frame_index)
        self.u.trajectory[int(self._analyzed_frames()[k])]

        self._populate_b_factors(k)
        self.u.atoms.write(os.path.join(self.outdir, 'b_factors.pdb'))
        np.save(os.path.join(self.outdir, 'b_factors'), np.asarray(self.all_counts))

    def _populate_b_factors(self, frame_index=None):
        """
        Set tempfactors: every atom of a CA residue gets that residue's value.

        All other atoms get 0.

        :param frame_index: Analyzed-frame index (for 2D counts).
        :type frame_index: int or None
        :raises ValueError: If the counts do not match ``prot_atoms``.
        """
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
        """Write frame_summary.csv with one row per analyzed frame."""
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
        Per-residue glycan density and SASA.

        Columns: Chain, ResNum, ResName, Glycan_density, SASA. ResNum
        includes the insertion code. Residues with a zero count are omitted
        in every mode.

        :param frame_index: Analyzed-frame index; None means the last.
        :type frame_index: int or None
        :param output: Output path; default ``<outdir>/residue_summary.csv``.
        :type output: str or pathlib.Path or None
        :return: Path of the written file.
        :rtype: pathlib.Path
        :raises ValueError: If residue attributes have inconsistent lengths.
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

        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Chain", "ResNum", "ResName", "Glycan_density", "SASA"])
            n_written = 0
            for chain_id, resnum, resname, density, sasa in zip(
                    chains, resnums, resnames, glycan_density, sasa_values):
                density = int(density)
                if density == 0:
                    continue
                writer.writerow([str(chain_id).strip(), str(resnum),
                                 str(resname).strip(), density,
                                 f"{float(sasa):.6g}"])
                n_written += 1

        print(f"Residue summary written to: {output} ({n_written} residues)")
        return output

    def _write_progress(self, current, total):
        """
        Write machine-readable progress JSON for the Streamlit app.

        Does nothing unless ``GLYCO_PROGRESS_FILE`` is set; written
        atomically via a temporary file.

        :param int current: Frames done.
        :param int total: Total frames.
        """
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
        Quote a string as one Tcl word.

        :param value: Value to quote (converted with :func:`str`).
        :rtype: str
        """
        value = str(value)
        if "{" not in value and "}" not in value and not value.endswith("\\"):
            return "{" + value + "}"
        return '"' + re.sub(r'([\\"$\[\]])', r'\\\1', value) + '"'

    @staticmethod
    def _index_ranges(sorted_indices):
        """
        Compress sorted ints into VMD ``'a to b'`` ranges.

        :param numpy.ndarray sorted_indices: Sorted atom indices.
        :rtype: str
        """
        if len(sorted_indices) == 0:
            return ""
        breaks = np.flatnonzero(np.diff(sorted_indices) != 1) + 1
        parts = []
        for run in np.split(sorted_indices, breaks):
            parts.append(str(run[0]) if len(run) == 1 else f"{run[0]} to {run[-1]}")
        return " ".join(parts)

    def write_vmd_user_script(self, output=None, frame_offset=0):
        """
        VMD Tcl script assigning per-residue counts to ``user`` for every analyzed frame.

        :param output: Output path; default ``<outdir>/load_user_attributes.vmd``.
        :type output: str or pathlib.Path or None
        :param int frame_offset: Added to each trajectory frame index.
        :return: Path of the written file.
        :rtype: pathlib.Path
        :raises ValueError: If topology / trajectory names cannot be
            determined or the data are inconsistent or non-finite.
        :raises FileNotFoundError: If a topology or trajectory file is missing.
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

    def write_excluded_glycan_atoms_csv(self, frame_index=None, output=None,
                                        v1=None, v2=None, cylinder=None):
        """
        Glycan atoms missing from each surface residue's visibility union.

        One row per excluded glycan atom, for every SURFACE residue whose
        count in the run is nonzero. ``reason`` is:

        * ``outside_distance_cutoff``: no start atom of the residue is within
          the distance cutoff of the glycan atom;
        * ``blocked_for_all_start_atoms``: at least one start atom is within
          the cutoff, but the line of sight is blocked for every one of them.

        ``glycan_atom_index`` is the PDB serial number.

        The visibility test follows the run mode unless ``v1``, ``v2`` or
        ``cylinder`` override it. ``final_glycan_count`` always comes from the
        run (``all_counts``). With an override, the recomputed union can
        differ from it, and a warning is issued for each residue where it
        does. Overrides change only the visibility test, not the atom groups
        or SASA built in :meth:`_prepare`.

        :param frame_index: Analyzed-frame index; None means the last.
        :type frame_index: int or None
        :param output: Output path; default
            ``<outdir>/excluded_glycan_atoms.csv``.
        :type output: str or pathlib.Path or None
        :param v1: GLYCO v1 box test; see :meth:`_resolve_mode`.
        :type v1: bool or None
        :param v2: GLYCO-2 float32 cylinder test; see :meth:`_resolve_mode`.
        :type v2: bool or None
        :param cylinder: Cylinder instead of the vdW blocking test; see
            :meth:`_frame_result`.
        :type cylinder: bool or None
        :return: Path of the written file.
        :rtype: pathlib.Path
        :raises ValueError: If v1 and v2 are both requested, or
            ``cylinder=False`` is combined with v2.
        """
        if output is None:
            output = os.path.join(self.outdir, "excluded_glycan_atoms.csv")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        k = self._resolve_frame_index(frame_index)
        trajectory_frame = int(self._analyzed_frames()[k])
        n_gly = len(self.gly_atoms)
        empty = np.empty(0, dtype=np.int64)

        start_idx, cand_keys, vis_keys = self._frame_visibility(k, v1, v2, cylinder)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_job(args, parser, top, traj, outdir, verbose):
    """
    Load one structure/trajectory, run the analysis and write its outputs.

    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :param parser: Parser for error reporting, or None to raise ValueError.
    :type parser: argparse.ArgumentParser or None
    :param str top: Topology / structure file.
    :param traj: Trajectory files.
    :type traj: list[str] or None
    :param str outdir: Output directory for this job.
    :param bool verbose: Progress bars and diagnostics.
    """
    u = load_universe(top, traj)
    analysis = GlycoAnalysis(u, parser=parser, args=args,
                             top=top, traj=traj, indir=None, outdir=outdir)
    analysis.run(verbose=verbose,
                 progressbar_kwargs={'desc': 'Running GlyCo Analysis'})


def _structure_outdirs(structures, base_outdir):
    """
    One output subfolder per structure, disambiguating clashing stems.

    :param structures: Structure file paths.
    :type structures: list[str]
    :param str base_outdir: Parent output directory.
    :return: One directory per structure, in input order.
    :rtype: list[str]
    """
    stems = [Path(p).stem for p in structures]
    dirs = []
    for path, stem in zip(structures, stems):
        name = stem if stems.count(stem) == 1 else Path(path).name.replace('.', '_')
        dirs.append(os.path.join(base_outdir, name))
    return dirs


def main(argv=None):
    """
    Command-line entry point.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :type argv: list[str] or None
    :return: Exit status (0 on success, 1 on failure or declined overwrite).
    :rtype: int
    """
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