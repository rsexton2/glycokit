import functools
import glob
import os
import platform
import sys
import warnings
from importlib import metadata
from itertools import chain
from pathlib import Path
from types import MappingProxyType

import MDAnalysis as mda
import numpy as np
import scipy

from glycokit.config import (
    _BLOCKING_TESTS,
    _BOX_PREFILTER_EPS,
    _HYDROGEN_NAME,
    DEFAULT_OUTDIR_NAME,
    OUTPUT_FILES,
)
from glycokit.console import error, info, style
from glycokit.parsers import base_parser

# Per-process state of the GLYCO v1 box workers
_BOX_STATE = {}

# ============= Helpers =============


def _fmt_float(value):
    """Format a float for the CSV outputs (``nan`` stays ``nan``)."""
    return f"{float(value):.6g}"


def _package_versions():
    """Versions recorded in run_config.json."""
    try:
        glycokit_version = metadata.version("glycokit")
    except metadata.PackageNotFoundError:
        glycokit_version = None
    return {
        "python": platform.python_version(),
        "glycokit": glycokit_version,
        "MDAnalysis": mda.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _resid_labels(ag):
    """
    PDB-style residue numbers including the insertion code ('52', '52A').

    :param ag: Atoms to label.
    :type ag: MDAnalysis.core.groups.AtomGroup
    :rtype: numpy.ndarray (dtype object)
    """
    resids = np.asarray(ag.resids)
    try:
        icodes = [str(i).strip() for i in ag.icodes]
    except (AttributeError, mda.exceptions.NoDataError):
        icodes = [""] * len(resids)
    return np.asarray([f"{int(r)}{i}" for r, i in zip(resids, icodes)], dtype=object)


def _cylinder_blocked(P, G, B, i, g, j, r):
    """
    GLYCO-2 ``c_count_points`` cylinder test.

    True where blocker ``B[j]`` lies inside the cylinder of radius ``r``
    around the segment ``G[g] -> P[i]``. End caps are strict; the radius is
    inclusive. float32 ``P``/``G`` reproduce GLYCO-2, float64 give the
    default test.

    :param numpy.ndarray P: Start-atom coordinates, shape (n, 3).
    :param numpy.ndarray G: Glycan-atom coordinates, shape (m, 3).
    :param numpy.ndarray B: Blocker coordinates, shape (k, 3).
    :param numpy.ndarray i: Row indices into ``P``.
    :param numpy.ndarray g: Row indices into ``G``.
    :param numpy.ndarray j: Row indices into ``B``.
    :param float r: Cylinder radius in Å.
    :return: Boolean mask, one entry per (i, g, j) triple.
    :rtype: numpy.ndarray
    """
    vec = P[i] - G[g]  # pt2 - pt1
    const = np.asarray(r, dtype=vec.dtype) * np.linalg.norm(vec, axis=1)
    q_pt1 = B[j] - G[g]
    q_pt2 = B[j] - P[i]
    return (
        (np.einsum("ij,ij->i", q_pt1, vec) > 0)
        & (np.einsum("ij,ij->i", q_pt2, vec) < 0)
        & (np.linalg.norm(np.cross(q_pt1, vec), axis=1) <= const)
    )


def _sphere_blocked(P, G, B, i, g, j, R2, exclude_start_overlap=True):
    """
    Van der Waals sphere test.

    True where the closed segment ``P[i] -> G[g]`` touches the sphere of
    squared radius ``R2`` around ``B[j]``. With ``exclude_start_overlap``,
    blockers whose sphere already contains the start atom never block.

    :param numpy.ndarray P: Start-atom coordinates, shape (n, 3).
    :param numpy.ndarray G: Glycan-atom coordinates, shape (m, 3).
    :param numpy.ndarray B: Blocker coordinates, shape (k, 3).
    :param numpy.ndarray i: Row indices into ``P``.
    :param numpy.ndarray g: Row indices into ``G``.
    :param numpy.ndarray j: Row indices into ``B``.
    :param numpy.ndarray R2: Squared effective radius per (i, g, j) triple.
    :param bool exclude_start_overlap: Ignore blockers overlapping the start atom.
    :return: Boolean mask, one entry per triple.
    :rtype: numpy.ndarray
    """
    p = P[i]
    d = G[g] - p  # start -> glycan
    cp = B[j] - p
    seg_len2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
    t = np.clip(np.einsum("ij,ij->i", cp, d) / seg_len2, 0.0, 1.0)
    diff = cp - t[:, None] * d
    hit = np.einsum("ij,ij->i", diff, diff) <= R2
    if exclude_start_overlap:
        hit &= np.einsum("ij,ij->i", cp, cp) > R2
    return hit


def _first_altloc(ag):
    """
    Keep blank altlocs plus the first altloc letter seen in each residue.

    :param ag: Atoms to filter.
    :type ag: MDAnalysis.core.groups.AtomGroup
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
            keep[k] = a == first[res]
    return ag[keep]


def _legacy_coords(positions):
    """
    float64 coordinates as GLYCO v1 parses them (``float(line[30:38])``).

    :param numpy.ndarray positions: Coordinates, shape (n, 3).
    :rtype: numpy.ndarray
    """
    return np.round(np.asarray(positions, dtype=np.float64), 3)


def _build_x_buckets(positions):
    """
    Reproduce glyco_mod.py's integer-x index (``gen_pro_dict``'s outdict2).

    :param numpy.ndarray positions: Coordinates, shape (n, 3).
    :return: Mapping ``int(x) -> tuple of row indices``.
    :rtype: dict[int, tuple[int, ...]]
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
    Initialize one box-test process with plain NumPy/Python data.

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

    :param tuple start_range: Half-open range ``(lo, hi)``.
    :return: (candidate keys, visible keys, number of blocked pairs, sum of
        blocked pair distances); keys are ``residue_row * n_gly + glycan``.
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
            if distance >= cutoff:  # GLYCO: distance < cutoff
                continue

            key = row * n_gly + gi
            candidate_keys.add(key)
            if distance == 0.0:  # check_protein(): visible
                visible_keys.add(key)
                continue

            blocked = False
            for t in np.arange(0.0, 1.0, 1.0 / distance):
                point = p + t * difference
                x, y, z = int(point[0]), int(point[1]), int(point[2])
                for bi in chain(
                    buckets.get(x, ()), buckets.get(x - 1, ()), buckets.get(x + 1, ())
                ):
                    bx, by, bz = B[bi]
                    if (
                        abs(y - by) <= 1.0
                        and abs(z - bz) <= 1.0
                        and abs(x - bx) <= 1.0
                        and bkeys[bi] != own
                    ):
                        blocked = True
                        break
                if blocked:
                    break

            if blocked:
                n_blocked += 1
                blocked_distance_sum += distance
            else:
                visible_keys.add(key)

    return (
        np.asarray(sorted(candidate_keys), dtype=np.int64),
        np.asarray(sorted(visible_keys), dtype=np.int64),
        n_blocked,
        blocked_distance_sum,
    )


def pdb_atom_ids(atom_group):
    """
    PDB-style (1-based) serial numbers for every atom in an AtomGroup.

    :param atom_group: Atoms to number.
    :type atom_group: MDAnalysis.core.groups.AtomGroup
    :rtype: numpy.ndarray of int64
    """
    try:
        return np.asarray(atom_group.ids, dtype=np.int64)
    except (AttributeError, mda.exceptions.NoDataError):
        return np.asarray(atom_group.indices, dtype=np.int64) + 1


def heavy_atoms(ag, legacy=False):
    """
    Return ``ag`` without hydrogens.

    Default: element H/D, or a name starting with H after optional digits.
    ``legacy=True``: the original GLYCO filter, dropping every atom with an
    'H' anywhere in its name (also NH1, OH, CH2, ...).

    :param ag: Atoms to filter.
    :type ag: MDAnalysis.core.groups.AtomGroup
    :param bool legacy: Use the original GLYCO name filter.
    :rtype: MDAnalysis.core.groups.AtomGroup
    """
    names = np.asarray(ag.names, dtype=str)
    if legacy:
        is_h = np.fromiter(("H" in n for n in names), dtype=bool, count=len(names))
        return ag[~is_h]
    is_h = np.fromiter(
        (bool(_HYDROGEN_NAME.match(n)) for n in names), dtype=bool, count=len(names)
    )
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
    :param list[str] extensions: Extensions without the leading dot.
    :rtype: list[str]
    """
    files = []
    for ext in extensions:
        files.extend(sorted(glob.glob(os.path.join(indir, f"*.{ext}"))))
    return list(dict.fromkeys(files))


@functools.lru_cache(maxsize=1)
def _parser_defaults():
    """
    Defaults of every :func:`base_parser` destination, built once per process.

    Read-only, because the cached mapping is shared by every caller. Call
    ``_parser_defaults.cache_clear()`` after changing :func:`base_parser`
    at runtime (e.g. in tests).

    :rtype: types.MappingProxyType
    """
    return MappingProxyType(vars(base_parser().parse_args([])))


def _warn_ignored_options(cfg, blocking):
    """Warn about options that have no effect in the chosen mode."""
    defaults = _parser_defaults()

    def changed(name):
        return getattr(cfg, name) != defaults[name]

    ignored = []
    if blocking != "vdw":
        ignored += [
            flag
            for name, flag in (
                ("vdw_scale", "--vdw-scale"),
                ("vdw_probe", "--vdw-probe"),
            )
            if changed(name)
        ]
    if blocking != "cylinder" and changed("cylinder_radius"):
        ignored.append("--cr")
    if cfg.mode == "default" and cfg.sasa_hetatm:
        ignored.append("--sasa-hetatm")
    if ignored:
        warnings.warn(
            f"{', '.join(ignored)}: no effect with --mode {cfg.mode} "
            f"--blocking {blocking}",
            stacklevel=3,
        )


def _validate_inputs(cfg, fail):
    """
    Validate -a/--top, -b/--traj and --indir, and set ``cfg.structures``.

    :param cfg: Parameters to validate (modified in place).
    :type cfg: argparse.Namespace or GlycoAnalysis
    :param callable fail: Called with a message on invalid input; must not
        return.
    """
    top_given = cfg.top is not None
    traj_given = bool(cfg.traj)
    indir_given = cfg.indir is not None

    if not top_given and not indir_given:
        fail("one of -a/--top(+-b/--traj) or --indir must be given")
    if indir_given and (top_given or traj_given):
        fail("--indir is mutually exclusive with -a/--top and -b/--traj")
    if traj_given and not top_given:
        fail("-b/--traj requires -a/--top")

    if top_given:
        if not os.path.isfile(cfg.top):
            fail(f"-a/--top: no such file: {cfg.top!r}")
        for t in cfg.traj or []:
            if not os.path.isfile(t):
                fail(f"-b/--traj: no such file: {t!r}")
        return

    if not os.path.isdir(cfg.indir):
        fail(f"--indir: no such directory: {cfg.indir!r}")
    extensions = [e.strip().lstrip(".") for e in cfg.in_types.split(",") if e.strip()]
    structures = find_structures(cfg.indir, extensions)
    if not structures:
        fail(f"--indir: no files with extension(s) {extensions} found in {cfg.indir!r}")
    cfg.structures = structures


def _validate_mode(cfg, fail, warn=True):
    """
    Validate --mode and --blocking and return the resolved blocking test.

    :param cfg: Parameters to validate.
    :type cfg: argparse.Namespace or GlycoAnalysis
    :param callable fail: Called with a message on invalid input; must not
        return.
    :param bool warn: Warn about legacy modes on non-PDB inputs.
    :return: The blocking test (``--blocking``, or the mode's default).
    :rtype: str
    """
    if cfg.mode not in _BLOCKING_TESTS:
        fail(f"--mode must be one of {sorted(_BLOCKING_TESTS)}, got {cfg.mode!r}")
    allowed = _BLOCKING_TESTS[cfg.mode]
    if cfg.blocking is not None and cfg.blocking not in allowed:
        fail(
            f"--blocking {cfg.blocking} is not available with --mode {cfg.mode} "
            f"(allowed: {', '.join(allowed)})"
        )
    blocking = cfg.blocking or allowed[0]
    if blocking == "box" and cfg.write_blockers:
        fail(
            "--write-blockers is not supported with --mode v1 (the box test "
            "does not identify blocking atoms); --write-excluded is supported"
        )

    if cfg.mode != "default" and warn:
        files = [cfg.top] if cfg.top is not None else list(cfg.structures)
        non_pdb = [f for f in files if Path(f).suffix.lower() not in (".pdb", ".ent")]
        if non_pdb:
            warnings.warn(
                f"--mode {cfg.mode} needs PDB record types; may fail for {non_pdb[:3]}",
                stacklevel=3,
            )
    return blocking


def _empty_forward_slice(start, stop, step):
    """True for a non-negative forward --start/--stop/--step that selects no frames."""
    if start is None or stop is None or start < 0 or stop < 0:
        return False
    return (step is None or step > 0) and start >= stop


def _validate_frames(cfg, fail):
    """
    Validate the frame selection and normalize ``cfg.frame_list``.

    :param cfg: Parameters to validate (modified in place).
    :type cfg: argparse.Namespace or GlycoAnalysis
    :param callable fail: Called with a message on invalid input; must not
        return.
    """
    start, stop, step = cfg.frame_start, cfg.frame_stop, cfg.frame_step
    slice_given = any(v is not None for v in (start, stop, step))
    frames_given = bool(cfg.frame_list)

    if frames_given and slice_given:
        fail("--frames is not allowed with --start/--stop/--step")
    if cfg.indir is not None and (frames_given or slice_given):
        fail(
            "--frames/--start/--stop/--step are not meaningful with --indir "
            "(each structure is a single, independent frame)"
        )
    if step is not None and step == 0:
        fail("--step cannot be 0")
    if _empty_forward_slice(start, stop, step):
        fail(f"--start ({start}) must be less than --stop ({stop})")

    if not frames_given:
        cfg.frame_list = None
        return
    try:
        frame_ints = [int(f) for f in cfg.frame_list]
    except (TypeError, ValueError) as exc:
        fail(f"--frames must be integers, got {cfg.frame_list!r}")
        raise AssertionError("fail() must not return") from exc  # unreachable
    if any(f < 0 for f in frame_ints):
        fail("--frames must be non-negative")
    cfg.frame_list = frame_ints


def _validate_numbers(cfg, fail):
    """
    Validate the numeric parameters (cutoffs, radii, probe, worker count).

    ``not value > 0`` is deliberate: unlike ``value <= 0`` it also rejects NaN.

    :param cfg: Parameters to validate.
    :type cfg: argparse.Namespace or GlycoAnalysis
    :param callable fail: Called with a message on invalid input; must not
        return.
    """
    positive = {
        "distance_cutoff": "--dc",
        "area_cutoff": "--ac",
        "cylinder_radius": "--cr",
        "sasa_probe": "--pr",
        "vdw_scale": "--vdw-scale",
    }
    for name, flag in positive.items():
        value = getattr(cfg, name)
        if value is not None and not value > 0:  # pylint: disable=unnecessary-negation
            fail(f"{flag} must be positive, got {value}")

    probe = cfg.vdw_probe
    if probe is not None and not probe >= 0:  # pylint: disable=unnecessary-negation
        fail(f"--vdw-probe must be non-negative, got {probe}")

    n = cfg.n_workers
    if isinstance(n, bool) or not isinstance(n, int) or (n != -1 and n < 1):
        fail(f"--n-workers must be -1 or an integer >= 1, got {n!r}")


def resolve_glycan_selection(universe, gly_sel, fail):
    """
    ``gly_sel``, or the selection from the standardize glycan guesser if empty.

    :param callable fail: Called with a message if nothing is detected.
    :rtype: str
    """
    if gly_sel and gly_sel.strip():
        return gly_sel
    # standardize imports parsers; imported here to keep utils import-light
    # pylint: disable-next=import-outside-toplevel
    from glycokit.standardize import glycan_selection, guess_glycans

    names, _ = guess_glycans(universe)
    selection = glycan_selection(names)
    if not selection:
        fail("--gs not given and no glycan residues were detected; pass --gs")
    info(f"--gs not given: using {style(selection, 'cyan')} (glycan guesser)")
    return selection


def validate_arguments(cfg, fail, check_inputs=True, warn=True):
    """
    Validate and normalize run parameters.

    Sets ``cfg.structures``, normalizes ``cfg.frame_list`` and fills
    ``cfg.outdir`` if empty. ``cfg`` must carry every :func:`base_parser`
    destination.

    :param cfg: Parameters to validate (modified in place).
    :type cfg: argparse.Namespace or GlycoAnalysis
    :param callable fail: Called with a message on invalid input; must not
        return.
    :param bool check_inputs: Also validate input files / directories.
    :param bool warn: Warn about options without effect and non-PDB inputs
        in legacy modes.
    """
    cfg.structures = []
    if check_inputs:
        _validate_inputs(cfg, fail)

    blocking = _validate_mode(cfg, fail, warn)
    _validate_frames(cfg, fail)
    _validate_numbers(cfg, fail)

    if warn:
        _warn_ignored_options(cfg, blocking)
    if not cfg.outdir:
        cfg.outdir = default_outdir(cfg)


def _ask_yes_no(question):
    """Prompt until the answer is yes or no; EOF or empty means no."""
    while True:
        try:
            answer = input(question).strip().lower()
        except EOFError:
            return False
        if answer in ("y", "yes", "", "n", "no"):
            return answer in ("y", "yes")
        print("Please answer 'y' or 'n'.")


def default_outdir(cfg):
    """
    Output directory name from the inputs when -o isn't given.

    :rtype: str
    """
    if cfg.indir is not None:
        return f"glyco_{Path(cfg.indir).resolve().name}"
    if cfg.top is not None:
        return f"glyco_{Path(cfg.top).stem}"
    return DEFAULT_OUTDIR_NAME


def planned_outputs(outdir, cfg):
    """Paths of the files a run with parameters ``cfg`` writes into ``outdir``."""
    return [
        os.path.join(outdir, name)
        for name, flag in OUTPUT_FILES.items()
        if flag is None or getattr(cfg, flag)
    ]


def confirm_overwrite(outdir, cfg):
    """
    Prompt before overwriting existing output files unless ``cfg.force`` is set.

    A directory that does not exist yet is never prompted for. Without a TTY
    the run is refused (unless forced).

    :param str outdir: Output directory.
    :param cfg: Run parameters (``force`` and the ``write_*`` flags are read).
    :type cfg: argparse.Namespace or GlycoAnalysis
    :return: True if the run may proceed.
    :rtype: bool
    """
    if not os.path.isdir(outdir):
        return True
    existing = [
        os.path.basename(p) for p in planned_outputs(outdir, cfg) if os.path.exists(p)
    ]
    if not existing:
        return True

    listing = ", ".join(existing)
    if cfg.force:
        info(f"overwriting in {style(outdir, 'cyan')}: {listing} (--force given)")
        return True
    if not sys.stdin.isatty():
        error(
            f"{outdir!r} already contains: {listing}; no TTY is available to "
            "prompt, pass --force to overwrite non-interactively"
        )
        return False
    return _ask_yes_no(f"{outdir!r} already contains: {listing}. Overwrite? [y/N] ")


def load_universe(top, traj=None):
    """
    Load a Universe; several trajectories go through the chain reader.

    :rtype: MDAnalysis.Universe
    """
    if not traj:
        return mda.Universe(top)
    return mda.Universe(top, traj if len(traj) > 1 else traj[0])
