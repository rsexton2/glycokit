"""
Unit and regression tests for the glycokit package.

Inputs come from conftest.py options, e.g.
    pytest --glyco-structure=a.pdb
    pytest --glyco-structure=a.pdb --glyco-modes=default,v2
    pytest -n auto --dist loadgroup --glyco-order slow-first
"""

import csv
import hashlib
import itertools
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from glycokit.tests.glyco_cases import MODE_SPECS
from glycokit.tests.sasa_probe import capture

EXPECTED_FILES = (
    "b_factors.pdb",
    "b_factors.npy",
    "frame_summary.csv",
    "residue_summary.csv",
    "excluded_glycan_atoms.csv",
    "run_config.json",
)
EXCLUSION_REASONS = {"outside_distance_cutoff", "blocked_for_all_start_atoms"}
COMPARED_CSVS = ("residue_summary.csv", "excluded_glycan_atoms.csv")

# blocking test each mode resolves to when --blocking is not given
RESOLVED_BLOCKING = {"default": "vdw", "v1": "box", "v2": "cylinder"}
BLOCKER_MODES = ("default", "default-cylinder", "v2")


# --------------------------------------------------------------------------
# helpers: constant-memory file comparison
# --------------------------------------------------------------------------


def file_digest(path, algorithm="sha256"):
    """Streamed digest; memory use does not depend on file size."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, algorithm).hexdigest()


def _first_difference(a, b, width=200):
    """(line number, line in a, line in b) of the first difference, streamed."""
    with open(a, newline="") as fa, open(b, newline="") as fb:
        for n, (la, lb) in enumerate(itertools.zip_longest(fa, fb), 1):
            if la != lb:
                return (
                    n,
                    (la or "<EOF>")[:width].rstrip("\n"),
                    (lb or "<EOF>")[:width].rstrip("\n"),
                )
    return None


def assert_same_file(a, b, label=None):
    """Byte-identical check without reading either file into memory."""
    a, b = Path(a), Path(b)
    label = label or a.name
    for p in (a, b):
        assert p.is_file(), f"{label}: missing {p}"
    if a.stat().st_size == b.stat().st_size and file_digest(a) == file_digest(b):
        return
    diff = _first_difference(a, b)
    where = (
        f"first difference at line {diff[0]}:\n  {a}: {diff[1]}\n  {b}: {diff[2]}"
        if diff
        else "files differ only in line endings"
    )
    pytest.fail(
        f"{label}: files differ ({a.stat().st_size} vs {b.stat().st_size} "
        f"bytes); {where}",
        pytrace=False,
    )


def iter_csv(path):
    """Stream CSV rows instead of materialising the whole file."""
    with open(path, newline="") as handle:
        yield from csv.DictReader(handle)


# --------------------------------------------------------------------------
# helpers: analysis results
# --------------------------------------------------------------------------


def ca_keys(glyco, analysis):
    """(chain, resid+icode, resname) of every CA column, as the CSVs write them."""
    chains = analysis._atom_attr(analysis.prot_atoms, "chainIDs")
    labels = glyco._resid_labels(analysis.prot_atoms)
    names = analysis._atom_attr(analysis.prot_atoms, "resnames")
    return [(str(c), str(l), str(n)) for c, l, n in zip(chains, labels, names)]


def unique_ca_keys(glyco, analysis):
    keys = ca_keys(glyco, analysis)
    if len(set(keys)) != len(keys):
        pytest.skip(
            "(chain, resid, resname) is not unique for this structure; "
            "CSV rows cannot be matched to residues"
        )
    return keys


def _last_k(analysis):
    return analysis.n_frames - 1


def _last_counts(analysis):
    return np.asarray(analysis.all_counts)[-1].astype(int)


def _last_analyzed(run):
    analyzed = run.analysis._analyzed_frames()
    if len(analyzed) < 2:
        pytest.skip("needs at least two analysed frames")
    return len(analyzed) - 1, int(analyzed[-1])


def _assert_positive_radii(calls, label):
    bad = [
        (
            c["caller"][0],
            a.get("chain"),
            a.get("resnum"),
            a.get("resname"),
            a.get("name"),
            a["radius"],
        )
        for c in calls
        for a in c["atoms"]
        if not a["radius"] > 0
    ]
    assert not bad, f"{label}: zero/invalid radius passed to FreeSASA: {bad[:10]}"


def _mode_cli_args(mode):
    """glyco CLI flags of a test mode."""
    glyco_mode, blocking = MODE_SPECS[mode]
    return ["--mode", glyco_mode] + (["--blocking", blocking] if blocking else [])


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def test_unknown_parameter_rejected(glyco, glyco_params, pytestconfig):
    """Typos in keyword arguments fail instead of being ignored."""
    if not pytestconfig.glyco_cases:
        pytest.skip("no structures given")
    case = pytestconfig.glyco_cases[0]
    u = glyco.load_universe(case.top, list(case.traj) or None)
    with pytest.raises(TypeError, match="vdw_scale"):
        glyco.GlycoAnalysis(u, **glyco_params(case), vdw_scal=0.8)


# --------------------------------------------------------------------------
# every input and mode
# --------------------------------------------------------------------------


def test_outputs(case, mode, runner):
    """All output files exist, and run_config.json describes the run."""
    r = runner(case, mode)
    missing = [f for f in EXPECTED_FILES if not (r.outdir / f).is_file()]
    assert not missing, f"missing output files: {missing}"

    config = json.loads((r.outdir / "run_config.json").read_text())
    glyco_mode, blocking = MODE_SPECS[mode]
    assert config["mode"] == glyco_mode
    assert config["blocking"] == (blocking or RESOLVED_BLOCKING[glyco_mode])
    assert config["trajectory_frames"] == r.analysis._analyzed_frames().tolist()


def test_counts_valid(case, mode, runner):
    a = runner(case, mode).analysis
    counts = np.asarray(a.all_counts)
    assert counts.shape == (a.n_frames, len(a.prot_atoms))
    assert np.issubdtype(counts.dtype, np.integer), counts.dtype
    assert np.all(counts >= 0)
    assert counts.max(initial=0) <= len(a.gly_atoms)


def test_buried_residues_have_zero_count(case, mode, runner):
    a = runner(case, mode).analysis
    buried = np.asarray(a.results.sasa) < a.area_cutoff
    assert np.all(np.asarray(a.all_counts)[buried] == 0)


def test_frame_summary(case, mode, runner):
    r = runner(case, mode)
    a = r.analysis
    analyzed = a._analyzed_frames().tolist()
    areas = np.asarray(a.residue_areas)
    n_rows = 0
    for k, row in enumerate(iter_csv(r.outdir / "frame_summary.csv")):
        assert k < a.n_frames, f"frame_summary.csv has more than {a.n_frames} rows"
        assert int(row["analyzed_frame"]) == k
        assert int(row["trajectory_frame"]) == analyzed[k]
        assert int(row["n_surface_residues"]) == a.results.n_surface_residues[k]
        surface_rows = np.flatnonzero(areas[k] >= a.area_cutoff)
        expected = int(np.isin(surface_rows, a._start_res_row).sum())
        assert a.results.n_surface_residues[k] == expected
        n_blocked = int(row["n_blocked_pairs"])
        mean_d = float(row["mean_blocked_distance"])
        assert n_blocked >= 0
        assert np.isnan(mean_d) == (n_blocked == 0)
        n_rows += 1
    assert n_rows == a.n_frames


def test_bfactors_npy(case, mode, runner):
    r = runner(case, mode)
    np.testing.assert_array_equal(
        np.load(r.outdir / "b_factors.npy"), r.analysis.all_counts, strict=True
    )


def test_residue_summary_matches_counts(case, mode, runner, glyco):
    r = runner(case, mode)
    a = r.analysis
    keys = unique_ca_keys(glyco, a)
    k_last = _last_k(a)
    tf_last = int(a._analyzed_frames()[k_last])
    counts, sasa = _last_counts(a), np.asarray(a.results.sasa)[k_last]

    got = {}
    for row in iter_csv(r.outdir / "residue_summary.csv"):
        assert int(row["analyzed_frame"]) == k_last
        assert int(row["trajectory_frame"]) == tf_last
        key = (row["chain"], row["resid"], row["resname"])
        assert key not in got, f"duplicate residue row {key}"
        got[key] = (int(row["glycan_count"]), float(row["sasa"]))

    # zero-count residues are omitted in every mode
    expected = {k: int(c) for k, c in zip(keys, counts) if c}
    assert {k: v[0] for k, v in got.items()} == expected
    for key, s in zip(keys, sasa):
        if key in got:
            assert got[key][1] == pytest.approx(s, rel=1e-5, abs=1e-6), key


def test_excluded_atoms_consistent(case, mode, runner, glyco):
    r = runner(case, mode)
    a = r.analysis
    keys = unique_ca_keys(glyco, a)
    n_gly = len(a.gly_atoms)
    count_of = {(c, rid): int(n) for (c, rid, _), n in zip(keys, _last_counts(a))}
    serials = {str(s) for s in glyco.pdb_atom_ids(a.gly_atoms)}

    per_residue, seen_in_residue = Counter(), {}
    for row in iter_csv(r.outdir / "excluded_glycan_atoms.csv"):
        key = (row["chain"], row["resid"])
        assert int(row["glycan_count"]) == count_of[key], key
        assert row["reason"] in EXCLUSION_REASONS, row["reason"]
        serial = row["glycan_atom_serial"]
        assert serial in serials, serial
        seen = seen_in_residue.setdefault(key, set())
        assert serial not in seen, f"duplicate row {key} {serial}"
        seen.add(serial)
        per_residue[key] += 1

    for key, n_rows in per_residue.items():
        assert n_rows == n_gly - count_of[key], key
    assert set(per_residue) == {k for k, n in count_of.items() if 0 < n < n_gly}
    assert not [w for w in r.warnings if "union size" in w]


def test_case_selections(case, case_settings, glyco):
    """Per-structure selections select atoms; legacy glycan names select the same atoms."""
    s = case_settings(case)
    u = glyco.load_universe(case.top, list(case.traj) or None)
    glycans = u.select_atoms(s["gs"])
    assert len(glycans), f"{case.id}: gs {s['gs']!r} selects no atoms"
    assert len(u.select_atoms(f"({s['ps']}) and name CA")), (
        f"{case.id}: ps {s['ps']!r} selects no CA atoms"
    )
    legacy = s["legacy_glycans"]
    if not legacy:
        pytest.skip("no legacy glycan names for this structure")
    too_long = [n for n in legacy if len(n) > 4]
    assert not too_long, (
        f"{case.id}: legacy tools read 4-character residue names: {too_long}"
    )
    legacy_atoms = u.select_atoms("resname " + " ".join(legacy))
    assert set(legacy_atoms.indices) == set(glycans.indices), (
        f"{case.id}: gs {s['gs']!r} selects {len(glycans)} atoms "
        f"({sorted(set(glycans.resnames))}), but legacy names {list(legacy)} select "
        f"{len(legacy_atoms)}; glycokit and the legacy tools would see different glycans"
    )


def test_cli_matches_library(
    case, mode, runner, glyco, glyco_cli_args, pytestconfig, tmp_path
):
    """The CLI writes the same results as the library; FreeSASA gets valid radii."""
    r = runner(case, mode)  # cached library run

    cli_dir = tmp_path / "cli"
    argv = ["-a", case.top]
    if case.traj:
        argv += ["-b", *case.traj]
    argv += [
        *glyco_cli_args(case),
        *_mode_cli_args(mode),
        "-o",
        str(cli_dir),
        "--force",
        "--write-excluded",
        "-j",
        str(pytestconfig.glyco_n_workers),
    ]
    if r.frames is not None:
        argv += ["--frames", *map(str, r.frames)]
    try:
        with capture(tmp_path / "radii_cli.json", require_calls=True) as probe:
            assert glyco.main(argv) == 0
    finally:
        logging.captureWarnings(False)

    _assert_positive_radii(probe["calls"], "CLI")
    for name in (*COMPARED_CSVS, "frame_summary.csv"):
        assert_same_file(r.outdir / name, cli_dir / name, f"CLI vs library: {name}")
    np.testing.assert_array_equal(
        np.load(cli_dir / "b_factors.npy"), r.analysis.all_counts, strict=True
    )


def test_bfactors_pdb_per_residue(case, mode, runner, glyco):
    """Every atom of a residue has the same B-factor; glycan atoms have 0."""
    r = runner(case, mode)
    a = r.analysis
    u = glyco.mda.Universe(str(r.outdir / "b_factors.pdb"))
    expected = dict(
        zip(
            a.prot_atoms.resindices.tolist(),
            a.all_counts[a._resolve_frame_index(a.bfactor_frame)].tolist(),
        )
    )
    for res in u.residues:
        values = set(res.atoms.tempfactors.round(2).tolist())
        assert len(values) == 1, f"{res}: several B-factors {sorted(values)}"
        assert values.pop() == pytest.approx(expected.get(res.resindex, 0), abs=0.01), (
            res
        )


@pytest.mark.slow
def test_n_workers_serial_equals_parallel(case, mode, runner, pytestconfig):
    """-j only changes how work is split (KD-tree threads, or v1 box processes)."""
    n = pytestconfig.glyco_n_workers
    n = 2 if n == 1 else n
    serial = runner(case, mode, n_workers=1)
    parallel = runner(case, mode, n_workers=n)  # cached if n is the session value
    for name in ("glycan_counts", "n_blocked_pairs", "mean_blocked_distance"):
        np.testing.assert_array_equal(
            getattr(serial.analysis.results, name),
            getattr(parallel.analysis.results, name),
            err_msg=name,
        )
    assert_same_file(
        serial.outdir / "excluded_glycan_atoms.csv",
        parallel.outdir / "excluded_glycan_atoms.csv",
        f"serial vs parallel ({n} workers)",
    )


@pytest.mark.glyco_modes("v1")
def test_get_blockers_rejects_box_test(case, mode, runner):
    a = runner(case, mode).analysis
    with pytest.raises(ValueError, match="box test"):
        a.get_blockers()


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------


@pytest.mark.glyco_kind("trajectory")
def test_frame_slicing_consistent(case, mode, runner):
    full = runner(case, mode)
    k, frame = _last_analyzed(full)
    single = runner(case, mode, frames=[frame])
    np.testing.assert_array_equal(
        single.analysis.all_counts[0], full.analysis.all_counts[k]
    )
    np.testing.assert_allclose(
        single.analysis.residue_areas[0], full.analysis.residue_areas[k]
    )


@pytest.mark.glyco_kind("trajectory")
def test_trajectory_frame_equals_single_structure(case, mode, runner, glyco):
    """Frame k of the trajectory == the same coordinates as a 1-frame Universe."""
    from MDAnalysis.coordinates.memory import MemoryReader

    full = runner(case, mode)
    k, frame = _last_analyzed(full)
    u = glyco.load_universe(case.top, list(case.traj))
    ts = u.trajectory[frame]
    extra = {} if ts.dimensions is None else {"dimensions": ts.dimensions}
    u1 = glyco.mda.Universe(
        case.top, u.atoms.positions.copy()[None], format=MemoryReader, **extra
    )
    single = runner(case, mode, frames=None, universe=u1)
    np.testing.assert_array_equal(
        single.analysis.all_counts[0], full.analysis.all_counts[k]
    )
    np.testing.assert_allclose(
        single.analysis.residue_areas[0], full.analysis.residue_areas[k]
    )


@pytest.mark.glyco_kind("trajectory")
@pytest.mark.glyco_modes(*BLOCKER_MODES)
def test_get_blockers_ignores_trajectory_position(case, mode, runner):
    """
    After run(), moving the trajectory must not change get_blockers(): SASA and
    coordinates both come from the requested analyzed frame, and the
    trajectory position is restored.
    """
    r = runner(case, mode)
    a = r.analysis
    k_last, _ = _last_analyzed(r)  # skips with < 2 frames
    other = int(a._analyzed_frames()[0])  # differs from the last frame
    traj = a.u.trajectory
    old = int(traj.frame)

    expected = a.get_blockers(frame_index=-1)
    try:
        traj[other]
        cached = a.get_blockers()  # default options: cache path
        assert traj.frame == other, "get_blockers() moved the trajectory"
        explicit = a.get_blockers(start_idx=a.surface_start_indices(k_last))
        assert traj.frame == other, "get_blockers(start_idx=...) moved the trajectory"
    finally:
        traj[old]  # runner results are shared

    for label, (blockers, dists) in (
        ("cached path", cached),
        ("explicit start_idx", explicit),
    ):
        np.testing.assert_array_equal(blockers, expected[0], err_msg=label)
        np.testing.assert_array_equal(dists, expected[1], err_msg=label)
    if len(expected[0]) == 0:
        pytest.skip(
            "no blocked pairs in the last frame; positions checked, "
            "blocker comparison was trivial"
        )


@pytest.mark.glyco_kind("trajectory")
@pytest.mark.glyco_modes(*BLOCKER_MODES)
def test_get_blockers_frame_index_matches_single_frame_run(case, mode, runner):
    """get_blockers(frame_index=0) of the full run == the same frame analyzed alone."""
    full = runner(case, mode)
    _last_analyzed(full)  # skips with < 2 frames
    first = int(full.analysis._analyzed_frames()[0])
    single = runner(case, mode, frames=[first])

    got_b, got_d = full.analysis.get_blockers(frame_index=0)  # recomputed (not cached)
    exp_b, exp_d = single.analysis.get_blockers()  # cached (only frame)
    np.testing.assert_array_equal(got_b, exp_b)
    np.testing.assert_array_equal(got_d, exp_d)
