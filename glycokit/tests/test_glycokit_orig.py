"""
Unit and regression test for the glycokit package.
"""
import csv
import json
import logging
import os
import shlex
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from glycokit.example_api import example_function
from glycokit.tests.sasa_probe import capture, load_calls

EXPECTED_FILES = ("b_factors.pdb", "b_factors.npy", "frame_summary.csv",
                  "residue_summary.csv", "excluded_glycan_atoms.csv")
EXCLUSION_REASONS = {"outside_distance_cutoff", "blocked_for_all_start_atoms"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _sasa_signature(calls):
    """What FreeSASA was given: call type, parameters, per-atom radius + coordinates."""
    return [(c["call"], c["params"], c["n_atoms"],
             [(a["radius"], tuple(a["xyz"])) for a in c["atoms"]])
            for c in calls]


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def ca_keys(glyco, analysis):
    """(chain, resid+icode, resname) of every CA column, as the CSVs write them."""
    chains = analysis._atom_attr(analysis.prot_atoms, "chainIDs")
    labels = glyco._resid_labels(analysis.prot_atoms)
    names = analysis.prot_atoms.resnames
    return [(str(c).strip(), str(l), str(n).strip())
            for c, l, n in zip(chains, labels, names)]


def unique_ca_keys(glyco, analysis):
    keys = ca_keys(glyco, analysis)
    if len(set(keys)) != len(keys):
        pytest.skip("(chain, resid, resname) is not unique for this structure; "
                    "CSV rows cannot be matched to residues")
    return keys


def ca_sasa(analysis, k=-1):
    return np.asarray(analysis.residue_areas)[k, analysis._prot_atom_res_row]


def last_counts(analysis):
    return np.asarray(analysis.all_counts)[-1].astype(int)


# --------------------------------------------------------------------------
# every input and mode
# --------------------------------------------------------------------------


def test_glycokit_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "glycokit" in sys.modules


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (1, 2),
        (-5, -10),
        (1.5, 3.0),
        (-2.0, -4.0)
    ],
)
def test_double_value(value, expected):
    assert example_function(value) == expected


def test_outputs_exist(case, mode, runner):
    r = runner(case, mode)
    missing = [f for f in EXPECTED_FILES if not (r.outdir / f).is_file()]
    assert not missing, f"missing output files: {missing}"


def test_counts_valid(case, mode, runner):
    a = runner(case, mode).analysis
    counts = np.asarray(a.all_counts)
    assert counts.shape == (a.n_frames, len(a.prot_atoms))
    assert np.all(counts >= 0)
    assert np.array_equal(counts, np.round(counts)), "counts must be integers"
    assert counts.max(initial=0) <= len(a.gly_atoms)


def test_buried_residues_have_zero_count(case, mode, runner):
    a = runner(case, mode).analysis
    areas = np.asarray(a.residue_areas)[:, a._prot_atom_res_row]
    buried = areas < a.area_cutoff
    assert np.all(np.asarray(a.all_counts)[buried] == 0)


def test_frame_summary(case, mode, runner):
    r = runner(case, mode)
    a = r.analysis
    rows = read_csv(r.outdir / "frame_summary.csv")
    assert len(rows) == a.n_frames
    assert [int(float(x["frame"])) for x in rows] == a._analyzed_frames().tolist()

    for k, row in enumerate(rows):
        assert int(float(row["n_surface_residues"])) == a.results.n_surface_residues[k]
        surface_rows = np.flatnonzero(np.asarray(a.residue_areas)[k] >= a.area_cutoff)
        expected = int(np.isin(surface_rows, a._start_res_row).sum())
        assert a.results.n_surface_residues[k] == expected

        n_blocked = int(float(row["n_blocked_pairs"]))
        mean_d = float(row["mean_blocked_distance"])
        assert n_blocked >= 0
        assert np.isnan(mean_d) == (n_blocked == 0)


def test_bfactors_npy(case, mode, runner):
    r = runner(case, mode)
    saved = np.load(r.outdir / "b_factors.npy")
    np.testing.assert_array_equal(saved, np.asarray(r.analysis.all_counts))


def test_residue_summary_matches_counts(case, mode, runner, glyco):
    r = runner(case, mode)
    a = r.analysis
    keys = unique_ca_keys(glyco, a)
    counts, sasa = last_counts(a), ca_sasa(a)

    rows = read_csv(r.outdir / "residue_summary.csv")
    got = {}
    for row in rows:
        key = (row["Chain"].strip(), row["ResNum"].strip(), row["ResName"].strip())
        assert key not in got, f"duplicate residue row {key}"
        got[key] = (int(row["Glycan_density"]), float(row["SASA"]))

    nonzero_only = mode != "default"
    expected = {k: int(c) for k, c in zip(keys, counts) if c or not nonzero_only}
    assert {k: v[0] for k, v in got.items()} == expected

    for key, s in zip(keys, sasa):
        if key in got:
            assert got[key][1] == pytest.approx(s, rel=1e-5, abs=1e-6), key


def test_excluded_atoms_consistent(case, mode, runner, glyco):
    r = runner(case, mode)
    a = r.analysis
    keys = unique_ca_keys(glyco, a)
    n_gly = len(a.gly_atoms)
    count_of = {(c, rid): int(n) for (c, rid, _), n in zip(keys, last_counts(a))}
    serials = {str(s) for s in glyco.pdb_atom_ids(a.gly_atoms)}

    per_residue, seen_in_residue = Counter(), {}
    with open(r.outdir / "excluded_glycan_atoms.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["protein_chain"].strip(), row["protein_resid"].strip())
            assert int(row["final_glycan_count"]) == count_of[key], key
            assert row["reason"] in EXCLUSION_REASONS, row["reason"]
            serial = row["glycan_atom_index"]
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
    assert len(u.select_atoms(f"({s['ps']}) and name CA")), \
        f"{case.id}: ps {s['ps']!r} selects no CA atoms"

    legacy = s["legacy_glycans"]
    if not legacy:
        pytest.skip("no legacy glycan names for this structure")
    too_long = [n for n in legacy if len(n) > 4]
    assert not too_long, f"{case.id}: legacy tools read 4-character residue names: {too_long}"
    legacy_atoms = u.select_atoms("resname " + " ".join(legacy))
    assert set(legacy_atoms.indices) == set(glycans.indices), (
        f"{case.id}: gs {s['gs']!r} selects {len(glycans)} atoms "
        f"({sorted(set(glycans.resnames))}), but legacy names {list(legacy)} select "
        f"{len(legacy_atoms)}; glycokit and the legacy tools would see different glycans")


import hashlib


def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def test_cli_matches_library(case, mode, runner, glyco, glyco_cli_args,
                             pytestconfig, tmp_path):
    with capture(tmp_path / "radii_lib.json", require_calls=True):
        r = runner(case, mode, cache=False)       # fresh run inside the block

    cli_dir = tmp_path / "cli"
    argv = ["-a", case.top]
    if case.traj:
        argv += ["-b", *case.traj]
    argv += [*glyco_cli_args(case), "-o", str(cli_dir), "--force",
             "--write-excluded", "--box-workers", str(pytestconfig.glyco_box_workers)]
    if mode == "compat":
        argv.append("--compat")
    elif mode == "compat-v2":
        argv.append("--compat-v2")
    if r.frames is not None:
        argv += ["--frames", *map(str, r.frames)]

    try:
        with capture(tmp_path / "radii_cli.json", require_calls=True):
            assert glyco.main(argv) == 0
    finally:
        logging.captureWarnings(False)

    for name in ("residue_summary.csv", "excluded_glycan_atoms.csv"):
        lib_file, cli_file = r.outdir / name, cli_dir / name
        assert cli_file.is_file(), f"CLI did not write {cli_file}"
        assert _sha256(cli_file) == _sha256(lib_file), name


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------

def _last_analyzed(run):
    analyzed = run.analysis._analyzed_frames()
    if len(analyzed) < 2:
        pytest.skip("needs at least two analysed frames")
    return len(analyzed) - 1, int(analyzed[-1])


@pytest.mark.glyco_kind("trajectory")
def test_frame_slicing_consistent(case, mode, runner):
    full = runner(case, mode)
    k, frame = _last_analyzed(full)
    single = runner(case, mode, frames=[frame])
    np.testing.assert_array_equal(single.analysis.all_counts[0],
                                  full.analysis.all_counts[k])
    np.testing.assert_allclose(single.analysis.residue_areas[0],
                               full.analysis.residue_areas[k])


@pytest.mark.glyco_kind("trajectory")
def test_trajectory_frame_equals_single_structure(case, mode, runner, glyco):
    """Frame k of the trajectory == the same coordinates as a 1-frame Universe."""
    from MDAnalysis.coordinates.memory import MemoryReader

    full = runner(case, mode)
    k, frame = _last_analyzed(full)

    u = glyco.load_universe(case.top, list(case.traj))
    ts = u.trajectory[frame]
    extra = {} if ts.dimensions is None else {"dimensions": ts.dimensions}
    u1 = glyco.mda.Universe(case.top, u.atoms.positions.copy()[None],
                            format=MemoryReader, **extra)

    single = runner(case, mode, frames=None, universe=u1)
    np.testing.assert_array_equal(single.analysis.all_counts[0],
                                  full.analysis.all_counts[k])
    np.testing.assert_allclose(single.analysis.residue_areas[0],
                               full.analysis.residue_areas[k])


# --------------------------------------------------------------------------
# --compat only
# --------------------------------------------------------------------------


def _assert_positive_radii(calls, label):
    bad = [(c["caller"][0], a.get("chain"), a.get("resnum"), a.get("resname"),
            a.get("name"), a["radius"])
           for c in calls for a in c["atoms"] if not a["radius"] > 0]
    assert not bad, f"{label}: zero/invalid radius passed to FreeSASA: {bad[:10]}"


@pytest.mark.slow
@pytest.mark.glyco_modes("compat")
def test_box_workers_serial_equals_parallel(case, mode, runner, pytestconfig, tmp_path):
    n = max(2, pytestconfig.glyco_box_workers)
    probe_dir = os.environ.get("GLYCO_PROBE_DIR", str(tmp_path))

    with capture(os.path.join(probe_dir, "serial.json")) as serial_probe:
        serial = runner(case, mode, box_workers=1)
    with capture(os.path.join(probe_dir, "parallel.json")) as parallel_probe:
        parallel = runner(case, mode, box_workers=n)

    # check FreeSASA input first: fails at the exact atom, not as a CSV diff
    _assert_positive_radii(serial_probe["calls"], "serial")
    _assert_positive_radii(parallel_probe["calls"], "parallel")

    np.testing.assert_array_equal(serial.analysis.all_counts,
                                  parallel.analysis.all_counts)
    for name in ("residue_summary.csv", "excluded_glycan_atoms.csv"):
        assert (serial.outdir / name).read_text() == (parallel.outdir / name).read_text()


# --------------------------------------------------------------------------
# legacy tools (opt-in)
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.legacy
@pytest.mark.glyco_kind("structure")
def test_legacy_compare(case, compare_params, compare_base_args, pytestconfig, tmp_path):
    """compare.py gating checks with the resolved (and optionally varied) parameters."""
    script = pytestconfig.glyco_compare_script
    if not script:
        pytest.skip("give --glyco-compare=PATH or GLYCO_COMPARE_SCRIPT to run")
    if not case.top.lower().endswith(".pdb"):
        pytest.skip("the legacy GLYCO tools read PDB files only")
    if not pytestconfig.glyco_legacy_glycans:
        pytest.skip("cannot derive legacy glycan names from the glycan selection; "
                    "give --glyco-legacy-glycans or GLYCO_COMPARE_GLYCANS")

    summary = tmp_path / "compare_summary.json"
    argv = [sys.executable, script, "--pdb", case.top, *compare_base_args]
    for key, value in compare_params.items():          # later flags override
        argv += [pytestconfig.glyco_vary_flags[key], str(value)]
    argv += ["--json-summary", str(summary)]
    if pytestconfig.getoption("--glyco-compare-keep"):
        argv.append("--keep-output")

    try:
        result = subprocess.run(argv, cwd=os.path.dirname(script), capture_output=True,
                                text=True,
                                timeout=pytestconfig.getoption("--glyco-compare-timeout"))
    except subprocess.TimeoutExpired:
        pytest.fail(f"compare.py timed out\ncommand: {shlex.join(argv)}", pytrace=False)

    report = json.loads(summary.read_text()) if summary.is_file() else None
    if result.returncode != 0 or not (report and report["overall"]):
        failed = ([label for label, ok in report["checks"].items() if not ok]
                  if report else "compare.py did not finish (see output)")
        tail = (result.stdout + result.stderr)[-5000:]
        pytest.fail(f"failing checks: {failed}\ncommand: {shlex.join(argv)}\n\n{tail}",
                    pytrace=False)