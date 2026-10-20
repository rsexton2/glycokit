"""
Unit and regression test for the glycokit package.
"""
import csv
import logging
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

import glycokit
from glycokit.example_api import example_function

EXPECTED_FILES = ("b_factors.pdb", "b_factors.npy", "frame_summary.csv",
                  "residue_summary.csv", "excluded_glycan_atoms.csv")
EXCLUSION_REASONS = {"outside_distance_cutoff", "blocked_for_all_start_atoms"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

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


def test_mdanalysis_logo_length(mdanalysis_logo_text):
    """Example test using a fixture defined in conftest.py"""
    logo_lines = mdanalysis_logo_text.split("\n")
    assert len(logo_lines) == 46, "Logo file does not have 46 lines!"


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

    rows = read_csv(r.outdir / "excluded_glycan_atoms.csv")
    per_residue = Counter((x["protein_chain"].strip(), x["protein_resid"].strip())
                          for x in rows)
    reported = {(x["protein_chain"].strip(), x["protein_resid"].strip()):
                int(x["final_glycan_count"]) for x in rows}

    # one row per glycan atom missing from each nonzero residue's union
    for key, n_rows in per_residue.items():
        assert reported[key] == count_of[key], key
        assert n_rows == n_gly - count_of[key], key

    expected = {k for k, n in count_of.items() if 0 < n < n_gly}
    assert set(per_residue) == expected

    assert {x["reason"] for x in rows} <= EXCLUSION_REASONS
    serials = {str(s) for s in glyco.pdb_atom_ids(a.gly_atoms)}
    assert {x["glycan_atom_index"] for x in rows} <= serials
    pairs = [(x["protein_chain"], x["protein_resid"], x["glycan_atom_index"]) for x in rows]
    assert len(pairs) == len(set(pairs)), "duplicate excluded-atom rows"

    mismatches = [w for w in r.warnings if "union size" in w]
    assert not mismatches, mismatches[:5]


def test_cli_matches_library(case, mode, runner, glyco, glyco_cli_args,
                             pytestconfig, tmp_path):
    r = runner(case, mode)                      # skips if mode is not possible
    argv = ["-a", case.top]
    if case.traj:
        argv += ["-b", *case.traj]
    argv += [*glyco_cli_args, "-o", str(tmp_path), "--force", "--write-excluded",
             "--box-workers", str(pytestconfig.getoption("--glyco-box-workers"))]
    if mode == "compat":
        argv.append("--compat")
    elif mode == "compat-v2":
        argv.append("--compat-v2")
    if r.frames is not None:
        argv += ["--frames", *map(str, r.frames)]

    try:
        assert glyco.main(argv) == 0
    finally:
        logging.captureWarnings(False)          # main() routes warnings to a log

    for name in ("residue_summary.csv", "excluded_glycan_atoms.csv"):
        assert (tmp_path / name).read_text() == (r.outdir / name).read_text(), name


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

@pytest.mark.slow
@pytest.mark.glyco_modes("compat")
def test_box_workers_serial_equals_parallel(case, mode, runner, pytestconfig):
    n = max(2, pytestconfig.getoption("--glyco-box-workers"))
    serial = runner(case, mode, box_workers=1)
    parallel = runner(case, mode, box_workers=n)
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
def test_legacy_compare(case, pytestconfig):
    compare = pytestconfig.getoption("--glyco-compare")
    if not compare:
        pytest.skip("give --glyco-compare PATH/TO/compare.py to run")
    if not case.top.lower().endswith(".pdb"):
        pytest.skip("the legacy GLYCO tools read PDB files only")
    compare = os.path.abspath(compare)
    result = subprocess.run([sys.executable, compare, "--pdb", case.top],
                            cwd=os.path.dirname(compare),
                            capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout + result.stderr)[-5000:]