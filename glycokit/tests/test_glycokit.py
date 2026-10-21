# """
# Unit and regression test for the glycokit package.
# """
# import csv
# import hashlib
# import itertools
# import json
# import logging
# import os
# import shlex
# import subprocess
# import sys
# from collections import Counter
# from pathlib import Path

# import numpy as np
# import pytest

# from glycokit.example_api import example_function
# from glycokit.tests.sasa_probe import capture, load_calls

# EXPECTED_FILES = ("b_factors.pdb", "b_factors.npy", "frame_summary.csv",
#                   "residue_summary.csv", "excluded_glycan_atoms.csv")
# EXCLUSION_REASONS = {"outside_distance_cutoff", "blocked_for_all_start_atoms"}
# NPY_BLOCK_BYTES = 1 << 24          # ~16 MB per comparison block


# # --------------------------------------------------------------------------
# # helpers: constant-memory file comparison
# # --------------------------------------------------------------------------
# def file_digest(path, algorithm="sha256"):
#     """Streamed digest; memory use does not depend on file size."""
#     with open(path, "rb") as fh:
#         return hashlib.file_digest(fh, algorithm).hexdigest()


# def _first_difference(a, b, width=200):
#     """(line number, line in a, line in b) of the first difference, streamed."""
#     with open(a, newline="") as fa, open(b, newline="") as fb:
#         for n, (la, lb) in enumerate(itertools.zip_longest(fa, fb), 1):
#             if la != lb:
#                 return n, (la or "<EOF>")[:width].rstrip("\n"), \
#                           (lb or "<EOF>")[:width].rstrip("\n")
#     return None


# def assert_same_file(a, b, label=None):
#     """Byte-identical check without reading either file into memory."""
#     a, b = Path(a), Path(b)
#     label = label or a.name
#     for p in (a, b):
#         assert p.is_file(), f"{label}: missing {p}"
#     if a.stat().st_size == b.stat().st_size and file_digest(a) == file_digest(b):
#         return
#     diff = _first_difference(a, b)
#     where = (f"first difference at line {diff[0]}:\n  {a}: {diff[1]}\n  {b}: {diff[2]}"
#              if diff else "files differ only in line endings")
#     pytest.fail(f"{label}: files differ ({a.stat().st_size} vs {b.stat().st_size} "
#                 f"bytes); {where}", pytrace=False)


# def iter_csv(path):
#     """Stream CSV rows instead of materialising the whole file."""
#     with open(path, newline="") as handle:
#         yield from csv.DictReader(handle)


# def sasa_digest(calls):
#     """Digest of what FreeSASA was given (call type, params, per-atom radius + xyz).
#     Same information as a full signature list, without building it."""
#     h = hashlib.sha256()
#     for c in calls:
#         h.update(repr((c["call"], c["params"], c["n_atoms"])).encode())
#         for atom in c["atoms"]:
#             h.update(np.asarray([atom["radius"], *atom["xyz"]], dtype=np.float64).tobytes())
#     return h.hexdigest()


# def ca_keys(glyco, analysis):
#     """(chain, resid+icode, resname) of every CA column, as the CSVs write them."""
#     chains = analysis._atom_attr(analysis.prot_atoms, "chainIDs")
#     labels = glyco._resid_labels(analysis.prot_atoms)
#     names = analysis.prot_atoms.resnames
#     return [(str(c).strip(), str(l), str(n).strip())
#             for c, l, n in zip(chains, labels, names)]


# def unique_ca_keys(glyco, analysis):
#     keys = ca_keys(glyco, analysis)
#     if len(set(keys)) != len(keys):
#         pytest.skip("(chain, resid, resname) is not unique for this structure; "
#                     "CSV rows cannot be matched to residues")
#     return keys


# def ca_sasa(analysis, k=-1):
#     return np.asarray(analysis.residue_areas)[k, analysis._prot_atom_res_row]


# def _last_counts(analysis):
#     return np.asarray(analysis.all_counts)[-1].astype(int)


# # --------------------------------------------------------------------------
# # every input and mode
# # --------------------------------------------------------------------------

# def test_glycokit_imported():
#     """Sample test, will always pass so long as import statement worked"""
#     assert "glycokit" in sys.modules


# @pytest.mark.parametrize("value, expected",
#                          [(0, 0), (1, 2), (-5, -10), (1.5, 3.0), (-2.0, -4.0)])
# def test_double_value(value, expected):
#     assert example_function(value) == expected


# def test_outputs_exist(case, mode, runner):
#     r = runner(case, mode)
#     missing = [f for f in EXPECTED_FILES if not (r.outdir / f).is_file()]
#     assert not missing, f"missing output files: {missing}"


# def test_counts_valid(case, mode, runner):
#     a = runner(case, mode).analysis
#     counts = np.asarray(a.all_counts)
#     assert counts.shape == (a.n_frames, len(a.prot_atoms))
#     assert np.all(counts >= 0)
#     assert np.array_equal(counts, np.round(counts)), "counts must be integers"
#     assert counts.max(initial=0) <= len(a.gly_atoms)


# def test_buried_residues_have_zero_count(case, mode, runner):
#     a = runner(case, mode).analysis
#     areas = np.asarray(a.residue_areas)[:, a._prot_atom_res_row]
#     buried = areas < a.area_cutoff
#     assert np.all(np.asarray(a.all_counts)[buried] == 0)


# def test_frame_summary(case, mode, runner):
#     r = runner(case, mode)
#     a = r.analysis
#     analyzed = a._analyzed_frames().tolist()
#     areas = np.asarray(a.residue_areas)
#     n_rows = 0
#     for k, row in enumerate(iter_csv(r.outdir / "frame_summary.csv")):
#         assert k < a.n_frames, f"frame_summary.csv has more than {a.n_frames} rows"
#         assert int(float(row["frame"])) == analyzed[k]
#         assert int(float(row["n_surface_residues"])) == a.results.n_surface_residues[k]
#         surface_rows = np.flatnonzero(areas[k] >= a.area_cutoff)
#         expected = int(np.isin(surface_rows, a._start_res_row).sum())
#         assert a.results.n_surface_residues[k] == expected
#         n_blocked = int(float(row["n_blocked_pairs"]))
#         mean_d = float(row["mean_blocked_distance"])
#         assert n_blocked >= 0
#         assert np.isnan(mean_d) == (n_blocked == 0)
#         n_rows += 1
#     assert n_rows == a.n_frames


# def test_bfactors_npy(case, mode, runner):
#     r = runner(case, mode)
#     expected = np.asarray(r.analysis.all_counts)
#     path = r.outdir / "b_factors.npy"
#     try:
#         saved = np.load(path, mmap_mode="r")      # header read; data paged on demand
#     except ValueError:                             # e.g. zero-size array cannot be mapped
#         saved = np.load(path)
#     try:
#         assert saved.shape == expected.shape, (saved.shape, expected.shape)
#         if saved.size:
#             row_bytes = max(1, saved[0].nbytes) if saved.ndim > 1 else saved.itemsize
#             step = max(1, NPY_BLOCK_BYTES // row_bytes)
#             for s in range(0, len(saved), step):
#                 np.testing.assert_array_equal(saved[s:s + step], expected[s:s + step],
#                                               err_msg=f"b_factors.npy rows {s}..{s + step - 1}")
#     finally:
#         del saved                                  # release the map (matters on Windows)


# def test_residue_summary_matches_counts(case, mode, runner, glyco):
#     r = runner(case, mode)
#     a = r.analysis
#     keys = unique_ca_keys(glyco, a)
#     counts, sasa = _last_counts(a), ca_sasa(a)
#     got = {}
#     for row in iter_csv(r.outdir / "residue_summary.csv"):
#         key = (row["Chain"].strip(), row["ResNum"].strip(), row["ResName"].strip())
#         assert key not in got, f"duplicate residue row {key}"
#         got[key] = (int(row["Glycan_density"]), float(row["SASA"]))
#     nonzero_only = mode != "default"
#     expected = {k: int(c) for k, c in zip(keys, counts) if c or not nonzero_only}
#     assert {k: v[0] for k, v in got.items()} == expected
#     for key, s in zip(keys, sasa):
#         if key in got:
#             assert got[key][1] == pytest.approx(s, rel=1e-5, abs=1e-6), key


# def test_excluded_atoms_consistent(case, mode, runner, glyco):
#     r = runner(case, mode)
#     a = r.analysis
#     keys = unique_ca_keys(glyco, a)
#     n_gly = len(a.gly_atoms)
#     count_of = {(c, rid): int(n) for (c, rid, _), n in zip(keys, _last_counts(a))}
#     serials = {str(s) for s in glyco.pdb_atom_ids(a.gly_atoms)}
#     per_residue, seen_in_residue = Counter(), {}
#     for row in iter_csv(r.outdir / "excluded_glycan_atoms.csv"):
#         key = (row["protein_chain"].strip(), row["protein_resid"].strip())
#         assert int(row["final_glycan_count"]) == count_of[key], key
#         assert row["reason"] in EXCLUSION_REASONS, row["reason"]
#         serial = row["glycan_atom_index"]
#         assert serial in serials, serial
#         seen = seen_in_residue.setdefault(key, set())
#         assert serial not in seen, f"duplicate row {key} {serial}"
#         seen.add(serial)
#         per_residue[key] += 1
#     for key, n_rows in per_residue.items():
#         assert n_rows == n_gly - count_of[key], key
#     assert set(per_residue) == {k for k, n in count_of.items() if 0 < n < n_gly}
#     assert not [w for w in r.warnings if "union size" in w]


# def test_case_selections(case, case_settings, glyco):
#     """Per-structure selections select atoms; legacy glycan names select the same atoms."""
#     s = case_settings(case)
#     u = glyco.load_universe(case.top, list(case.traj) or None)
#     glycans = u.select_atoms(s["gs"])
#     assert len(glycans), f"{case.id}: gs {s['gs']!r} selects no atoms"
#     assert len(u.select_atoms(f"({s['ps']}) and name CA")), \
#         f"{case.id}: ps {s['ps']!r} selects no CA atoms"
#     legacy = s["legacy_glycans"]
#     if not legacy:
#         pytest.skip("no legacy glycan names for this structure")
#     too_long = [n for n in legacy if len(n) > 4]
#     assert not too_long, f"{case.id}: legacy tools read 4-character residue names: {too_long}"
#     legacy_atoms = u.select_atoms("resname " + " ".join(legacy))
#     assert set(legacy_atoms.indices) == set(glycans.indices), (
#         f"{case.id}: gs {s['gs']!r} selects {len(glycans)} atoms "
#         f"({sorted(set(glycans.resnames))}), but legacy names {list(legacy)} select "
#         f"{len(legacy_atoms)}; glycokit and the legacy tools would see different glycans")


# def test_cli_matches_library(case, mode, runner, glyco, glyco_cli_args,
#                              pytestconfig, tmp_path):
#     with capture(tmp_path / "radii_lib.json", require_calls=True):
#         r = runner(case, mode, cache=False)       # fresh run inside the block
#     cli_dir = tmp_path / "cli"
#     argv = ["-a", case.top]
#     if case.traj:
#         argv += ["-b", *case.traj]
#     argv += [*glyco_cli_args(case), "-o", str(cli_dir), "--force",
#              "--write-excluded", "--box-workers", str(pytestconfig.glyco_box_workers)]
#     if mode == "compat":
#         argv.append("--compat")
#     elif mode == "compat-v2":
#         argv.append("--compat-v2")
#     if r.frames is not None:
#         argv += ["--frames", *map(str, r.frames)]
#     try:
#         with capture(tmp_path / "radii_cli.json", require_calls=True):
#             assert glyco.main(argv) == 0
#     finally:
#         logging.captureWarnings(False)
#     for name in ("residue_summary.csv", "excluded_glycan_atoms.csv"):
#         assert_same_file(r.outdir / name, cli_dir / name, f"CLI vs library: {name}")


# # --------------------------------------------------------------------------
# # trajectories
# # --------------------------------------------------------------------------
# def lastanalyzed(run):
#     analyzed = run.analysis._analyzed_frames()
#     if len(analyzed) < 2:
#         pytest.skip("needs at least two analysed frames")
#     return len(analyzed) - 1, int(analyzed[-1])


# @pytest.mark.glyco_kind("trajectory")
# def test_frame_slicing_consistent(case, mode, runner):
#     full = runner(case, mode)
#     k, frame = lastanalyzed(full)
#     single = runner(case, mode, frames=[frame])
#     np.testing.assert_array_equal(single.analysis.all_counts[0],
#                                   full.analysis.all_counts[k])
#     np.testing.assert_allclose(single.analysis.residue_areas[0],
#                                full.analysis.residue_areas[k])


# @pytest.mark.glyco_kind("trajectory")
# def test_trajectory_frame_equals_single_structure(case, mode, runner, glyco):
#     """Frame k of the trajectory == the same coordinates as a 1-frame Universe."""
#     from MDAnalysis.coordinates.memory import MemoryReader
#     full = runner(case, mode)
#     k, frame = lastanalyzed(full)
#     u = glyco.load_universe(case.top, list(case.traj))
#     ts = u.trajectory[frame]
#     extra = {} if ts.dimensions is None else {"dimensions": ts.dimensions}
#     u1 = glyco.mda.Universe(case.top, u.atoms.positions.copy()[None],
#                             format=MemoryReader, **extra)
#     single = runner(case, mode, frames=None, universe=u1)
#     np.testing.assert_array_equal(single.analysis.all_counts[0],
#                                   full.analysis.all_counts[k])
#     np.testing.assert_allclose(single.analysis.residue_areas[0],
#                                full.analysis.residue_areas[k])


# # --------------------------------------------------------------------------
# # --compat only
# # --------------------------------------------------------------------------
# def assertpositive_radii(calls, label):
#     bad = [(c["caller"][0], a.get("chain"), a.get("resnum"), a.get("resname"),
#             a.get("name"), a["radius"])
#            for c in calls for a in c["atoms"] if not a["radius"] > 0]
#     assert not bad, f"{label}: zero/invalid radius passed to FreeSASA: {bad[:10]}"


# @pytest.mark.slow
# @pytest.mark.glyco_modes("compat")
# def test_box_workers_serial_equals_parallel(case, mode, runner, pytestconfig, tmp_path):
#     n = max(2, pytestconfig.glyco_box_workers)
#     probe_dir = os.environ.get("GLYCO_PROBE_DIR", str(tmp_path))
#     with capture(os.path.join(probe_dir, "serial.json")) as serial_probe:
#         serial = runner(case, mode, box_workers=1)
#     with capture(os.path.join(probe_dir, "parallel.json")) as parallel_probe:
#         parallel = runner(case, mode, box_workers=n)
#     # check FreeSASA input first: fails at the exact atom, not as a CSV diff
#     assertpositive_radii(serial_probe["calls"], "serial")
#     assertpositive_radii(parallel_probe["calls"], "parallel")
#     np.testing.assert_array_equal(serial.analysis.all_counts,
#                                   parallel.analysis.all_counts)
#     for name in ("residue_summary.csv", "excluded_glycan_atoms.csv"):
#         assert_same_file(serial.outdir / name, parallel.outdir / name,
#                          f"serial vs parallel ({n} workers): {name}")


# # --------------------------------------------------------------------------
# # legacy tools (opt-in)
# # --------------------------------------------------------------------------
# @pytest.mark.slow
# @pytest.mark.legacy
# @pytest.mark.glyco_kind("structure")
# def test_legacy_compare(case, compare_params, compare_base_args, pytestconfig, tmp_path):
#     """compare.py gating checks with the resolved (and optionally varied) parameters."""
#     script = pytestconfig.glyco_compare_script
#     if not script:
#         pytest.skip("give --glyco-compare=PATH or GLYCO_COMPARE_SCRIPT to run")
#     if not case.top.lower().endswith(".pdb"):
#         pytest.skip("the legacy GLYCO tools read PDB files only")
#     if not pytestconfig.glyco_legacy_glycans:
#         pytest.skip("cannot derive legacy glycan names from the glycan selection; "
#                     "give --glyco-legacy-glycans or GLYCO_COMPARE_GLYCANS")
#     summary = tmp_path / "compare_summary.json"
#     argv = [sys.executable, script, "--pdb", case.top, *compare_base_args]
#     for key, value in compare_params.items():          # later flags override
#         argv += [pytestconfig.glyco_vary_flags[key], str(value)]
#     argv += ["--json-summary", str(summary)]
#     if pytestconfig.getoption("--glyco-compare-keep"):
#         argv.append("--keep-output")
#     try:
#         result = subprocess.run(argv, cwd=os.path.dirname(script), capture_output=True,
#                                 text=True,
#                                 timeout=pytestconfig.getoption("--glyco-compare-timeout"))
#     except subprocess.TimeoutExpired:
#         pytest.fail(f"compare.py timed out\ncommand: {shlex.join(argv)}", pytrace=False)
#     report = json.loads(summary.read_text()) if summary.is_file() else None
#     if result.returncode != 0 or not (report and report["overall"]):
#         failed = ([label for label, ok in report["checks"].items() if not ok]
#                   if report else "compare.py did not finish (see output)")
#         tail = (result.stdout + result.stderr)[-5000:]
#         pytest.fail(f"failing checks: {failed}\ncommand: {shlex.join(argv)}\n\n{tail}",
#                     pytrace=False)

# def test_mdanalysis_logo_length(mdanalysis_logo_text):
#     """Example test using a fixture defined in conftest.py"""
#     logo_lines = mdanalysis_logo_text.split("\n")
#     assert len(logo_lines) == 46, "Logo file does not have 46 lines!"


# @pytest.mark.parametrize(
#     "value, expected",
#     [
#         (0, 0),
#         (1, 2),
#         (-5, -10),
#         (1.5, 3.0),
#         (-2.0, -4.0)
#     ],
# )
# def test_double_value(value, expected):
#     assert example_function(value) == expected


# def test_outputs_exist(case, mode, runner):
#     r = runner(case, mode)
#     missing = [f for f in EXPECTED_FILES if not (r.outdir / f).is_file()]
#     assert not missing, f"missing output files: {missing}"


# def test_counts_valid(case, mode, runner):
#     a = runner(case, mode).analysis
#     counts = np.asarray(a.all_counts)
#     assert counts.shape == (a.n_frames, len(a.prot_atoms))
#     assert np.all(counts >= 0)
#     assert np.array_equal(counts, np.round(counts)), "counts must be integers"
#     assert counts.max(initial=0) <= len(a.gly_atoms)


# def test_buried_residues_have_zero_count(case, mode, runner):
#     a = runner(case, mode).analysis
#     areas = np.asarray(a.residue_areas)[:, a._prot_atom_res_row]
#     buried = areas < a.area_cutoff
#     assert np.all(np.asarray(a.all_counts)[buried] == 0)


# def test_frame_summary(case, mode, runner):
#     r = runner(case, mode)
#     a = r.analysis
#     analyzed = a._analyzed_frames().tolist()
#     areas = np.asarray(a.residue_areas)
#     n_rows = 0
#     for k, row in enumerate(iter_csv(r.outdir / "frame_summary.csv")):
#         assert k < a.n_frames, f"frame_summary.csv has more than {a.n_frames} rows"
#         assert int(float(row["frame"])) == analyzed[k]
#         assert int(float(row["n_surface_residues"])) == a.results.n_surface_residues[k]
#         surface_rows = np.flatnonzero(areas[k] >= a.area_cutoff)
#         expected = int(np.isin(surface_rows, a._start_res_row).sum())
#         assert a.results.n_surface_residues[k] == expected
#         n_blocked = int(float(row["n_blocked_pairs"]))
#         mean_d = float(row["mean_blocked_distance"])
#         assert n_blocked >= 0
#         assert np.isnan(mean_d) == (n_blocked == 0)
#         n_rows += 1
#     assert n_rows == a.n_frames


# def test_bfactors_npy(case, mode, runner):
#     r = runner(case, mode)
#     expected = np.asarray(r.analysis.all_counts)
#     path = r.outdir / "b_factors.npy"
#     try:
#         saved = np.load(path, mmap_mode="r")      # header read; data paged on demand
#     except ValueError:                             # e.g. zero-size array cannot be mapped
#         saved = np.load(path)
#     try:
#         assert saved.shape == expected.shape, (saved.shape, expected.shape)
#         if saved.size:
#             row_bytes = max(1, saved[0].nbytes) if saved.ndim > 1 else saved.itemsize
#             step = max(1, NPY_BLOCK_BYTES // row_bytes)
#             for s in range(0, len(saved), step):
#                 np.testing.assert_array_equal(saved[s:s + step], expected[s:s + step],
#                                               err_msg=f"b_factors.npy rows {s}..{s + step - 1}")
#     finally:
#         del saved                                  # release the map (matters on Windows)


# def test_residue_summary_matches_counts(case, mode, runner, glyco):
#     r = runner(case, mode)
#     a = r.analysis
#     keys = unique_ca_keys(glyco, a)
#     counts, sasa = _last_counts(a), ca_sasa(a)
#     got = {}
#     for row in iter_csv(r.outdir / "residue_summary.csv"):
#         key = (row["Chain"].strip(), row["ResNum"].strip(), row["ResName"].strip())
#         assert key not in got, f"duplicate residue row {key}"
#         got[key] = (int(row["Glycan_density"]), float(row["SASA"]))
#     nonzero_only = mode != "default"
#     expected = {k: int(c) for k, c in zip(keys, counts) if c or not nonzero_only}
#     assert {k: v[0] for k, v in got.items()} == expected
#     for key, s in zip(keys, sasa):
#         if key in got:
#             assert got[key][1] == pytest.approx(s, rel=1e-5, abs=1e-6), key


# def test_excluded_atoms_consistent(case, mode, runner, glyco):
#     r = runner(case, mode)
#     a = r.analysis
#     keys = unique_ca_keys(glyco, a)
#     n_gly = len(a.gly_atoms)
#     count_of = {(c, rid): int(n) for (c, rid, _), n in zip(keys, _last_counts(a))}
#     serials = {str(s) for s in glyco.pdb_atom_ids(a.gly_atoms)}
#     per_residue, seen_in_residue = Counter(), {}
#     for row in iter_csv(r.outdir / "excluded_glycan_atoms.csv"):
#         key = (row["protein_chain"].strip(), row["protein_resid"].strip())
#         assert int(row["final_glycan_count"]) == count_of[key], key
#         assert row["reason"] in EXCLUSION_REASONS, row["reason"]
#         serial = row["glycan_atom_index"]
#         assert serial in serials, serial
#         seen = seen_in_residue.setdefault(key, set())
#         assert serial not in seen, f"duplicate row {key} {serial}"
#         seen.add(serial)
#         per_residue[key] += 1
#     for key, n_rows in per_residue.items():
#         assert n_rows == n_gly - count_of[key], key
#     assert set(per_residue) == {k for k, n in count_of.items() if 0 < n < n_gly}
#     assert not [w for w in r.warnings if "union size" in w]


# def test_case_selections(case, case_settings, glyco):
#     """Per-structure selections select atoms; legacy glycan names select the same atoms."""
#     s = case_settings(case)
#     u = glyco.load_universe(case.top, list(case.traj) or None)
#     glycans = u.select_atoms(s["gs"])
#     assert len(glycans), f"{case.id}: gs {s['gs']!r} selects no atoms"
#     assert len(u.select_atoms(f"({s['ps']}) and name CA")), \
#         f"{case.id}: ps {s['ps']!r} selects no CA atoms"
#     legacy = s["legacy_glycans"]
#     if not legacy:
#         pytest.skip("no legacy glycan names for this structure")
#     too_long = [n for n in legacy if len(n) > 4]
#     assert not too_long, f"{case.id}: legacy tools read 4-character residue names: {too_long}"
#     legacy_atoms = u.select_atoms("resname " + " ".join(legacy))
#     assert set(legacy_atoms.indices) == set(glycans.indices), (
#         f"{case.id}: gs {s['gs']!r} selects {len(glycans)} atoms "
#         f"({sorted(set(glycans.resnames))}), but legacy names {list(legacy)} select "
#         f"{len(legacy_atoms)}; glycokit and the legacy tools would see different glycans")


# def test_cli_matches_library(case, mode, runner, glyco, glyco_cli_args,
#                              pytestconfig, tmp_path):
#     with capture(tmp_path / "radii_lib.json", require_calls=True):
#         r = runner(case, mode, cache=False)       # fresh run inside the block
#     cli_dir = tmp_path / "cli"
#     argv = ["-a", case.top]
#     if case.traj:
#         argv += ["-b", *case.traj]
#     argv += [*glyco_cli_args(case), "-o", str(cli_dir), "--force",
#              "--write-excluded", "--box-workers", str(pytestconfig.glyco_box_workers)]
#     if mode == "compat":
#         argv.append("--compat")
#     elif mode == "compat-v2":
#         argv.append("--compat-v2")
#     if r.frames is not None:
#         argv += ["--frames", *map(str, r.frames)]
#     try:
#         with capture(tmp_path / "radii_cli.json", require_calls=True):
#             assert glyco.main(argv) == 0
#     finally:
#         logging.captureWarnings(False)
#     for name in ("residue_summary.csv", "excluded_glycan_atoms.csv"):
#         assert_same_file(r.outdir / name, cli_dir / name, f"CLI vs library: {name}")


# # --------------------------------------------------------------------------
# # trajectories
# # --------------------------------------------------------------------------
# def lastanalyzed(run):
#     analyzed = run.analysis._analyzed_frames()
#     if len(analyzed) < 2:
#         pytest.skip("needs at least two analysed frames")
#     return len(analyzed) - 1, int(analyzed[-1])


# @pytest.mark.glyco_kind("trajectory")
# def test_frame_slicing_consistent(case, mode, runner):
#     full = runner(case, mode)
#     k, frame = lastanalyzed(full)
#     single = runner(case, mode, frames=[frame])
#     np.testing.assert_array_equal(single.analysis.all_counts[0],
#                                   full.analysis.all_counts[k])
#     np.testing.assert_allclose(single.analysis.residue_areas[0],
#                                full.analysis.residue_areas[k])


# @pytest.mark.glyco_kind("trajectory")
# def test_trajectory_frame_equals_single_structure(case, mode, runner, glyco):
#     """Frame k of the trajectory == the same coordinates as a 1-frame Universe."""
#     from MDAnalysis.coordinates.memory import MemoryReader
#     full = runner(case, mode)
#     k, frame = lastanalyzed(full)
#     u = glyco.load_universe(case.top, list(case.traj))
#     ts = u.trajectory[frame]
#     extra = {} if ts.dimensions is None else {"dimensions": ts.dimensions}
#     u1 = glyco.mda.Universe(case.top, u.atoms.positions.copy()[None],
#                             format=MemoryReader, **extra)
#     single = runner(case, mode, frames=None, universe=u1)
#     np.testing.assert_array_equal(single.analysis.all_counts[0],
#                                   full.analysis.all_counts[k])
#     np.testing.assert_allclose(single.analysis.residue_areas[0],
#                                full.analysis.residue_areas[k])


# # --------------------------------------------------------------------------
# # --compat only
# # --------------------------------------------------------------------------
# def assertpositive_radii(calls, label):
#     bad = [(c["caller"][0], a.get("chain"), a.get("resnum"), a.get("resname"),
#             a.get("name"), a["radius"])
#            for c in calls for a in c["atoms"] if not a["radius"] > 0]
#     assert not bad, f"{label}: zero/invalid radius passed to FreeSASA: {bad[:10]}"


# @pytest.mark.slow
# @pytest.mark.glyco_modes("compat")
# def test_box_workers_serial_equals_parallel(case, mode, runner, pytestconfig, tmp_path):
#     n = max(2, pytestconfig.glyco_box_workers)
#     probe_dir = os.environ.get("GLYCO_PROBE_DIR", str(tmp_path))
#     with capture(os.path.join(probe_dir, "serial.json")) as serial_probe:
#         serial = runner(case, mode, box_workers=1)
#     with capture(os.path.join(probe_dir, "parallel.json")) as parallel_probe:
#         parallel = runner(case, mode, box_workers=n)
#     # check FreeSASA input first: fails at the exact atom, not as a CSV diff
#     assertpositive_radii(serial_probe["calls"], "serial")
#     assertpositive_radii(parallel_probe["calls"], "parallel")
#     np.testing.assert_array_equal(serial.analysis.all_counts,
#                                   parallel.analysis.all_counts)
#     for name in ("residue_summary.csv", "excluded_glycan_atoms.csv"):
#         assert_same_file(serial.outdir / name, parallel.outdir / name,
#                          f"serial vs parallel ({n} workers): {name}")


# # --------------------------------------------------------------------------
# # legacy tools (opt-in)
# # --------------------------------------------------------------------------
# @pytest.mark.slow
# @pytest.mark.legacy
# @pytest.mark.glyco_kind("structure")
# def test_legacy_compare(case, compare_params, compare_base_args, pytestconfig, tmp_path):
#     """compare.py gating checks with the resolved (and optionally varied) parameters."""
#     script = pytestconfig.glyco_compare_script
#     if not script:
#         pytest.skip("give --glyco-compare=PATH or GLYCO_COMPARE_SCRIPT to run")
#     if not case.top.lower().endswith(".pdb"):
#         pytest.skip("the legacy GLYCO tools read PDB files only")
#     compare = os.path.abspath(compare)
#     result = subprocess.run([sys.executable, compare, "--pdb", case.top],
#                             cwd=os.path.dirname(compare),
#                             capture_output=True, text=True)
#     assert result.returncode == 0, (result.stdout + result.stderr)[-5000:]
#     # if not pytestconfig.glyco_legacy_glycans:
#     #     pytest.skip("cannot derive legacy glycan names from the glycan selection; "
#     #                 "give --glyco-legacy-glycans or GLYCO_COMPARE_GLYCANS")
#     # summary = tmp_path / "compare_summary.json"
#     # argv = [sys.executable, script, "--pdb", case.top, *compare_base_args]
#     # for key, value in compare_params.items():          # later flags override
#     #     argv += [pytestconfig.glyco_vary_flags[key], str(value)]
#     # argv += ["--json-summary", str(summary)]
#     # if pytestconfig.getoption("--glyco-compare-keep"):
#     #     argv.append("--keep-output")
#     # try:
#     #     result = subprocess.run(argv, cwd=os.path.dirname(script), capture_output=True,
#     #                             text=True,
#     #                             timeout=pytestconfig.getoption("--glyco-compare-timeout"))
#     # except subprocess.TimeoutExpired:
#     #     pytest.fail(f"compare.py timed out\ncommand: {shlex.join(argv)}", pytrace=False)
#     # report = json.loads(summary.read_text()) if summary.is_file() else None
#     # if result.returncode != 0 or not (report and report["overall"]):
#     #     failed = ([label for label, ok in report["checks"].items() if not ok]
#     #               if report else "compare.py did not finish (see output)")
#     #     tail = (result.stdout + result.stderr)[-5000:]
#     #     pytest.fail(f"failing checks: {failed}\ncommand: {shlex.join(argv)}\n\n{tail}",
#     #                 pytrace=False)
"""
Unit and regression tests for the glycokit package.

Inputs come from conftest.py options, e.g.
    pytest --glyco-structure=a.pdb
    pytest -n auto --dist loadgroup --glyco-order slow-first
"""
import csv
import hashlib
import itertools
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from glycokit.example_api import example_function
from glycokit.tests.sasa_probe import capture

EXPECTED_FILES = ("b_factors.pdb", "b_factors.npy", "frame_summary.csv",
                  "residue_summary.csv", "excluded_glycan_atoms.csv")
EXCLUSION_REASONS = {"outside_distance_cutoff", "blocked_for_all_start_atoms"}
COMPARED_CSVS = ("residue_summary.csv", "excluded_glycan_atoms.csv")
NPY_BLOCK_BYTES = 1 << 24          # ~16 MB per comparison block


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
                return (n, (la or "<EOF>")[:width].rstrip("\n"),
                        (lb or "<EOF>")[:width].rstrip("\n"))
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
    where = (f"first difference at line {diff[0]}:\n  {a}: {diff[1]}\n  {b}: {diff[2]}"
             if diff else "files differ only in line endings")
    pytest.fail(f"{label}: files differ ({a.stat().st_size} vs {b.stat().st_size} "
                f"bytes); {where}", pytrace=False)


def iter_csv(path):
    """Stream CSV rows instead of materialising the whole file."""
    with open(path, newline="") as handle:
        yield from csv.DictReader(handle)


def sasa_digest(calls):
    """Digest of what FreeSASA was given (call type, params, per-atom radius + xyz)."""
    h = hashlib.sha256()
    for c in calls:
        h.update(repr((c["call"], c["params"], c["n_atoms"])).encode())
        for atom in c["atoms"]:
            h.update(np.asarray([atom["radius"], *atom["xyz"]], dtype=np.float64).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# helpers: analysis results
# --------------------------------------------------------------------------
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


def _last_counts(analysis):
    return np.asarray(analysis.all_counts)[-1].astype(int)


def _last_analyzed(run):
    analyzed = run.analysis._analyzed_frames()
    if len(analyzed) < 2:
        pytest.skip("needs at least two analysed frames")
    return len(analyzed) - 1, int(analyzed[-1])


def _assert_positive_radii(calls, label):
    bad = [(c["caller"][0], a.get("chain"), a.get("resnum"), a.get("resname"),
            a.get("name"), a["radius"])
           for c in calls for a in c["atoms"] if not a["radius"] > 0]
    assert not bad, f"{label}: zero/invalid radius passed to FreeSASA: {bad[:10]}"


def _check_passed(value):
    """compare.py checks may be booleans or {'passed': bool, ...} dicts."""
    return bool(value.get("passed")) if isinstance(value, dict) else bool(value)


# --------------------------------------------------------------------------
# sanity
# --------------------------------------------------------------------------
def test_glycokit_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "glycokit" in sys.modules


@pytest.mark.parametrize("value, expected",
                         [(0, 0), (1, 2), (-5, -10), (1.5, 3.0), (-2.0, -4.0)])
def test_double_value(value, expected):
    assert example_function(value) == expected


# --------------------------------------------------------------------------
# every input and mode
# --------------------------------------------------------------------------
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
    analyzed = a._analyzed_frames().tolist()
    areas = np.asarray(a.residue_areas)
    n_rows = 0
    for k, row in enumerate(iter_csv(r.outdir / "frame_summary.csv")):
        assert k < a.n_frames, f"frame_summary.csv has more than {a.n_frames} rows"
        assert int(float(row["frame"])) == analyzed[k]
        assert int(float(row["n_surface_residues"])) == a.results.n_surface_residues[k]
        surface_rows = np.flatnonzero(areas[k] >= a.area_cutoff)
        expected = int(np.isin(surface_rows, a._start_res_row).sum())
        assert a.results.n_surface_residues[k] == expected
        n_blocked = int(float(row["n_blocked_pairs"]))
        mean_d = float(row["mean_blocked_distance"])
        assert n_blocked >= 0
        assert np.isnan(mean_d) == (n_blocked == 0)
        n_rows += 1
    assert n_rows == a.n_frames


def test_bfactors_npy(case, mode, runner):
    r = runner(case, mode)
    expected = np.asarray(r.analysis.all_counts)
    path = r.outdir / "b_factors.npy"
    try:
        saved = np.load(path, mmap_mode="r")      # header read; data paged on demand
    except ValueError:                             # e.g. zero-size array cannot be mapped
        saved = np.load(path)
    try:
        assert saved.shape == expected.shape, (saved.shape, expected.shape)
        if saved.size:
            row_bytes = max(1, saved[0].nbytes) if saved.ndim > 1 else saved.itemsize
            step = max(1, NPY_BLOCK_BYTES // row_bytes)
            for s in range(0, len(saved), step):
                np.testing.assert_array_equal(
                    saved[s:s + step], expected[s:s + step],
                    err_msg=f"b_factors.npy rows {s}..{s + step - 1}")
    finally:
        del saved                                  # release the map (matters on Windows)


def test_residue_summary_matches_counts(case, mode, runner, glyco):
    r = runner(case, mode)
    a = r.analysis
    keys = unique_ca_keys(glyco, a)
    counts, sasa = _last_counts(a), ca_sasa(a)
    got = {}
    for row in iter_csv(r.outdir / "residue_summary.csv"):
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
    count_of = {(c, rid): int(n) for (c, rid, _), n in zip(keys, _last_counts(a))}
    serials = {str(s) for s in glyco.pdb_atom_ids(a.gly_atoms)}
    per_residue, seen_in_residue = Counter(), {}
    for row in iter_csv(r.outdir / "excluded_glycan_atoms.csv"):
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
    for name in COMPARED_CSVS:
        assert_same_file(r.outdir / name, cli_dir / name, f"CLI vs library: {name}")


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------
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
    for name in COMPARED_CSVS:
        assert_same_file(serial.outdir / name, parallel.outdir / name,
                         f"serial vs parallel ({n} workers): {name}")


# --------------------------------------------------------------------------
# legacy tools (opt-in)
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.legacy
@pytest.mark.glyco_kind("structure")
def test_legacy_compare(case, compare_params, legacy_report):
    """compare.py gating checks with the resolved (and optionally varied) parameters.

    legacy_report() handles the skips (no compare.py, non-PDB input, no legacy
    glycan names), caches one run per (structure, parameter set), honours
    --glyco-legacy-reuse and feeds the terminal summary table.
    """
    r = legacy_report(case, compare_params)
    if r.returncode is None:
        pytest.fail(f"compare.py timed out\n{r.tail()}", pytrace=False)
    if r.report is None:
        pytest.fail(f"compare.py did not write a JSON summary "
                    f"(exit code {r.returncode})\n{r.tail()}", pytrace=False)
    checks = r.report.get("checks", {})
    failed = [label for label, value in checks.items() if not _check_passed(value)]
    if r.returncode != 0 or not r.report.get("overall") or failed:
        pytest.fail(f"[{r.params_id}] exit code {r.returncode}; failing checks: "
                    f"{failed or 'none reported'}\n{r.tail()}", pytrace=False)