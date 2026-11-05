import MDAnalysis as mda
import numpy as np
import pytest

from glycokit.mdafreesasa import FreesasaAnalysis
from glycokit.tests.conftest import LEGACY_PDB
from glycokit.utils import heavy_atoms


def _small_traj(small_universe, n=6):
    rng = np.random.default_rng(0)
    xyz = small_universe.atoms.positions
    return mda.Merge(small_universe.atoms).load_new(
        np.stack([xyz + rng.normal(0, 0.2, xyz.shape) for _ in range(n)]),
        order="fac")


def test_parallel_matches_serial(small_universe):
    ag = heavy_atoms(_small_traj(small_universe).atoms)
    ser = FreesasaAnalysis(ag, 1.4, keep_atom_areas=True).run()
    par = FreesasaAnalysis(ag, 1.4,
                           keep_atom_areas=True).run_parallel(2, n_parts=3)
    for key in ("residue_areas", "total_area", "atom_areas"):
        np.testing.assert_array_equal(ser.results[key], par.results[key])
    np.testing.assert_array_equal(ser.frames, par.frames)


def test_n_selected_frames(small_universe):
    fsa = FreesasaAnalysis(heavy_atoms(_small_traj(small_universe).atoms), 1.4)
    assert fsa.n_selected_frames() == 6
    assert fsa.n_selected_frames(step=2) == 3
    assert fsa.n_selected_frames(frames=[0, 5]) == 2
    assert fsa.n_selected_frames(frames=[True, False] * 3) == 3
    with pytest.raises(ValueError, match="step"):
        fsa.n_selected_frames(step=0)


def test_threads_clamped_once(small_universe):
    with pytest.warns(UserWarning, match="single-threaded"):
        fsa = FreesasaAnalysis(small_universe.atoms, 1.4, n_threads=8)
    assert fsa.n_threads == 1  # pickled workers rebuild params silently


@pytest.mark.parametrize(
    "extra", [["--step", "0"], ["--frames", "-1"], ["--frames", "99"]])
def test_glyco_sasa_bad_frames(tmp_path, extra):
    from glycokit import sasa  # pylint: disable=import-outside-toplevel
    with pytest.raises(SystemExit) as exc:
        sasa.main(
            ["-a",
             str(LEGACY_PDB), "-o",
             str(tmp_path / "o"), "-y", *extra])
    assert exc.value.code == 2  # argparse error, not a traceback
