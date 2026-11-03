import MDAnalysis as mda
import numpy as np
import pytest

from glycokit import sasa
from glycokit.glyco import GlycoAnalysis
from glycokit.residue_output import ResidueOutputMixin


@pytest.mark.parametrize(
    "cls", ResidueOutputMixin._registry, ids=lambda c: c.__name__)
def test_all_hooks_resolve(cls, universe):
    obj = _run(cls, universe)  # constructor helper from before
    for name in ResidueOutputMixin._HOOK_SPECS:  # new hooks are tested automatically
        obj._hook(name)
    cols, vals = obj._hook("columns"), obj._hook("values")
    assert vals.shape == (obj.n_frames, len(cols))
    assert obj._hook("universe") is cols.universe


def _run(cls, universe):
    """Build and run a small instance of each host class."""
    if cls.__name__ == "GlycoAnalysis":
        return cls(universe, outdir="unused").run(write=False)
    if cls.__name__ == "FreesasaAnalysis":
        return cls(universe.select_atoms("protein and not name H*"), 1.4).run()
    pytest.fail(f"add a constructor for {cls.__name__} to _run()")


@pytest.mark.parametrize(
    "cls", ResidueOutputMixin._registry, ids=lambda c: c.__name__)
def test_hooks_resolve(cls, universe):
    obj = _run(cls, universe)
    cols, vals = obj._residue_column_atoms(), obj._residue_values()
    assert vals.ndim == 2 and vals.shape == (obj.n_frames, len(cols))
    assert obj._out_universe is cols.universe


def test_glyco_run_writes_outputs(tmp_path, universe):
    g = GlycoAnalysis(universe, outdir=str(tmp_path))
    g.run()  # 1a: write_bfactors in _conclude
    assert (tmp_path / "b_factors.pdb").is_file()


def test_glyco_sasa_cli(tmp_path):  # 1b + 1c
    out = tmp_path / "out"
    assert sasa.main(
        [
            "-a",
            str(LEGACY_PDB), "-o",
            str(out), "-y", "--sel", "protein and resid 1:30"
        ]) == 0
    for name in sasa.OUTPUT_NAMES:
        assert (out / name).is_file()


def test_stats_pdb_has_stats(tmp_path, universe):  # section 2
    g = GlycoAnalysis(universe, outdir=str(tmp_path), write_stats=True)
    g.run()
    u = mda.Universe(str(tmp_path / "b_factors_stats.pdb"))
    mode = g.residue_statistics()["mode"]
    np.testing.assert_allclose(
        u.select_atoms("name CA").tempfactors[:len(mode)], mode, atol=0.01)
