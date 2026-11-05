""" Tests for the residue output mixin and its integration with GlycoAnalysis
and FreesasaAnalysis.
"""
import MDAnalysis as mda
import numpy as np
import pytest

from glycokit import sasa
from glycokit.glyco import GlycoAnalysis
from glycokit.mdafreesasa import FreesasaAnalysis  # imported so it registers
from glycokit.residue_output import ResidueOutputMixin
from glycokit.tests.conftest import LEGACY_PDB

# pylint: disable=protected-access

# Methods __init_subclass__ requires; keep in sync with residue_output.py.
REQUIRED_METHODS = ("_residue_column_atoms", "_residue_values")
EXPECTED_HOSTS = {GlycoAnalysis, FreesasaAnalysis}


def _run(cls, universe):
    """Build and run a small instance of each host class."""
    if cls is GlycoAnalysis:
        obj = GlycoAnalysis(universe, outdir="unused")
        obj.run(write=False)
        return obj
    if cls is FreesasaAnalysis:
        obj = FreesasaAnalysis(
            universe.select_atoms("protein and not name H*"), 1.4)
        obj.run()
        return obj
    pytest.fail(f"add a constructor for {cls.__name__} to _run()")


def _stub_hooks(names):
    return {name: (lambda self: None) for name in names}


@pytest.fixture(name="isolated_registry")
def fixture_isolated_registry(monkeypatch):
    """Copy of the registry, so classes defined in a test don't leak out."""
    reg = ResidueOutputMixin._registry
    monkeypatch.setattr(ResidueOutputMixin, "_registry", type(reg)(reg))
    return ResidueOutputMixin._registry


# ---------------------------------------------------------------- registry


def test_registry_contains_host_classes():
    # An empty registry would make the parametrized test below skip silently.
    assert EXPECTED_HOSTS <= set(ResidueOutputMixin._registry)


@pytest.mark.parametrize("missing", REQUIRED_METHODS)
def test_missing_hook_rejected_at_definition(missing, isolated_registry):
    hooks = [n for n in REQUIRED_METHODS if n != missing]
    with pytest.raises(TypeError, match=missing):
        type("Incomplete", (ResidueOutputMixin, ), _stub_hooks(hooks))
    assert not any(c.__name__ == "Incomplete" for c in isolated_registry)


def test_complete_subclass_registered(isolated_registry):
    cls = type(
        "Complete", (ResidueOutputMixin, ), _stub_hooks(REQUIRED_METHODS))
    assert cls in isolated_registry


def test_inherited_hooks_accepted(isolated_registry):
    base = type("Base", (ResidueOutputMixin, ), _stub_hooks(REQUIRED_METHODS))
    child = type("Child", (base, ), {})  # no hooks of its own
    assert child in isolated_registry


# ------------------------------------------------------- runtime contract


@pytest.mark.parametrize(
    "cls", ResidueOutputMixin._registry, ids=lambda c: c.__name__)
def test_hooks_resolve(cls, universe):
    obj = _run(cls, universe)
    cols, vals = obj._residue_column_atoms(), obj._residue_values()
    assert vals.ndim == 2 and vals.shape == (obj.n_frames, len(cols))
    # columns must belong to the analysed system, since writers set its
    # tempfactors
    assert cols.universe is universe


# ------------------------------------------------------------- outputs


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
            str(out), "-y", "--sel", "protein and resid 1:30",
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


@pytest.mark.parametrize(
    "cls", ResidueOutputMixin._registry, ids=lambda c: c.__name__)
def test_viewer_scripts_for_hosts(cls, universe, tmp_path):
    obj = _run(cls, universe)
    vmd = obj.write_vmd_user_script(tmp_path / "u.vmd").read_text()
    assert vmd.count("$sel set user {") == obj.n_frames

    n_cols = len(obj._residue_column_atoms())
    cxc = obj.write_chimerax_script(
        output=tmp_path / "v.cxc", values=np.arange(n_cols, dtype=float),
        attribute="test_value").read_text().splitlines()
    assert any(l.startswith("color byattribute test_value") for l in cxc)
    assert (tmp_path / "v.defattr").is_file()


def test_viewer_values_shape_checked(universe, tmp_path):
    obj = _run(GlycoAnalysis, universe)
    with pytest.raises(ValueError, match="expected"):
        obj.write_vmd_user_script(tmp_path / "x.vmd", values=[1.0, 2.0])


def test_viewer_scripts_need_run(universe, tmp_path):
    g = GlycoAnalysis(universe, outdir=str(tmp_path))
    with pytest.raises(RuntimeError):
        g.write_vmd_user_script()


@pytest.mark.parametrize(
    "cls", ResidueOutputMixin._registry, ids=lambda c: c.__name__)
def test_plot_timeseries_for_hosts(cls, universe, tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    obj = _run(cls, universe)
    assert obj.plot_timeseries(output=tmp_path / "a.png").stat().st_size > 0
    assert obj.plot_timeseries(
        residues=[0, 1], output=tmp_path / "b.png", x="time").is_file()


def test_plot_timeseries_rejects_bad_input(universe, tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    obj = _run(GlycoAnalysis, universe)
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match="reduce"):
        obj.plot_timeseries(output=out, reduce="max")
    with pytest.raises(ValueError, match="residues"):
        obj.plot_timeseries(output=out, residues=list(range(9)))
    with pytest.raises(ValueError, match="x must be"):
        obj.plot_timeseries(output=out, x="ns")


def test_glyco_write_plots(tmp_path, universe):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    GlycoAnalysis(universe, outdir=str(tmp_path), write_plots=True).run()
    assert (tmp_path / "glyco_timeseries.png").is_file()
    assert (tmp_path / "residue_histograms.png").is_file()
