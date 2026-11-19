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


# @pytest.mark.parametrize(
#     "cls", ResidueOutputMixin._registry, ids=lambda c: c.__name__)
# def test_plot_timeseries_for_hosts(cls, universe, tmp_path):
#     mpl = pytest.importorskip("matplotlib")
#     mpl.use("Agg")
#     obj = _run(cls, universe)
#     assert obj.plot_timeseries(output=tmp_path / "a.png").stat().st_size > 0
#     assert obj.plot_timeseries(
#         residues=[0, 1], output=tmp_path / "b.png", x="frame").is_file()

# def test_plot_timeseries_rejects_bad_input(universe, tmp_path):
#     mpl = pytest.importorskip("matplotlib")
#     mpl.use("Agg")
#     obj = _run(GlycoAnalysis, universe)
#     out = tmp_path / "x.png"
#     with pytest.raises(ValueError, match="reduce"):
#         obj.plot_timeseries(output=out, reduce="max")
#     with pytest.raises(ValueError, match="residues"):
#         obj.plot_timeseries(output=out, residues=list(range(9)))
#     with pytest.raises(ValueError, match="x must be"):
#         obj.plot_timeseries(output=out, x="ns")


def test_glyco_write_plots(tmp_path, universe):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    GlycoAnalysis(universe, outdir=str(tmp_path), write_plots=True).run()
    assert (tmp_path / "glyco_timeseries.png").is_file()
    assert (tmp_path / "residue_histograms.png").is_file()


def test_plot_x_uses_frames_without_time_info(universe):
    obj = _run(GlycoAnalysis, universe)  # PDB: reader time unit is None
    _, label = obj._plot_x("auto")
    assert label == "trajectory frame"
    with pytest.raises(ValueError, match="no time information"):
        obj._plot_x("time")


def test_plot_x_uses_time_when_available(universe, monkeypatch):
    obj = _run(GlycoAnalysis, universe)
    monkeypatch.setattr(obj, "n_frames", 3)
    monkeypatch.setattr(obj, "times", np.array([0.0, 10.0, 20.0]))
    monkeypatch.setattr(
        obj.u.trajectory, "units", {
            "length": "Angstrom",
            "time": "ps"
        })
    xs, label = obj._plot_x("auto")
    assert label == "time (ps)" and xs.tolist() == [0.0, 10.0, 20.0]


def _chained(tmp_path, duplicate=True):
    u0 = mda.Universe(str(LEGACY_PDB))
    ref = u0.atoms.positions.copy()
    paths = []
    for i, ks in enumerate([range(0, 3), range(2 if duplicate else 3, 6)]):
        path = tmp_path / f"part{i}.xtc"
        with mda.Writer(str(path), u0.atoms.n_atoms) as w:
            for k in ks:
                u0.atoms.positions = ref + k
                w.write(u0.atoms)
        paths.append(str(path))
    return mda.Universe(str(LEGACY_PDB), paths)


def test_concatenated_axis_drops_duplicate_first_frame(tmp_path):
    obj = _run(FreesasaAnalysis, _chained(tmp_path))
    x, _, keep, label, _ = obj._plot_axis("concatenated")
    assert keep.tolist() == [True, True, True, False, True, True]
    assert x[keep].tolist() == [0, 1, 2, 3, 4]
    assert "1 duplicate frame(s) removed" in label


def test_separate_axis_keeps_every_part(tmp_path):
    obj = _run(FreesasaAnalysis, _chained(tmp_path))
    x, seg, keep, label, names = obj._plot_axis("separate")
    assert keep.all() and x.tolist() == [0, 1, 2, 0, 1, 2]
    assert seg.tolist() == [0, 0, 0, 1, 1, 1]
    assert label == "frame within segment" and len(names) == 2


def test_no_duplicate_without_overlap(tmp_path):
    obj = _run(FreesasaAnalysis, _chained(tmp_path, duplicate=False))
    x, _, keep, label, _ = obj._plot_axis("concatenated")
    assert keep.all() and x.tolist() == list(range(6))
    assert "duplicate" not in label


@pytest.mark.parametrize("mode", ["concatenated", "separate"])
def test_plot_top_residues(tmp_path, mode):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    obj = _run(FreesasaAnalysis, _chained(tmp_path))
    out = obj.plot_top_residues(
        output=tmp_path / f"{mode}.png", trajectories=mode)
    assert out.stat().st_size > 0
    assert len(obj.top_residues(9)) == min(9, len(obj._residue_column_atoms()))


from glycokit.residue_output import _time_unit


def _chained_timed(tmp_path, restart):
    """Two XTC parts, dt = 10 ps; part 2 repeats part 1's last frame and
    either restarts its clock at 0 or continues it."""
    u0 = mda.Universe(str(LEGACY_PDB))
    ref = u0.atoms.positions.copy()
    paths = []
    for i, ks in enumerate([range(0, 3), range(2, 5)]):
        path = tmp_path / f"t{i}.xtc"
        with mda.Writer(str(path), u0.atoms.n_atoms) as w:
            for j, k in enumerate(ks):
                u0.atoms.positions = ref + k
                u0.trajectory.ts.time = 10.0 * (j if restart and i else k)
                w.write(u0.atoms)
        paths.append(str(path))
    return mda.Universe(str(LEGACY_PDB), paths)


@pytest.mark.parametrize("restart", [True, False])
def test_concatenated_time_axis_has_no_jump(tmp_path, restart):
    obj = _run(FreesasaAnalysis, _chained_timed(tmp_path, restart))
    axis = obj._plot_axis("concatenated")
    assert axis.keep.tolist() == [True, True, True, False, True, True]
    assert axis.x[axis.keep].tolist() == [0, 10, 20, 30, 40]
    assert axis.label.startswith("time (ps, concatenated)")


@pytest.mark.parametrize(
    "span, unit, factor",
    [(500.0, "ps", 1.0), (5e4, "ns", 1e-3), (2e6, "µs", 1e-6)])
def test_time_unit(span, unit, factor):
    f, label = _time_unit(span)
    assert label == unit and f == pytest.approx(factor)


@pytest.mark.parametrize("mode", ["concatenated", "separate"])
def test_plot_timeseries_chained(tmp_path, mode):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    obj = _run(FreesasaAnalysis, _chained_timed(tmp_path, restart=True))
    assert obj.plot_timeseries(
        output=tmp_path / f"{mode}.png", trajectories=mode).stat().st_size > 0
    assert obj.plot_timeseries(
        residues=[0, 1], output=tmp_path / f"r_{mode}.png",
        trajectories=mode).is_file()


from glycokit.residue_output import _part_labels


@pytest.mark.parametrize(
    "paths, expected, renamed", [
        (["a/x.xtc", "b/y.xtc"], ["x.xtc", "y.xtc"], False),
        (["run1/traj.xtc", "run2/traj.xtc"], ["run_1", "run_2"], True),
        (["a.xtc", "b.xtc", "a.xtc"], ["run_1", "run_2", "run_3"], True),
        (["a.xtc", None], ["run_1", "run_2"], True),
    ])
def test_part_labels(paths, expected, renamed):
    assert _part_labels(paths) == (expected, renamed)


def test_same_file_names_get_run_labels(tmp_path):
    u0 = mda.Universe(str(LEGACY_PDB))
    paths = []
    for i in range(2):
        d = tmp_path / f"run{i + 1}"
        d.mkdir()
        with mda.Writer(str(d / "traj.xtc"), u0.atoms.n_atoms) as w:
            for k in range(3):
                u0.atoms.positions += 1.0  # distinct frames, no duplicate
                w.write(u0.atoms)
        paths.append(str(d / "traj.xtc"))
    obj = _run(FreesasaAnalysis, mda.Universe(str(LEGACY_PDB), paths))
    axis = obj._plot_axis("separate")
    assert axis.names == ["run_1", "run_2"] and axis.renamed

    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    obj.outdir = str(tmp_path)
    obj.plot_timeseries(output=tmp_path / "ts.png", trajectories="separate")
    assert (tmp_path / "trajectory_runs.csv").read_text().splitlines()[1] \
        .startswith("run_1,")


@pytest.mark.parametrize("mode", ["concatenated", "separate"])
@pytest.mark.parametrize("n", [4, 9])
def test_plot_top_residues_with_margins(tmp_path, mode, n, monkeypatch):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    figs = []
    monkeypatch.setattr(plt, "close", figs.append)  # keep the figure
    obj = _run(FreesasaAnalysis, _chained(tmp_path))
    out = obj.plot_top_residues(
        output=tmp_path / f"{mode}{n}.png", n=n, trajectories=mode)
    assert out.stat().st_size > 0

    shown = min(n, len(obj._residue_column_atoms()))
    visible = [a for a in figs[0].axes if a.get_visible()]
    assert len(visible) == 2 * shown  # one histogram per panel
    hists = [a for a in visible if a.patches]
    assert len(hists) == shown
