"""
pytest configuration for glycokit tests with user-supplied inputs.

Examples
  pytest --glyco-structure a.pdb --glyco-structure b.pdb
  pytest --glyco-indir structures/ --glyco-in-types pdb,gro
  pytest --glyco-system top.psf,run1.dcd,run2.dcd --glyco-max-frames 5
  pytest --glyco-structure test_PROR.pdb --glyco-modes compat --glyco-dc 26.5
"""
import glob
import importlib
import os
import re
import warnings
from dataclasses import dataclass, field

import pytest

from glycokit.data.files import MDANALYSIS_LOGO

MODES = ("default", "compat-v2", "compat")

# pytest option -> (GlycoAnalysis keyword, glyco CLI flag)
_PASSTHROUGH = {
    "--glyco-bs": ("block_sel", "--bs"),
    "--glyco-sasa-env": ("sasa_env", "--sasa-env"),
    "--glyco-dc": ("distance_cutoff", "--dc"),
    "--glyco-ac": ("area_cutoff", "--ac"),
    "--glyco-cr": ("radius", "--cr"),
    "--glyco-pr": ("probe", "--pr"),
}


@pytest.fixture
def mdanalysis_logo_text() -> str:
    """Example fixture demonstrating how data files can be accessed"""
    with open(MDANALYSIS_LOGO, "r", encoding="utf8") as f:
        logo_text = f.read()
    return logo_text


@dataclass(frozen=True)
class Case:
    kind: str                   # "structure" or "trajectory"
    top: str
    traj: tuple = ()

    @property
    def id(self):
        name = os.path.basename(self.top)
        if self.traj:
            name += "+" + "+".join(os.path.basename(t) for t in self.traj)
        return name


@dataclass
class Run:
    analysis: object
    outdir: object
    frames: object
    warnings: list = field(default_factory=list)


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------

def pytest_addoption(parser):
    g = parser.getgroup("glycokit", "glycokit test inputs")
    g.addoption("--glyco-structure", action="append", default=[], metavar="PATH",
                help="single-frame structure file (repeatable)")
    g.addoption("--glyco-indir", action="append", default=[], metavar="DIR",
                help="directory of independent structures (repeatable)")
    g.addoption("--glyco-in-types", default="pdb", metavar="EXTS",
                help="comma-separated extensions used with --glyco-indir")
    g.addoption("--glyco-system", action="append", default=[], metavar="TOP,TRAJ...",
                help="topology followed by trajectory file(s), comma-separated "
                     "(repeatable)")
    g.addoption("--glyco-ps", default="protein", help="protein selection (--ps)")
    g.addoption("--glyco-gs", default="resname BGLC", help="glycan selection (--gs)")
    g.addoption("--glyco-bs", default=None, help="blocker selection (--bs)")
    g.addoption("--glyco-sasa-env", default=None, help="--sasa-env selection")
    g.addoption("--glyco-dc", type=float, default=None, help="--dc")
    g.addoption("--glyco-ac", type=float, default=None, help="--ac")
    g.addoption("--glyco-cr", type=float, default=None, help="--cr")
    g.addoption("--glyco-pr", type=float, default=None, help="--pr")
    g.addoption("--glyco-modes", default=",".join(MODES),
                help="comma-separated subset of: " + ", ".join(MODES))
    g.addoption("--glyco-max-frames", type=int, default=3,
                help="frames analysed per trajectory (evenly spaced; 0 = all)")
    g.addoption("--glyco-box-workers", type=int, default=2,
                help="worker processes for the --compat box test")
    g.addoption("--glyco-module", default="glycokit.glyco",
                help="import path of the module that defines GlycoAnalysis")
    g.addoption("--glyco-compare", default=None, metavar="PATH",
                help="path to compare.py; enables test_legacy_compare")


def _abs_existing(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise pytest.UsageError(f"glycokit tests: no such file: {path}")
    return path


def _collect_cases(config):
    opt = config.getoption
    cases = [Case("structure", _abs_existing(p)) for p in opt("--glyco-structure")]

    exts = [e.strip().lstrip(".") for e in opt("--glyco-in-types").split(",") if e.strip()]
    for d in opt("--glyco-indir"):
        if not os.path.isdir(d):
            raise pytest.UsageError(f"--glyco-indir: no such directory: {d}")
        for ext in exts:
            for p in sorted(glob.glob(os.path.join(d, f"*.{ext}"))):
                cases.append(Case("structure", os.path.abspath(p)))

    for spec in opt("--glyco-system"):
        parts = [s.strip() for s in spec.split(",") if s.strip()]
        if not parts:
            raise pytest.UsageError(f"--glyco-system: empty value {spec!r}")
        top, trajs = _abs_existing(parts[0]), tuple(_abs_existing(t) for t in parts[1:])
        cases.append(Case("trajectory" if trajs else "structure", top, trajs))

    return list(dict.fromkeys(cases))           # de-duplicate, keep order


def pytest_configure(config):
    for line in (
        "glyco_kind(*kinds): restrict a test to 'structure' and/or 'trajectory' inputs",
        "glyco_modes(*modes): restrict a test to some counting modes",
        "slow: expensive test",
        "legacy: runs the legacy GLYCO tools through compare.py",
    ):
        config.addinivalue_line("markers", line)

    modes = [m.strip() for m in config.getoption("--glyco-modes").split(",") if m.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise pytest.UsageError(f"--glyco-modes: unknown mode(s) {unknown}")
    config.glyco_modes = modes
    config.glyco_cases = _collect_cases(config)


def pytest_generate_tests(metafunc):
    config = metafunc.config
    if "case" in metafunc.fixturenames:
        cases = config.glyco_cases
        marker = metafunc.definition.get_closest_marker("glyco_kind")
        if marker:
            cases = [c for c in cases if c.kind in marker.args]
        metafunc.parametrize("case", [pytest.param(c, id=c.id) for c in cases])
    if "mode" in metafunc.fixturenames:
        modes = config.glyco_modes
        marker = metafunc.definition.get_closest_marker("glyco_modes")
        if marker:
            modes = [m for m in modes if m in marker.args]
        metafunc.parametrize("mode", modes)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def glyco(pytestconfig):
    """The module that defines GlycoAnalysis, load_universe and main."""
    return importlib.import_module(pytestconfig.getoption("--glyco-module"))


@pytest.fixture(scope="session")
def glyco_params(pytestconfig):
    """GlycoAnalysis keyword arguments shared by every run."""
    o = pytestconfig.getoption
    params = {"prot_sel": o("--glyco-ps"), "gly_sel": o("--glyco-gs")}
    for option, (keyword, _) in _PASSTHROUGH.items():
        if o(option) is not None:
            params[keyword] = o(option)
    return params


@pytest.fixture(scope="session")
def glyco_cli_args(pytestconfig):
    """The same parameters as glyco CLI arguments."""
    o = pytestconfig.getoption
    args = ["--ps", o("--glyco-ps"), "--gs", o("--glyco-gs")]
    for option, (_, flag) in _PASSTHROUGH.items():
        if o(option) is not None:
            args += [flag, str(o(option))]
    return args


def _has_record_types(glyco, universe):
    try:
        universe.atoms.record_types  # noqa: B018
        return True
    except (AttributeError, glyco.mda.exceptions.NoDataError):
        return False


def _default_frames(n_frames, max_frames):
    if n_frames <= 1 or max_frames <= 0 or max_frames >= n_frames:
        return None                                  # all frames
    import numpy as np
    return tuple(int(f) for f in np.unique(
        np.linspace(0, n_frames - 1, max_frames).round().astype(int)))


@pytest.fixture(scope="session")
def runner(glyco, glyco_params, pytestconfig, tmp_path_factory):
    """
    runner(case, mode, frames="auto", universe=None, **overrides) -> Run

    Runs GlycoAnalysis once per (case, mode, frames, overrides) and caches
    it. Pass `universe` to run on a prepared Universe (not cached).
    """
    cache = {}
    max_frames = pytestconfig.getoption("--glyco-max-frames")
    box_workers = pytestconfig.getoption("--glyco-box-workers")

    def run(case, mode, frames="auto", universe=None, **overrides):
        frames_key = frames if frames == "auto" or frames is None else tuple(frames)
        key = (case, mode, frames_key, tuple(sorted(overrides.items())))
        if universe is None and key in cache:
            return cache[key]

        u = universe if universe is not None else glyco.load_universe(
            case.top, list(case.traj) or None)

        if (mode != "default" and not _has_record_types(glyco, u)
                and "sasa_env" not in glyco_params):
            pytest.skip(f"{mode} needs PDB record types for its FreeSASA atom set; "
                        "this topology has none (give --glyco-sasa-env)")

        if frames == "auto":
            frames = _default_frames(u.trajectory.n_frames, max_frames)
        frames = list(frames) if frames is not None else None

        name = re.sub(r"[^\w.-]", "_", f"{case.id}-{mode}")[:60]
        outdir = tmp_path_factory.mktemp(name)
        kwargs = {
            **glyco_params,
            "outdir": str(outdir),
            "write_excluded": True,
            "compat": mode == "compat",
            "compat_v2": mode == "compat-v2",
            "box_workers": box_workers,
            **overrides,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analysis = glyco.GlycoAnalysis(u, **kwargs)
            analysis.run(frames=frames)

        result = Run(analysis, outdir, frames, [str(w.message) for w in caught])
        if universe is None:
            cache[key] = result
        return result

    return run
