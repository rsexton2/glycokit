"""
<<<<<<< HEAD
pytest configuration for glycokit tests with user-supplied inputs.

Shared parameters resolve as: pytest option > GLYCO_COMPARE_<NAME> > built-in default.
compare.py reads the same environment variables, and the legacy test also
passes the resolved values to it explicitly.

Examples (use '=' for options whose value is a path):
  pytest --glyco-structure=a.pdb
  pytest --glyco-system=top.psf,run1.dcd --glyco-max-frames=5
  GLYCO_COMPARE_DISTANCE_CUTOFF=26.5 pytest -m legacy \
      --glyco-structure=test_PROR.pdb --glyco-compare=/path/compare.py
  pytest -m legacy --glyco-structure=test_PROR.pdb --glyco-compare=/path/compare.py \
      --glyco-compare-vary="dc=26.5,30" --glyco-compare-vary="cr=1.0,1.4"
"""
import dataclasses
import glob
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pytest

try:                                  # tests/ as a package
    from . import glyco_cases
except ImportError:                   # tests/ as a plain directory
    import glyco_cases

MODES = glyco_cases.MODES             # replaces the old MODES tuple

ENV_PREFIX = "GLYCO_COMPARE_"

# shared by glycokit and compare.py:
# key -> (help, env name, type, GlycoAnalysis keyword, glyco flag, compare.py flag)
SHARED = {
    "ps": ("protein selection (--ps)", "PROTEIN_SELECTION", str,
           "prot_sel", "--ps", "--protein-selection"),
    "gs": ("glycan selection (--gs)", "GLYCAN_SELECTION", str,
           "gly_sel", "--gs", "--glycan-selection"),
    "dc": ("distance cutoff (--dc)", "DISTANCE_CUTOFF", float,
           "distance_cutoff", "--dc", "--distance-cutoff"),
    "ac": ("area cutoff (--ac)", "AREA_CUTOFF", float,
           "area_cutoff", "--ac", "--area-cutoff"),
    "cr": ("cylinder radius (--cr)", "CYLINDER_RADIUS", float,
           "radius", "--cr", "--cylinder-radius"),
    "pr": ("probe radius (--pr)", "PROBE", float,
           "probe", "--pr", "--probe"),
}

# glycokit only (the legacy tools have no equivalent): option -> (keyword, flag)
GLYCO_ONLY = {
    "--glyco-bs": ("block_sel", "--bs"),
    "--glyco-sasa-env": ("sasa_env", "--sasa-env"),
}

# keys accepted by --glyco-compare-vary -> (type, compare.py flag)
VARY_KEYS = {
    "dc": (float, "--distance-cutoff"),
    "ac": (float, "--area-cutoff"),
    "cr": (float, "--cylinder-radius"),
    "pr": (float, "--probe"),
    "box_workers": (int, "--box-workers"),
    "sasa_tolerance": (float, "--sasa-tolerance"),
}

LEGACY_CHECKS = (
    ("v1-counts",   "compat versus original GLYCO counts"),
    ("v1-sasa",     "compat versus original GLYCO SASA"),
    ("v1-excluded", "compat versus original GLYCO excluded atoms"),
    ("v2-counts",   "compat-v2 versus GLYCO-2 counts"),
    ("v2-sasa",     "compat-v2 versus GLYCO-2 SASA"),
    ("v2-excluded", "compat-v2 versus GLYCO-2 excluded atoms"),
)

CASE_ONLY = {"bs": ("block_sel", "--bs"),
             "sasa_env": ("sasa_env", "--sasa-env")}


_DUR_KEY = "glycokit/group_durations"
_timings = defaultdict(float)     # group -> seconds, collected this session
_report_groups = {}


def _group_key(item):
    """All tests that share one cached runner(case, mode) form a group."""
    cs = getattr(item, "callspec", None)
    case = cs.params.get("case") if cs else None
    case_id = getattr(case, "id", None)                 # None for NOTSET
    if case_id is None:
        return item.nodeid
    return f"{case_id}|{cs.params.get('mode', '')}"


def pytest_runtest_logreport(report):
    # setup + call + teardown, so the group that builds the analysis is charged for it
    _timings[_group_key_from_report(report)] += report.duration


def _group_key_from_report(report):
    return _report_groups.get(report.nodeid, report.nodeid)


def pytest_itemcollected(item):
    _report_groups[item.nodeid] = _group_key(item)


def pytest_sessionfinish(session):
    if hasattr(session.config, "workerinput"):     # xdist worker: controller writes
        return
    if not _timings:
        return
    stored = session.config.cache.get(_DUR_KEY, {})
    stored.update({g: round(t, 3) for g, t in _timings.items()})
    session.config.cache.set(_DUR_KEY, stored)


def _attach_case_settings(config):
    path = config.getoption("--glyco-cases") or _env("CASES")
    strict = config.getoption("--glyco-cases-strict")
    try:
        entries = glyco_cases.load_manifest(path) if path else []
        cases = []
        for case in config.glyco_cases:
            values, sources = glyco_cases.overrides_for(case.top, entries)
            if strict and "gs" not in values:
                raise glyco_cases.CaseConfigError(
                    f"{case.id}: no glycan selection 'gs' in {path or 'a case file'} "
                    f"or {case.top}{glyco_cases.SIDECAR_SUFFIX}")
            cases.append(dataclasses.replace(
                case, overrides=tuple(sorted(values.items())), sources=tuple(sources)))
    except glyco_cases.CaseConfigError as exc:
        raise pytest.UsageError(f"glycokit case settings: {exc}") from exc

    unused = [e.label for e in entries if not e.used]
    if unused and strict:
        raise pytest.UsageError(f"--glyco-cases-strict: entries match no structure: {unused}")
    config.glyco_cases = cases
    config.glyco_case_file = path
    config.glyco_unused_case_entries = unused


def resolve_case_settings(config, case):
    """Global settings merged with the per-structure overrides of `case`."""
    s = dict(config.glyco_settings)                     # ps, gs, dc, ac, cr, pr
    s["bs"] = config.getoption("--glyco-bs")
    s["sasa_env"] = config.getoption("--glyco-sasa-env")
    over = case.settings
    s.update({k: v for k, v in over.items() if k in SHARED or k in CASE_ONLY})

    if "legacy_glycans" in over:
        names = over["legacy_glycans"]
    elif "gs" in over:                                  # derive from THIS structure's gs
        names = glyco_cases.legacy_names_from_selection(over["gs"])
    else:
        names = config.glyco_legacy_glycans
    s["legacy_glycans"] = tuple(names)
    s["modes"] = over.get("modes")
    return s

@dataclass
class LegacyReport:
    case: object
    params_id: str
    argv: list
    returncode: object          # int, or None after a timeout
    output: str
    report: object              # parsed JSON summary, or None
    reused: bool = False

    def tail(self, n=4000):
        return f"command: {shlex.join(self.argv)}\n\n{self.output[-n:]}"


def _file_digest(path, h):
    if path and os.path.isfile(path):
        h.update(Path(path).read_bytes())


@pytest.fixture(scope="session")
def legacy_report(pytestconfig, compare_base_args, case_settings,
                  tmp_path_factory):
    """
    legacy_report(case, params) -> LegacyReport

    Runs compare.py once per (structure, parameter set) and caches the result for
    the session; with --glyco-legacy-reuse also across sessions.
    """
    script = pytestconfig.glyco_compare_script
    timeout = pytestconfig.getoption("--glyco-compare-timeout")
    keep = pytestconfig.getoption("--glyco-compare-keep")
    store = (pytestconfig.cache.mkdir("glyco_legacy")
             if pytestconfig.getoption("--glyco-legacy-reuse") else None)
    module = pytestconfig.getoption("--glyco-module")
    spec = importlib.util.find_spec(module)
    module_file = spec.origin if spec else None
    flags = pytestconfig.glyco_vary_flags
    results = {}

    def run(case, params):
        if not script:
            pytest.skip("give --glyco-compare=PATH or GLYCO_COMPARE_SCRIPT")
        if not case.top.lower().endswith(".pdb"):
            pytest.skip("the legacy GLYCO tools read PDB files only")
        if not pytestconfig.glyco_legacy_glycans:
            pytest.skip("cannot derive legacy glycan names; give "
                        "--glyco-legacy-glycans or GLYCO_COMPARE_GLYCANS")
        if not case_settings(case)["legacy_glycans"]:
            pytest.skip("cannot derive legacy glycan names from this "
                        "structure's gs; set legacy_glycans for it in the case "
                        "file")

        params_id = "-".join(f"{k}={v}" for k, v in params.items()) or "base"
        key = (case, params_id)
        if key in results:
            return results[key]

        extra = []
        for name, value in params.items():          # grid values override base
            extra += [flags[name], str(value)]
        argv = [sys.executable, script, "--pdb", case.top, *compare_base_args(case), *extra]
        if keep:
            argv.append("--keep-output")

        cached = None
        if store is not None:
            h = hashlib.sha256()
            for path in (case.top, script, module_file):
                _file_digest(path, h)
            h.update("\0".join(argv[4:]).encode())   # args without interpreter/paths
            cached = store / f"{h.hexdigest()[:32]}.json"

        if cached is not None and cached.is_file():
            result = LegacyReport(case, params_id, argv, 0, "(reused cached report)",
                                  json.loads(cached.read_text()), reused=True)
        else:
            summary = tmp_path_factory.mktemp("legacy") / "summary.json"
            try:
                proc = subprocess.run(argv + ["--json-summary", str(summary)],
                                      cwd=os.path.dirname(script), capture_output=True,
                                      text=True, timeout=timeout)
                returncode, output = proc.returncode, proc.stdout + proc.stderr
            except subprocess.TimeoutExpired as exc:
                returncode = None
                output = f"timed out after {timeout} s\n{exc.stdout or ''}{exc.stderr or ''}"
            report = json.loads(summary.read_text()) if summary.is_file() else None
            result = LegacyReport(case, params_id, argv, returncode, output, report)
            if cached is not None and report is not None:
                cached.write_text(json.dumps(report))

        results[key] = result
        pytestconfig.glyco_legacy_results[(case.id, params_id)] = result
        return result

    return run


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    results = getattr(config, "glyco_legacy_results", {})
    if not results:
        return
    tr = terminalreporter
    tr.section("glycokit --compat / --compat-v2 versus GLYCO v1 / GLYCO-2")
    width = max(len(f"{c} [{p}]") for c, p in results) + 2
    tr.write_line(f"{'structure':<{width}}" + "".join(f"{s:>13}" for s, _ in LEGACY_CHECKS))
    for (case_id, params_id), r in sorted(results.items()):
        cells = []
        for _, label in LEGACY_CHECKS:
            if r.report is None:
                cells.append("ERROR")
            else:
                details = r.report.get("checks", {}).get(label)
                cells.append("MISSING" if details is None
                             else "PASS" if details["passed"] else "FAIL")
        reused = " (reused)" if r.reused else ""
        tr.write_line(f"{case_id + ' [' + params_id + ']':<{width}}"
                      + "".join(f"{c:>13}" for c in cells) + reused)


@dataclass(frozen=True)
class Case:
    kind: str                         # "structure" or "trajectory"
    top: str
    traj: tuple = ()
    overrides: tuple = ()             # sorted (key, value) pairs; part of the cache key
    sources: tuple = field(default=(), compare=False)

    @property
    def settings(self):
        return dict(self.overrides)

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


def _env(name, cast=str, default=None):
    raw = os.environ.get(ENV_PREFIX + name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise pytest.UsageError(f"{ENV_PREFIX}{name}={raw!r}: {exc}") from exc


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
                help="topology followed by trajectory file(s), comma-separated")

    for key, (text, env, typ, *_rest) in SHARED.items():
        g.addoption(f"--glyco-{key}", type=typ, default=None,
                    help=f"{text}; default ${ENV_PREFIX}{env}")
    g.addoption("--glyco-bs", default=None, help="blocker selection (--bs)")
    g.addoption("--glyco-sasa-env", default=None, help="--sasa-env selection")
    g.addoption("--glyco-legacy-glycans", default=None, metavar="NAMES",
                help=f"comma-separated glycan names for the legacy tools; default "
                     f"${ENV_PREFIX}GLYCANS, else derived from 'resname ...' in --glyco-gs")
    g.addoption("--glyco-box-workers", type=int, default=None,
                help=f"--box-workers; default ${ENV_PREFIX}BOX_WORKERS, else 2")

    g.addoption("--glyco-modes", default=",".join(MODES),
                help="comma-separated subset of: " + ", ".join(MODES))
    g.addoption("--glyco-max-frames", type=int, default=3,
                help="frames analysed per trajectory (evenly spaced; 0 = all)" )
    g.addoption("--glyco-module", default="glycokit.glyco",
                help="import path of the module that defines GlycoAnalysis")
    g.addoption("--glyco-cases", default=None, metavar="FILE",
                help=f"TOML/JSON file with per-structure settings ([[case]] entries); "
                     f"default ${ENV_PREFIX}CASES. Sidecars <structure>"
                     f"{glyco_cases.SIDECAR_SUFFIX} are always read")
    g.addoption("--glyco-cases-strict", action="store_true",
                help="error if a structure has no glycan selection (gs) of its "
                "own, or a [[case]] entry matches no structure")
    g.addoption("--glyco-order", choices=["none", "fast-first", "slow-first"],
                     default="none",
                     help="order test groups by recorded duration")

    c = parser.getgroup("glyco-compare", "legacy comparison through compare.py")
    c.addoption("--glyco-compare", default=None, metavar="PATH",
                help=f"path to compare.py; default ${ENV_PREFIX}SCRIPT")
    c.addoption("--glyco-compare-vary", action="append", default=[], metavar="KEY=V1,V2",
                help="run compare.py for every value (repeatable, cartesian product; "
                     f"also ${ENV_PREFIX}VARY='dc=26.5,30;cr=1.0,1.4'). Keys: "
                     + ", ".join(VARY_KEYS))
    c.addoption("--glyco-compare-arg", action="append", default=[], metavar="ARG",
                help="extra argument passed to compare.py, e.g. "
                     "--glyco-compare-arg=--no-normalize (repeatable)")
    c.addoption("--glyco-compare-installed", action="store_true",
                help="let compare.py run the installed 'glyco' command instead of "
                     "'python -m <--glyco-module>'")
    c.addoption("--glyco-compare-keep", action="store_true",
                help="pass --keep-output to compare.py")
    c.addoption("--glyco-compare-timeout", type=float, default=None,
                help="timeout per compare.py run in seconds")
    c.addoption("--glyco-legacy-reuse", action="store_true",
                help="reuse compare.py reports from earlier runs (pytest cache"
                "when the PDB, parameters, compare.py and the glyco module are"
                "unchanged")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

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

    return list(dict.fromkeys(cases))


def _resolve_settings(config):
    opt = config.getoption
    values = {}
    for key, (_text, env, typ, *_rest) in SHARED.items():
        value = opt(f"--glyco-{key}")
        values[key] = value if value is not None else _env(env, typ)

    glycans = opt("--glyco-legacy-glycans") or _env("GLYCANS")
    names = tuple(n for n in re.split(r"[,\s]+", glycans or "") if n)
    if values["gs"] is None:
        values["gs"] = "resname " + " ".join(names) if names else "resname BGLC"
    if values["ps"] is None:
        values["ps"] = "protein"
    if not names:
        m = re.fullmatch(r"\s*resname\s+([\w\s]+?)\s*", values["gs"])
        names = tuple(m.group(1).split()) if m else ()

    box_workers = opt("--glyco-box-workers")
    if box_workers is None:
        box_workers = _env("BOX_WORKERS", int, 2)
    if box_workers < 1:
        raise pytest.UsageError("box workers must be >= 1")
    return values, names, box_workers


def _parse_vary(specs):
    axes = {}
    for spec in specs:
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            key, sep, raw = part.partition("=")
            key = key.strip().replace("-", "_")
            if not sep or key not in VARY_KEYS:
                raise pytest.UsageError(
                    f"--glyco-compare-vary: expected KEY=V1,V2 with KEY in "
                    f"{sorted(VARY_KEYS)}, got {part!r}")
            cast = VARY_KEYS[key][0]
            try:
                values = [cast(v) for v in raw.split(",") if v.strip()]
            except ValueError as exc:
                raise pytest.UsageError(f"--glyco-compare-vary {part!r}: {exc}") from exc
            if not values:
                raise pytest.UsageError(f"--glyco-compare-vary {part!r}: no values")
            axes[key] = list(dict.fromkeys(values))
    return axes


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

    (config.glyco_settings, config.glyco_legacy_glycans,
     config.glyco_box_workers) = _resolve_settings(config)
    _attach_case_settings(config)

    script = config.getoption("--glyco-compare") or _env("SCRIPT")
    config.glyco_compare_script = _abs_existing(script) if script else None

    vary_specs = list(config.getoption("--glyco-compare-vary"))
    if _env("VARY"):
        vary_specs.insert(0, _env("VARY"))      # pytest options override the env
    config.glyco_compare_axes = _parse_vary(vary_specs)
    config.glyco_vary_flags = {k: flag for k, (_t, flag) in VARY_KEYS.items()}
    config.glyco_legacy_results = {}


def pytest_report_header(config):
    s = config.glyco_settings
    shown = ", ".join(f"{k}={s[k]!r}" for k in SHARED if s[k] is not None)
    lines = [f"glycokit settings: {shown}, box_workers={config.glyco_box_workers}"]
    if config.glyco_compare_axes:
        lines.append(f"compare.py grid: {config.glyco_compare_axes}")

    if config.glyco_case_file:
        lines.append(f"per-structure settings: {config.glyco_case_file}")
    for c in config.glyco_cases:
        if c.overrides:
            shown = ", ".join(f"{k}={v!r}" for k, v in c.overrides)
            lines.append(f"  {c.id}: {shown}  [{', '.join(c.sources)}]")
    if config.glyco_unused_case_entries:
        lines.append(f"  WARNING: entries matching no structure: "
                     f"{config.glyco_unused_case_entries}")

    return lines


def _case_marks(case):
    s = case.settings
    marks = []
    if "skip" in s:
        marks.append(pytest.mark.skip(reason=s["skip"]))
    if "xfail" in s:
        marks.append(pytest.mark.xfail(reason=s["xfail"], strict=False))
    return marks


def pytest_generate_tests(metafunc):
    config = metafunc.config
    names = metafunc.fixturenames

    modes = config.glyco_modes
    mode_marker = metafunc.definition.get_closest_marker("glyco_modes")
    if mode_marker:
        modes = [m for m in modes if m in mode_marker.args]

    if "case" in names:
        cases = config.glyco_cases
        kind_marker = metafunc.definition.get_closest_marker("glyco_kind")
        if kind_marker:
            cases = [c for c in cases if c.kind in kind_marker.args]
        if "mode" in names:
            params = [pytest.param(c, m, id=f"{c.id}-{m}", marks=_case_marks(c))
                      for c in cases for m in modes
                      if c.settings.get("modes") is None or m in c.settings["modes"]]
            metafunc.parametrize(("case", "mode"), params)
        else:
            metafunc.parametrize(
                "case", [pytest.param(c, id=c.id, marks=_case_marks(c)) for c in cases])
    elif "mode" in names:
        metafunc.parametrize("mode", modes)

    if "compare_params" in names:
        axes = config.glyco_compare_axes
        keys = list(axes)
        combos = [dict(zip(keys, values))
                  for values in itertools.product(*(axes[k] for k in keys))]
        ids = ["-".join(f"{k}={v}" for k, v in c.items()) or "base" for c in combos]
        metafunc.parametrize("compare_params", combos, ids=ids)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def glyco(pytestconfig):
    """The module that defines GlycoAnalysis, load_universe and main."""
    return importlib.import_module(pytestconfig.getoption("--glyco-module"))


@pytest.fixture(scope="session")
def case_settings(pytestconfig):
    """case_settings(case) -> merged settings of one structure."""
    return lambda case: resolve_case_settings(pytestconfig, case)


@pytest.fixture(scope="session")
def glyco_params(case_settings):
    """glyco_params(case) -> GlycoAnalysis keyword arguments."""
    def get(case):
        s = case_settings(case)
        params = {spec[3]: s[key] for key, spec in SHARED.items() if s[key] is not None}
        params.update({kw: s[key] for key, (kw, _f) in CASE_ONLY.items()
                       if s[key] is not None})
        return params
    return get


@pytest.fixture(scope="session")
def glyco_cli_args(case_settings):
    """glyco_cli_args(case) -> the same parameters as glyco CLI arguments."""
    def get(case):
        s = case_settings(case)
        args = []
        for key, spec in SHARED.items():
            if s[key] is not None:
                args += [spec[4], str(s[key])]
        for key, (_kw, flag) in CASE_ONLY.items():
            if s[key] is not None:
                args += [flag, str(s[key])]
        return args
    return get


@pytest.fixture(scope="session")
def compare_base_args(pytestconfig, case_settings):
    """compare_base_args(case) -> compare.py arguments for one structure."""
    def get(case):
        s = case_settings(case)
        args = []
        for key, spec in SHARED.items():
            if s[key] is not None:
                args += [spec[5], str(s[key])]
        if s["legacy_glycans"]:
            args += ["--glycans", ",".join(s["legacy_glycans"])]
        args += ["--box-workers", str(pytestconfig.glyco_box_workers)]
        if not pytestconfig.getoption("--glyco-compare-installed"):
            module = pytestconfig.getoption("--glyco-module")
            args += ["--glyco-command", shlex.join([sys.executable, "-m", module])]
        return args + pytestconfig.getoption("--glyco-compare-arg")
    return get


def _has_record_types(glyco, universe):
    try:
        universe.atoms.record_types  # noqa: B018
        return True
    except (AttributeError, glyco.mda.exceptions.NoDataError):
        return False


def _default_frames(n_frames, max_frames):
    if n_frames <= 1 or max_frames <= 0 or max_frames >= n_frames:
        return None
    import numpy as np
    return tuple(int(f) for f in np.unique(
        np.linspace(0, n_frames - 1, max_frames).round().astype(int)))


@pytest.fixture(scope="session")
def runner(glyco, glyco_params, pytestconfig, tmp_path_factory):
    """runner(case, mode, frames="auto", universe=None, cache=True, **overrides) -> Run"""
    results = {}
    max_frames = pytestconfig.getoption("--glyco-max-frames")
    box_workers = pytestconfig.glyco_box_workers

    def run(case, mode, frames="auto", universe=None, cache=True, **overrides):
        params = glyco_params(case)
        frames_key = frames if frames == "auto" or frames is None else tuple(frames)
        key = (case, mode, frames_key, tuple(sorted(overrides.items())))
        if cache and universe is None and key in results:
            return results[key]

        u = universe if universe is not None else glyco.load_universe(
            case.top, list(case.traj) or None)

        if (mode != "default" and not _has_record_types(glyco, u)
                and "sasa_env" not in params):
            pytest.skip(f"{mode} needs PDB record types for its FreeSASA atom set; "
                        "this topology has none (give --glyco-sasa-env)")

        if frames == "auto":
            frames = _default_frames(u.trajectory.n_frames, max_frames)
        frames = list(frames) if frames is not None else None

        name = re.sub(r"[^\w.-]", "_", f"{case.id}-{mode}")[:60]
        outdir = tmp_path_factory.mktemp(name)

        kwargs = {**params, "outdir": str(outdir), "write_excluded": True,
                  "compat": mode == "compat", "compat_v2": mode == "compat-v2",
                  "box_workers": box_workers, **overrides}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analysis = glyco.GlycoAnalysis(u, **kwargs)
            analysis.run(frames=frames)

        result = Run(analysis, outdir, frames, [str(w.message) for w in caught])

        if universe is None:
            results[key] = result
        return result

    return run
<<<<<<< HEAD
=======

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        g = _group_key(item)
        if g != item.nodeid:
            item.add_marker(pytest.mark.xdist_group(g))

    order = config.getoption("--glyco-order")
    if order == "none":
        return
    known = config.cache.get(_DUR_KEY, {})
    first = {}
    for i, it in enumerate(items):
        first.setdefault(_group_key(it), i)

    def key(it):
        c = known.get(_group_key(it), math.inf)        # unmeasured: treated as slow
        return (c if order == "fast-first" else -c, first[_group_key(it)])
    items.sort(key=key)
>>>>>>> b59cecf (fixed failing tests)
=======
Global pytest fixtures
"""

# Use this file if you need to share any fixtures
# across multiple modules
# More information at
# https://docs.pytest.org/en/stable/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session

import pytest

from glycokit.data.files import MDANALYSIS_LOGO


@pytest.fixture
def mdanalysis_logo_text() -> str:
    """Example fixture demonstrating how data files can be accessed"""
    with open(MDANALYSIS_LOGO, "r", encoding="utf8") as f:
        logo_text = f.read()
    return logo_text
>>>>>>> 1f065a3 (Initial creation with MDAnalysis Cookiecutter 0.1)
