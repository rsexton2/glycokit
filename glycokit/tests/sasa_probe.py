"""sasa_probe.py -- record every FreeSASA call (plus subprocess / process-pool
launches) made inside a ``with capture(path):`` block.

Output JSON: {"version": 2, "pid": int, "calls": [...], "events": [...]}

Only the current process is patched: SASA computed in worker processes or by
the freesasa executable shows up under "events", not "calls".
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import json
import multiprocessing
import os
import subprocess
import sys
import traceback

import freesasa
import numpy as np

_THIS_FILE = os.path.abspath(__file__)

# Originals are stored ONCE, on the freesasa module itself, so that importing
# this file twice under different names can never store an already-wrapped object.
_ORIG = getattr(freesasa, "_probe_originals", None)
if _ORIG is None:
    _ORIG = {
        "freesasa.calc": freesasa.calc,
        "freesasa.calcCoord": getattr(freesasa, "calcCoord", None),
        "freesasa.calcBioPDB": getattr(freesasa, "calcBioPDB", None),
        "freesasa.structureArray": getattr(freesasa, "structureArray", None),
        "freesasa.classifyResults": getattr(freesasa, "classifyResults", None),
        "subprocess.Popen": subprocess.Popen,
        "multiprocessing.Pool": multiprocessing.Pool,
        "concurrent.futures.ProcessPoolExecutor": concurrent.futures.ProcessPoolExecutor,
    }

    freesasa._probe_originals = _ORIG  # pylint: disable=protected-access

# {"calls": [...], "events": [...]} while a capture() block is active
_RUN: dict = {"state": None}

# Errors from inspecting arbitrary entries of sys.modules (lazy modules,
# extension modules without __dict__, non-string __file__)
_MODULE_SCAN_ERRORS = (AttributeError, TypeError, ValueError, ImportError)

# -------------------------------------- helpers -------------------------------


def _freesasa_call(fn, *args):
    """
    Call a freesasa function and return ``(value, None)`` or ``(None, exc)``.

    freesasa's Cython bindings raise plain ``Exception``, so nothing
    narrower can be caught here. Used only for optional, diagnostic values.
    """
    try:
        return fn(*args), None
    except Exception as exc:  # pylint: disable=broad-exception-caught  # noqa: BLE001
        return None, exc


@functools.lru_cache(maxsize=1)
def _reference_classifier():
    """Default FreeSASA classifier (ProtOr), or None if it cannot be built."""
    classifier, _ = _freesasa_call(freesasa.Classifier)
    return classifier


def _caller(depth: int = 4) -> list[str]:
    """Innermost frames outside this file, e.g. 'mdafreesasa.py:182 in _compute'."""
    out = []
    for fr in reversed(traceback.extract_stack()):
        if os.path.abspath(fr.filename) == _THIS_FILE:
            continue
        out.append(f"{os.path.basename(fr.filename)}:{fr.lineno} in {fr.name}")
        if len(out) == depth:
            break
    return out


def _event(kind: str, arg0) -> None:
    state = _RUN["state"]
    if state is not None:
        state["events"].append(
            {"event": kind, "arg0": repr(arg0)[:300], "caller": _caller()}
        )


def _params(p) -> dict:
    if p is None:
        p = freesasa.Parameters()  # what FreeSASA uses when None is passed
    out = {}
    for name in ("probeRadius", "algorithm", "nPoints", "nSlices", "nThreads"):
        fn = getattr(p, name, None)
        if callable(fn):
            # some getters only apply to one algorithm
            value, exc = _freesasa_call(fn)
            out[name] = value if exc is None else f"<{exc}>"
    return out


def _structure_atoms(structure) -> list[dict]:
    ref = _reference_classifier()
    atoms = []
    for i in range(structure.nAtoms()):
        resname = structure.residueName(i).strip()
        name = structure.atomName(i).strip()
        ref_r = None
        if ref is not None:
            radius, _ = _freesasa_call(ref.radius, resname, name)
            if radius is not None:
                ref_r = round(radius, 4)
        x, y, z = structure.coord(i)
        atoms.append(
            {
                "i": i,
                "chain": structure.chainLabel(i).strip(),
                "resnum": structure.residueNumber(i).strip(),
                "resname": resname,
                "name": name,
                "radius": round(structure.radius(i), 4),
                "protor_radius": ref_r,
                "xyz": [round(x, 3), round(y, 3), round(z, 3)],
            }
        )
    return atoms


def _flat(seq) -> list[float]:
    try:
        return [float(v) for v in np.asarray(seq, dtype=float).ravel()]
    except (TypeError, ValueError):  # ragged or non-numeric input
        return [float(v) for v in seq]


# -------------------------------------------------------------------- wrappers


def _calc(structure, *args, **kwargs):
    result = _ORIG["freesasa.calc"](structure, *args, **kwargs)
    state = _RUN["state"]
    if state is not None:
        params = args[0] if args else kwargs.get("parameters")
        rec = {
            "call": "calc",
            "caller": _caller(),
            "params": _params(params),
            "n_atoms": structure.nAtoms(),
            "atoms": _structure_atoms(structure),
        }
        areas = [result.atomArea(i) for i in range(structure.nAtoms())]
        for atom, area in zip(rec["atoms"], areas):
            atom["area"] = round(area, 4)
        rec["total"] = round(result.totalArea(), 4)
        state["calls"].append(rec)
    return result


def _orig_callable(key: str):
    orig = _ORIG.get(key)
    if orig is None or not callable(orig):
        raise AttributeError(f"{key} is unavailable in this FreeSASA build")
    return orig


def _calc_coord(coord, radii, *args, **kwargs):
    state = _RUN["state"]
    if state is not None:
        xyz, radius_list = _flat(coord), _flat(radii)
        params = args[0] if args else kwargs.get("parameters")
        state["calls"].append(
            {
                "call": "calcCoord",
                "caller": _caller(),
                "params": _params(params),
                "n_atoms": len(radius_list),
                "atoms": [
                    {
                        "i": i,
                        "radius": round(ri, 4),
                        "xyz": [round(v, 3) for v in xyz[3 * i : 3 * i + 3]],
                    }
                    for i, ri in enumerate(radius_list)
                ],
            }
        )
    return _orig_callable("freesasa.calcCoord")(coord, radii, *args, **kwargs)


def _logged(key: str):
    orig = _orig_callable(key)

    @functools.wraps(orig)  # also sets __wrapped__
    def wrapper(*args, **kwargs):
        _event(key, args[0] if args else kwargs)
        return orig(*args, **kwargs)

    return wrapper


class _LoggedPopen(_ORIG["subprocess.Popen"]):  # subclass keeps isinstance() working
    def __init__(self, args, *a, **kw):
        _event("subprocess.Popen", args)  # also catches subprocess.run/check_output
        super().__init__(args, *a, **kw)


class _LoggedProcessPool(_ORIG["concurrent.futures.ProcessPoolExecutor"]):
    def __init__(self, *a, **kw):
        _event("ProcessPoolExecutor", {"args": a, "kwargs": kw})
        super().__init__(*a, **kw)


def _swaps() -> dict[int, tuple[str, object]]:
    def logged_if_present(key):
        return _logged(key) if _ORIG[key] else None

    repl = {
        "freesasa.calc": _calc,
        "freesasa.calcCoord": _calc_coord,
        "freesasa.calcBioPDB": logged_if_present("freesasa.calcBioPDB"),
        "freesasa.structureArray": logged_if_present("freesasa.structureArray"),
        "freesasa.classifyResults": logged_if_present("freesasa.classifyResults"),
        "subprocess.Popen": _LoggedPopen,
        "multiprocessing.Pool": _logged("multiprocessing.Pool"),
        "concurrent.futures.ProcessPoolExecutor": _LoggedProcessPool,
    }
    return {
        id(_ORIG[k]): (k, v)
        for k, v in repl.items()
        if v is not None and _ORIG.get(k) is not None
    }


def _patch_everywhere(swaps) -> list[tuple[dict, str, object]]:
    """Swap originals in every loaded module (covers `from freesasa import calc`)."""
    undo = []
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if os.path.abspath(getattr(mod, "__file__", "") or "") == _THIS_FILE:
                continue  # never patch the probe itself
            namespace = vars(mod)
        except _MODULE_SCAN_ERRORS:
            continue
        for attr, val in list(namespace.items()):
            hit = swaps.get(id(val))
            if hit is not None and val is _ORIG[hit[0]]:
                namespace[attr] = hit[1]
                undo.append((namespace, attr, val))
    return undo


# ------------------------------------------------------------------ public API
@contextlib.contextmanager
def capture(path, *, require_calls: bool = False):
    """Record FreeSASA calls inside the block and write them to ``path``.

    require_calls=True raises if no calc/calcCoord call was seen (e.g. a cached
    result, or SASA computed in another process -- check "events").

    :return: The live record, a dict with ``"calls"`` and ``"events"`` lists.
    """
    if _RUN["state"] is not None:
        raise RuntimeError("capture() blocks cannot be nested")
    state = {"calls": [], "events": []}
    _RUN["state"] = state
    undo = _patch_everywhere(_swaps())
    completed = False
    try:
        yield state
        completed = True
    finally:
        for namespace, attr, val in reversed(undo):
            namespace[attr] = val
        _RUN["state"] = None
        data = {"version": 2, "pid": os.getpid(), **state}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, default=str)
        calls = [
            (c["call"], c["n_atoms"], c["caller"][0] if c["caller"] else "?")
            for c in data["calls"]
        ]
        events = [e["event"] for e in data["events"]]
        print(f"{path}: {len(calls)} FreeSASA call(s) {calls}; events: {events}")
        assert freesasa.calc is _ORIG["freesasa.calc"], "freesasa.calc was not restored"
    if completed and require_calls and not data["calls"]:
        raise AssertionError(
            f"{path}: no FreeSASA calls captured (cached result, worker process, "
            f"or subprocess? events={events})"
        )


def load_calls(path) -> list[dict]:
    """Read a probe file; accepts both the old list format and the v2 dict format."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["calls"] if isinstance(data, dict) else data


def _selftest(pdb):
    """Run FreeSASA on ``pdb`` inside capture() and check the patching."""
    with capture("probe_selftest.json", require_calls=True):
        structure = freesasa.Structure(pdb)
        result = freesasa.calc(structure)
    print(
        "total SASA:",
        round(result.totalArea(), 3),
        "(no recursion, originals restored)",
    )


if __name__ == "__main__":
    # Self-test: python sasa_probe.py structure.pdb
    _selftest(sys.argv[1])
