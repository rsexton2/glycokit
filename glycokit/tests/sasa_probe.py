"""capture.py -- record every FreeSASA call (plus subprocess / process-pool
launches) made inside a ``with capture(path):`` block.

Output JSON: {"version": 2, "pid": int, "calls": [...], "events": [...]}

Only the current process is patched: SASA computed in worker processes or by
the freesasa executable shows up under "events", not "calls".
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import json
import multiprocessing
import os
import subprocess
import sys
import traceback

import freesasa

_THIS_FILE = os.path.abspath(__file__)

# Originals are stored ONCE, on the freesasa module itself, so that importing
# this file twice under different names can never store an already-wrapped object.
if not hasattr(freesasa, "_probe_originals"):
    freesasa._probe_originals = {
        "freesasa.calc": freesasa.calc,
        "freesasa.calcCoord": getattr(freesasa, "calcCoord", None),
        "freesasa.calcBioPDB": getattr(freesasa, "calcBioPDB", None),
        "freesasa.structureArray": getattr(freesasa, "structureArray", None),
        "freesasa.classifyResults": getattr(freesasa, "classifyResults", None),
        "subprocess.Popen": subprocess.Popen,
        "multiprocessing.Pool": multiprocessing.Pool,
        "concurrent.futures.ProcessPoolExecutor": concurrent.futures.ProcessPoolExecutor,
    }
_ORIG = freesasa._probe_originals
_state: dict | None = None  # {"calls": [...], "events": [...]} while capturing
_REF_CLASSIFIER = None

# -------------------------------------- helpers -------------------------------


def _reference_classifier():
    global _REF_CLASSIFIER
    if _REF_CLASSIFIER is None:
        try:
            _REF_CLASSIFIER = freesasa.Classifier()
        except Exception:
            _REF_CLASSIFIER = False
    return _REF_CLASSIFIER or None


def _caller(depth: int = 4) -> list[str]:
    """Innermost frames outside this file, e.g. 'mdafreesasa.py:182 in _compute_freesasa_result'."""
    out = []
    for fr in reversed(traceback.extract_stack()):
        if os.path.abspath(fr.filename) == _THIS_FILE:
            continue
        out.append(f"{os.path.basename(fr.filename)}:{fr.lineno} in {fr.name}")
        if len(out) == depth:
            break
    return out


def _event(kind: str, arg0) -> None:
    if _state is not None:
        _state["events"].append({"event": kind, "arg0": repr(arg0)[:300], "caller": _caller()})


def _params(p) -> dict:
    if p is None:
        p = freesasa.Parameters()               # what FreeSASA uses when None is passed
    out = {}
    for name in ("probeRadius", "algorithm", "nPoints", "nSlices", "nThreads"):
        fn = getattr(p, name, None)
        if callable(fn):
            try:
                out[name] = fn()
            except Exception as exc:            # some getters only apply to one algorithm
                out[name] = f"<{exc}>"
    return out


def _reference_classifier():
    try:
        return freesasa.Classifier()            # default classifier (ProtOr)
    except Exception:
        return None


def _structure_atoms(s) -> list[dict]:
    ref = _reference_classifier()
    atoms = []
    for i in range(s.nAtoms()):
        resname = s.residueName(i).strip()
        name = s.atomName(i).strip()
        ref_r = None
        if ref is not None:
            try:
                ref_r = round(ref.radius(resname, name), 4)
            except Exception:
                pass
        x, y, z = s.coord(i)
        atoms.append({
            "i": i,
            "chain": s.chainLabel(i).strip(),
            "resnum": s.residueNumber(i).strip(),
            "resname": resname,
            "name": name,
            "radius": round(s.radius(i), 4),
            "protor_radius": ref_r,
            "xyz": [round(x, 3), round(y, 3), round(z, 3)],
        })
    return atoms


def _flat(seq) -> list[float]:
    try:
        import numpy as np
        return [float(v) for v in np.asarray(seq, dtype=float).ravel()]
    except Exception:
        return [float(v) for v in seq]


# -------------------------------------------------------------------- wrappers
# def _calc(structure, *args, **kwargs):
#     if _state is not None:
#         params = args[0] if args else kwargs.get("parameters")
#         _state["calls"].append({
#             "call": "calc", "caller": _caller(), "params": _params(params),
#             "n_atoms": structure.nAtoms(), "atoms": _structure_atoms(structure)})
#     return _ORIG["freesasa.calc"](structure, *args, **kwargs)
def _calc(structure, *args, **kwargs):
    result = _ORIG["freesasa.calc"](structure, *args, **kwargs)
    if _state is not None:
        params = args[0] if args else kwargs.get("parameters")
        rec = {"call": "calc", "caller": _caller(), "params": _params(params),
               "n_atoms": structure.nAtoms(), "atoms": _structure_atoms(structure)}
        areas = [result.atomArea(i) for i in range(structure.nAtoms())]
        for a, area in zip(rec["atoms"], areas):
            a["area"] = round(area, 4)
        rec["total"] = round(result.totalArea(), 4)
        _state["calls"].append(rec)
    return result

def _calc_coord(coord, radii, *args, **kwargs):
    if _state is not None:
        c, r = _flat(coord), _flat(radii)
        params = args[0] if args else kwargs.get("parameters")
        _state["calls"].append({
            "call": "calcCoord", "caller": _caller(), "params": _params(params),
            "n_atoms": len(r),
            "atoms": [{"i": i, "radius": round(ri, 4),
                       "xyz": [round(v, 3) for v in c[3 * i:3 * i + 3]]}
                      for i, ri in enumerate(r)]})
    return _ORIG["freesasa.calcCoord"](coord, radii, *args, **kwargs)


def _logged(key: str):
    orig = _ORIG[key]

    def wrapper(*args, **kwargs):
        _event(key, args[0] if args else kwargs)
        return orig(*args, **kwargs)
    wrapper.__wrapped__ = orig
    return wrapper


class _LoggedPopen(_ORIG["subprocess.Popen"]):       # subclass keeps isinstance() working
    def __init__(self, args, *a, **kw):
        _event("subprocess.Popen", args)             # also catches subprocess.run/check_output
        super().__init__(args, *a, **kw)


class _LoggedProcessPool(_ORIG["concurrent.futures.ProcessPoolExecutor"]):
    def __init__(self, *a, **kw):
        _event("ProcessPoolExecutor", {"args": a, "kwargs": kw})
        super().__init__(*a, **kw)


def _swaps() -> dict[int, tuple[str, object]]:
    repl = {
        "freesasa.calc": _calc,
        "freesasa.calcCoord": _calc_coord,
        "freesasa.calcBioPDB": _logged("freesasa.calcBioPDB") if _ORIG["freesasa.calcBioPDB"] else None,
        "freesasa.structureArray": _logged("freesasa.structureArray") if _ORIG["freesasa.structureArray"] else None,
        "freesasa.classifyResults": _logged("freesasa.classifyResults") if _ORIG["freesasa.classifyResults"] else None,
        "subprocess.Popen": _LoggedPopen,
        "multiprocessing.Pool": _logged("multiprocessing.Pool"),
        "concurrent.futures.ProcessPoolExecutor": _LoggedProcessPool,
    }
    return {id(_ORIG[k]): (k, v) for k, v in repl.items()
            if v is not None and _ORIG.get(k) is not None}


def _patch_everywhere(swaps) -> list[tuple[dict, str, object]]:
    """Swap originals in every loaded module (covers `from freesasa import calc`)."""
    undo = []
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if os.path.abspath(getattr(mod, "__file__", "") or "") == _THIS_FILE:
                continue                         # never patch the probe itself
            d = vars(mod)
        except Exception:
            continue
        for attr, val in list(d.items()):
            hit = swaps.get(id(val))
            if hit is not None and val is _ORIG[hit[0]]:
                d[attr] = hit[1]
                undo.append((d, attr, val))
    return undo


# ------------------------------------------------------------------ public API
@contextlib.contextmanager
def capture(path, *, require_calls: bool = False):
    """Record FreeSASA calls inside the block and write them to ``path``.

    require_calls=True raises if no calc/calcCoord call was seen (e.g. a cached
    result, or SASA computed in another process -- check "events").
    """
    global _state
    if _state is not None:
        raise RuntimeError("capture() blocks cannot be nested")
    _state = {"calls": [], "events": []}
    undo = _patch_everywhere(_swaps())
    completed = False
    try:
        yield _state
        completed = True
    finally:
        for d, attr, val in reversed(undo):
            d[attr] = val
        data = {"version": 2, "pid": os.getpid(), **_state}
        _state = None
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1, default=str)
        print(f"{path}: {len(data['calls'])} FreeSASA call(s) "
              f"{[(c['call'], c['n_atoms'], c['caller'][0] if c['caller'] else '?') for c in data['calls']]}; "
              f"events: {[e['event'] for e in data['events']]}")
        assert freesasa.calc is _ORIG["freesasa.calc"], "freesasa.calc was not restored"
    if completed and require_calls and not data["calls"]:
        raise AssertionError(
            f"{path}: no FreeSASA calls captured (cached result, worker process, "
            f"or subprocess? events={[e['event'] for e in data['events']]})")


def load_calls(path) -> list[dict]:
    """Read a probe file; accepts both the old list format and the v2 dict format."""
    with open(path) as fh:
        data = json.load(fh)
    return data["calls"] if isinstance(data, dict) else data


if __name__ == "__main__":
    # Self-test: python capture.py structure.pdb
    pdb = sys.argv[1]
    with capture("probe_selftest.json", require_calls=True):
        s = freesasa.Structure(pdb)
        r = freesasa.calc(s)
    print("total SASA:", round(r.totalArea(), 3), "(no recursion, originals restored)")