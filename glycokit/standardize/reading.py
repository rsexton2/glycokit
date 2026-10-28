"""Reading the input and splitting it into residues."""

from collections import defaultdict

import MDAnalysis as mda
import numpy as np

from glycokit.standardize.model import Residue


def _get(ag, attr, default):
    """Per-atom attribute as a list; ``default`` if missing (or per atom if None)."""
    try:
        vals = getattr(ag, attr)
    except (AttributeError, mda.exceptions.NoDataError):
        return [default] * len(ag)
    return [default if v is None else v for v in vals]


def _int(x):
    """``int(x)``, or 0 if ``x`` cannot be converted."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _atom_table(ag):
    """
    Plain per-atom data of ``ag`` at the current frame.

    :return: ``(t, pos, bonds)`` as described in :func:`_read`.
    """
    t = {
        "name": [str(x).strip() for x in ag.names],
        "resname": [str(x).strip() for x in ag.resnames],
        "resid": [int(x) for x in ag.resids],
        "chain": [str(x).strip() for x in _get(ag, "chainIDs", "")],
        "icode": [str(x).strip() for x in _get(ag, "icodes", "")],
        "segid": [str(x).strip() for x in _get(ag, "segids", "")],
        "record": [str(x).strip() for x in _get(ag, "record_types", "")],
        "altloc": [str(x).strip() for x in _get(ag, "altLocs", "")],
        "element": [str(x).strip().upper() for x in _get(ag, "elements", "")],
        "occ": [float(x) for x in _get(ag, "occupancies", 1.0)],
        "bfac": [float(x) for x in _get(ag, "tempfactors", 0.0)],
        "charge": [_int(x) for x in _get(ag, "formalcharges", 0)],
    }
    try:
        bonds = [(int(i), int(j)) for i, j in ag.bonds.indices]
    except (AttributeError, mda.exceptions.NoDataError):
        bonds = []
    pos = np.round(ag.positions.astype(np.float64), 3)
    return t, pos, bonds


def _read(topology, trajectory, topology_format, frame):
    """(docstring unchanged)"""
    kw = {"topology_format": topology_format} if topology_format else {}
    u = (
        mda.Universe(topology, trajectory, **kw)
        if trajectory
        else mda.Universe(topology, **kw)
    )
    u.trajectory[frame]  # pylint: disable=pointless-statement  # load the frame
    t, pos, bonds = _atom_table(u.atoms)
    return u, t, pos, bonds


def _read_cryst1(path):
    """
    The CRYST1 line of a PDB file, if it appears before the first atom.

    :rtype: str or None
    """
    p = str(path).lower()
    if not p.endswith((".pdb", ".ent")):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                return line.rstrip("\n")
            if line.startswith(("ATOM", "HETATM", "MODEL")):
                return None
    return None


def _drop_altlocs(t, pos, rep):
    """
    Keep blank altlocs and the first altloc of each residue (``altloc='first'``).

    :return: ``(t, pos, keep)`` with ``keep`` the kept input atom indices.
    """
    keep, first = [], {}
    for i, alt in enumerate(t["altloc"]):
        rkey = (t["chain"][i], t["resid"][i], t["icode"][i], t["resname"][i])
        if alt and first.setdefault(rkey, alt) != alt:
            continue
        keep.append(i)
    if len(keep) < len(t["name"]):
        rep["warnings"].append(
            f"altloc='first' dropped {len(t['name']) - len(keep)} atoms"
        )
        rep["atom_order_preserved"] = False
    t = {k: [v[i] for i in keep] for k, v in t.items()}
    return t, pos[keep], keep


def _segment(t):
    """
    Split atoms into residues.

    A new residue starts on any identity change (record, resname, chain,
    resid, icode, segid) OR on a repeated atom name (merged residues). Atoms
    with the same name but different non-blank altlocs are NOT a repeat.

    :rtype: list[Residue]
    """

    def key(i):
        return (
            t["record"][i],
            t["resname"][i],
            t["chain"][i],
            t["resid"][i],
            t["icode"][i],
            t["segid"][i],
        )

    def mk(s, e, note, split):
        return Residue(
            s,
            e,
            t["resname"][s],
            t["chain"][s],
            t["resid"][s],
            t["icode"][s],
            t["segid"][s],
            t["record"][s],
            orig={
                "resname": t["resname"][s],
                "chain": t["chain"][s],
                "resid": t["resid"][s],
                "icode": t["icode"][s],
                "segid": t["segid"][s],
                "record": t["record"][s],
            },
            note=note,
            split=split,
        )

    res, start, note, split = [], 0, "", False
    seen = defaultdict(set)
    seen[t["name"][0]].add(t["altloc"][0])
    for i in range(1, len(t["name"])):
        name, alt = t["name"][i], t["altloc"][i]
        s = seen.get(name)
        dup = bool(s) and (alt in s or "" in s or alt == "")
        same_key = key(i) == key(i - 1)
        if not same_key or dup:
            res.append(mk(start, i, note, split))
            split = same_key and dup
            note = f"split: repeated atom name {name!r}" if split else ""
            start, seen = i, defaultdict(set)
        seen[name].add(alt)
    res.append(mk(start, len(t["name"]), note, split))
    return res
