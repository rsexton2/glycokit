"""Per-residue cleanup: caps, renames, completeness checks, elements, gaps."""

from collections import Counter
from dataclasses import replace

import numpy as np

from glycokit.standardize.constants import (
    BACKBONE,
    CAP_C,
    CAP_N,
    HIS_ALIASES,
    POLY,
    RES_ALIAS,
    SIDECHAIN,
    TERM_O,
    TWO_LETTER,
)


def _split_caps(res, names, rep):
    """
    Split CHARMM terminal caps (CAY/CY..., NT/CAT...) into ACE / NME residues
    (``caps='split'``). Non-contiguous cap atoms are kept with a warning.

    :rtype: list[Residue]
    """
    out = []
    for r in res:
        if r.kind != "protein":
            out.append(r)
            continue
        nm = names[r.start : r.stop]
        pieces = [r]
        if {"CAY", "CY"} <= set(nm):
            k = next((j for j, n in enumerate(nm) if n not in CAP_N), len(nm))
            if k == sum(n in CAP_N for n in nm) and k < len(nm):
                pieces = [
                    replace(
                        r,
                        stop=r.start + k,
                        resname="ACE",
                        resid=r.resid - 1,
                        kind="cap",
                        note="CHARMM N-cap split",
                    ),
                    replace(r, start=r.start + k),
                ]
            else:
                rep["warnings"].append(
                    f"N-cap atoms not contiguous in {r.resname} {r.resid}; kept"
                )
        last = pieces[-1]
        nm = names[last.start : last.stop]
        if {"NT", "CAT"} <= set(nm):
            k = next((j for j, n in enumerate(reversed(nm)) if n not in CAP_C), len(nm))
            if k == sum(n in CAP_C for n in nm) and k < len(nm):
                cut = last.stop - k
                pieces[-1:] = [
                    replace(last, stop=cut),
                    replace(
                        last,
                        start=cut,
                        resname="NME",
                        resid=last.resid + 1,
                        kind="cap",
                        note="CHARMM C-cap split",
                    ),
                ]
            else:
                rep["warnings"].append(
                    f"C-cap atoms not contiguous in {last.resname} {last.resid}; kept"
                )
        out.extend(pieces)
    return out


def _rename(res, t, opt, rep):
    """
    Rename atoms and residues in place.

    Terminal oxygens (OT1/OT2 -> O/OXT), cap atoms, histidine aliases and
    ``resname_map``. Counts are stored in ``rep['renames']``.

    :raises ValueError: On resnames longer than 4 characters, or 4-character
        resnames with ``long_resnames='error'``.
    """
    renames = Counter()
    for r in res:
        present = set(t["name"][r.start : r.stop])
        for i in range(r.start, r.stop):
            n = t["name"][i]
            new = n
            if r.kind == "protein" and opt.terminal_oxygens and n in TERM_O:
                tgt = TERM_O[n]
                if tgt in present:
                    msg = (
                        f"{r.chain}:{r.resname}:{r.resid} has both {n} and {tgt}; "
                        f"{n} not renamed"
                    )
                    if msg not in rep["warnings"]:
                        rep["warnings"].append(msg)
                else:
                    new = tgt
                    present.discard(n)
                    present.add(tgt)
            elif r.kind == "cap" and r.resname in ("ACE", "NME"):
                new = (CAP_N if r.resname == "ACE" else CAP_C).get(n, n)
            if new != n:
                renames[f"{n}->{new}"] += 1
                t["name"][i] = new
        if r.kind == "protein" and opt.histidine and r.resname.upper() in HIS_ALIASES:
            renames[f"{r.resname}->{opt.histidine}"] += 1
            r.resname = opt.histidine
        if r.resname in opt.resname_map:
            renames[f"{r.resname}->{opt.resname_map[r.resname]}"] += 1
            r.resname = opt.resname_map[r.resname]
        if len(r.resname) > 4:
            raise ValueError(f"resname {r.resname!r} longer than 4 characters")
        if len(r.resname) == 4 and opt.long_resnames != "allow":
            msg = (
                f"4-character resname {r.resname!r} written in cols 18-21 "
                "(nonstandard; consider resname_map)"
            )
            if opt.long_resnames == "error":
                raise ValueError(msg)
            if msg not in rep["warnings"]:
                rep["warnings"].append(msg)
    rep["renames"] = dict(renames)


def _check_sidechains(res, t, opt, rep):
    """
    Truncated residues are invalid input unless explicitly allowed.

    Missing backbone / side-chain heavy atoms are listed in
    ``rep['incomplete_residues']``.

    :raises ValueError: With ``incomplete_sidechains='error'``.
    """
    missing = []
    for r in res:
        if r.kind != "protein":
            continue
        name = r.orig["resname"].upper()
        tmpl = SIDECHAIN.get(RES_ALIAS.get(name, name))
        if tmpl is None:
            continue
        names = set(t["name"][r.start : r.stop])
        lost = [
            spec.split("|")[0]
            for spec in BACKBONE + tmpl
            if not any(a in names for a in spec.split("|"))
        ]
        if lost:
            o = r.orig
            missing.append(
                f"{o['chain'] or '-'}:{o['resname']}:{o['resid']}{o['icode']} "
                f"missing {','.join(lost)}"
            )
    rep["incomplete_residues"] = missing
    if missing:
        msg = (
            f"{len(missing)} protein residue(s) have missing heavy atoms, e.g. "
            f"{'; '.join(missing[:5])}"
        )
        if opt.incomplete_sidechains == "error":
            raise ValueError(
                msg + ". Rebuild them first (e.g. PDBFixer/MODELLER) "
                "or pass --allow-incomplete to proceed anyway."
            )
        rep["warnings"].append(msg)


def _elements(res, t):
    """
    Fill missing elements from atom names, in place.

    Two-letter elements (CL, NA, ZN, ...) are only used for 'other' residues
    that are single atoms or whose name is the element itself, so CA in a
    protein stays carbon.
    """
    for r in res:
        single = (r.stop - r.start) == 1
        for i in range(r.start, r.stop):
            if t["element"][i]:
                continue
            n = t["name"][i].upper()
            s = n.lstrip("0123456789")
            two = (
                s[:2] in TWO_LETTER
                and r.kind == "other"
                and (single or s == s[:2] or n == r.resname.upper())
            )
            t["element"][i] = s[:2] if two else s[:1]


def _find_gaps(res, t, pos, cutoff):
    """
    Chain gaps: consecutive polymer residues of one chain whose C(i)-N(i+1)
    distance exceeds ``cutoff``.

    :rtype: list[dict]
    """
    gaps = []
    poly = [r for r in res if r.kind in POLY]
    for a, b in zip(poly, poly[1:]):
        if a.chain != b.chain or a.segid != b.segid:
            continue
        c = [i for i in range(a.start, a.stop) if t["name"][i] == "C"]
        n = [i for i in range(b.start, b.stop) if t["name"][i] == "N"]
        if c and n:
            d = float(np.linalg.norm(pos[c[0]] - pos[n[0]]))
            if d > cutoff:
                gaps.append(
                    {
                        "chain": a.chain,
                        "segid": a.segid,
                        "after": f"{a.resname}{a.resid}{a.icode}",
                        "before": f"{b.resname}{b.resid}{b.icode}",
                        "missing_by_numbering": max(b.resid - a.resid - 1, 0),
                        "C_N_distance": round(d, 3),
                    }
                )
    return gaps
