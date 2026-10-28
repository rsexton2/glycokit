"""Output: fixed-column PDB, residue CSV, atom map, reordered trajectory."""

import csv
import string
from collections import defaultdict

import MDAnalysis as mda
import numpy as np

from glycokit.standardize.reading import _read_cryst1

_DIG_U = string.digits + string.ascii_uppercase
_DIG_L = string.digits + string.ascii_lowercase


def _b36(v, width, digits):
    """Base-36 representation of ``v`` with exactly ``width`` digits."""
    s = ""
    for _ in range(width):
        v, d = divmod(v, 36)
        s = digits[d] + s
    return s


def hy36(value, width):
    """
    Hybrid-36 encoding (decimal while it fits, then A000.. / a000..).

    :param int value: Non-negative integer (atom serial or resid).
    :param int width: Field width (5 for serials, 4 for resids).
    :rtype: str
    :raises ValueError: If ``value`` does not fit.
    """
    if value < 10**width:
        return f"{value:>{width}d}"
    v, block = value - 10**width, 26 * 36 ** (width - 1)
    if v < block:
        return _b36(v + 10 * 36 ** (width - 1), width, _DIG_U)
    v -= block
    if v < block:
        return _b36(v + 10 * 36 ** (width - 1), width, _DIG_L)
    raise ValueError(f"{value} too large for hybrid-36 width {width}")


def _fmt_name(name, element):
    """Atom name in PDB columns 13-16 (two-letter elements start in column 13)."""
    if len(name) >= 4:
        return name[:4]
    if len(element) == 2 and name.upper().startswith(element):
        return f"{name:<4}"
    return f" {name:<3}"


def _charge(q):
    """Formal charge in PDB columns 79-80 (e.g. ``'1+'``); blank for 0."""
    return "  " if not q else f"{abs(q)}{'+' if q > 0 else '-'}"[:2]


def _res_fields(r):
    """``(resname, chain, resid, icode)`` formatted for PDB columns 18-27."""
    rn = f"{r.resname:>3} " if len(r.resname) <= 3 else f"{r.resname:<4}"
    return rn, (r.chain[:1] or " "), hy36(r.resid, 4), (r.icode or " ")[:1]


def _write_pdb(path, res, t, pos, rep, cryst_line, conect, ter):
    """
    Write the fixed-column PDB file.

    :param dict ter: Output of :func:`_ter_positions`.
    :param set conect: Output of :func:`_conect_pairs`.
    :param cryst_line: CRYST1 record, or None.
    """
    big_serial = len(t["name"]) > 99999
    big_resid = any(r.resid > 9999 for r in res)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("REMARK   1 STANDARDIZED BY glyco_standardize\n")
        if cryst_line:
            fh.write(cryst_line + "\n")
        for k, r in enumerate(res):
            rn, chain, rid, ic = _res_fields(r)
            for i in range(r.start, r.stop):
                x, y, z = pos[i]
                el = t["element"][i][:2]
                fh.write(
                    f"{r.record:<6}{hy36(i + 1, 5)} {_fmt_name(t['name'][i], el)}"
                    f"{(t['altloc'][i] or ' ')[:1]}{rn}{chain}{rid}{ic}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{t['occ'][i]:6.2f}{t['bfac'][i]:6.2f}"
                    f"      {r.segid[:4]:<4}{el:>2}{_charge(t['charge'][i])}\n"
                )
            if k in ter:
                # cols 1-6 'TER', 7-11 serial (blank: atom serials are preserved),
                # 18-20 resname, 22 chain, 23-26 resid, 27 icode
                trn, tch, trid, tic = _res_fields(ter[k])
                fh.write(f"{'TER':<6}{'':5}{'':6}{trn}{tch}{trid}{tic}\n")

        if conect:
            adj = defaultdict(set)
            for i, j in conect:
                adj[i].add(j)
                adj[j].add(i)
            for i in sorted(adj):
                nb = sorted(adj[i])
                fh.writelines(
                    "CONECT"
                    + hy36(i + 1, 5)
                    + "".join(hy36(j + 1, 5) for j in nb[s : s + 4])
                    + "\n"
                    for s in range(0, len(nb), 4)
                )
        fh.write("END\n")
    if big_serial:
        rep["warnings"].append("atom serials > 99999 written in hybrid-36")
    if big_resid:
        rep["warnings"].append("resids > 9999 written in hybrid-36")
    rep["hybrid36"] = big_serial or big_resid


def _write_csv(path, res):
    """Write ``<prefix>_residues.csv``: final and original labels of every residue."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "resindex",
                "kind",
                "record",
                "chain",
                "resname",
                "resid",
                "icode",
                "segid",
                "first_atom",
                "n_atoms",
                "orig_resname",
                "orig_chain",
                "orig_resid",
                "orig_icode",
                "orig_segid",
                "note",
            ]
        )
        for k, r in enumerate(res):
            o = r.orig
            w.writerow(
                [
                    k,
                    r.kind,
                    r.record,
                    r.chain,
                    r.resname,
                    r.resid,
                    r.icode,
                    r.segid,
                    r.start,
                    r.stop - r.start,
                    o["resname"],
                    o["chain"],
                    o["resid"],
                    o["icode"],
                    o["segid"],
                    r.note,
                ]
            )


def _cryst1_line(topology, trajectory, u, opt, rep):
    """
    CRYST1 record to write, or None.

    Priority: explicit ``opt.cryst1``; the input CRYST1 line (PDB without a
    trajectory); a line built from the box dimensions (space group P 1).
    Missing or placeholder boxes give None.

    :rtype: str or None
    """
    if opt.cryst1:
        a, b, c, *ang = opt.cryst1
        al, be, ga = (list(ang) + [90.0, 90.0, 90.0])[:3]
        return (
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} P 1           1"
        )
    dims = u.dimensions
    real = (
        dims is not None
        and not np.allclose(dims[:3], 1.0)
        and np.all(np.asarray(dims[:3]) > 0)
    )
    if not real:
        rep["warnings"].append(
            "input box missing/placeholder; no CRYST1 written unless cryst1 given"
        )
        return None
    if not opt.keep_cryst1:
        rep["warnings"].append("input box discarded (keep_cryst1=False)")
        return None
    line = None if trajectory else _read_cryst1(topology)
    if line:
        return line
    a, b, c, al, be, ga = (float(x) for x in dims)
    rep["warnings"].append(
        "CRYST1 written from box dimensions with space group P 1 "
        "(original space group not available)"
    )
    return f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} P 1           1"


def _write_atom_map(path, perm):
    """Write ``new_serial, orig_index`` for every atom after reordering."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["new_serial", "orig_index"])
        for new, old in enumerate(perm):
            w.writerow([new + 1, int(old)])


def _write_reordered_traj(u, perm, path):
    """Write every trajectory frame with atoms in the reordered (``perm``) order."""
    ag = u.atoms[perm]  # AtomGroup indexing keeps this order
    with mda.Writer(path, ag.n_atoms) as w:
        for _ in u.trajectory:
            w.write(ag)
