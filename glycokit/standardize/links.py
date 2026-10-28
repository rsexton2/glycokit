"""Inter-residue glycan links and CONECT records."""

import numpy as np

from glycokit.standardize.constants import POLY
from glycokit.standardize.model import _owner


def _link_pairs(res, t, pos, opt, bonds):
    """
    Inter-residue bonds involving glycans, as atom-index pairs.

    From topology bonds if present, otherwise geometric (glycan C1 -> nearest
    protein heavy atom or another glycan's O within ``link_cutoff``).
    Independent of chain/resid labels, so it can run before :func:`_assign_ids`.

    :return: ``(pairs, source)`` with source ``'topology'`` or ``'geometry'``.
    :rtype: tuple[set[tuple[int, int]], str]
    """
    owner = _owner(res, len(t["name"]))
    inter = set()
    if bonds:
        source = "topology"
        for i, j in bonds:
            if owner[i] != owner[j] and "glycan" in (
                res[owner[i]].kind,
                res[owner[j]].kind,
            ):
                inter.add((min(i, j), max(i, j)))
    else:
        source = "geometry"
        cand = np.array(
            [
                i
                for r in res
                if r.kind in POLY
                for i in range(r.start, r.stop)
                if t["element"][i] != "H"
            ]
            + [
                i
                for r in res
                if r.kind == "glycan"
                for i in range(r.start, r.stop)
                if t["element"][i] == "O"
            ],
            dtype=int,
        )
        for k, r in enumerate(res):
            if r.kind != "glycan" or not cand.size:
                continue
            c1 = [i for i in range(r.start, r.stop) if t["name"][i] == "C1"]
            if not c1:
                continue
            c = cand[owner[cand] != k]
            if not c.size:
                continue
            d = np.linalg.norm(pos[c] - pos[c1[0]], axis=1)
            j = int(np.argmin(d))
            if d[j] <= opt.link_cutoff:
                inter.add(tuple(sorted((c1[0], int(c[j])))))
    return inter, source


def _link_records(res, t, pos, pairs, source):
    """
    Report entries for link pairs, labelled with the CURRENT residue labels.

    :rtype: list[dict]
    """
    owner = _owner(res, len(t["name"]))

    def lab(i):
        r = res[owner[i]]
        return f"{r.chain}:{r.resname}:{r.resid}{r.icode}:{t['name'][i]}"

    return [
        {
            "atom1": lab(i),
            "atom2": lab(j),
            "distance": round(float(np.linalg.norm(pos[i] - pos[j])), 3),
            "source": source,
        }
        for i, j in sorted(pairs)
    ]


def _conect_pairs(res, t, bonds, extra, mode):
    """
    Bonds written as CONECT records.

    ``none``: nothing; ``het``: bonds involving non-polymer residues plus
    inter-residue polymer bonds other than peptide bonds (e.g. disulfides);
    ``all``: every topology bond. ``extra`` (glycan links) is always added
    unless ``mode='none'``.

    :rtype: set[tuple[int, int]]
    """
    if mode == "none":
        return set()
    owner = _owner(res, len(t["name"]))
    pairs = set(extra)
    for i, j in bonds:
        i, j = min(i, j), max(i, j)
        a, b = res[owner[i]], res[owner[j]]
        if mode == "all":
            pairs.add((i, j))
        elif a.kind not in POLY or b.kind not in POLY:
            pairs.add((i, j))  # hetero groups and their links
        elif owner[i] != owner[j] and {t["name"][i], t["name"][j]} != {"C", "N"}:
            pairs.add((i, j))  # e.g. disulfides; skip peptide bonds
    return pairs
