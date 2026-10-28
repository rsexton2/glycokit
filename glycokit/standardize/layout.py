"""TER placement and glycan reordering."""

import numpy as np

from glycokit.standardize.constants import POLY


def _ter_positions(res, attach, opt, rep):
    """
    Return {residue index after which TER is written: terminal polymer residue}.

    A chain ends only where the next *polymer* residue is in a different
    (segid, chain) or there is none. Glycans/ligands interleaved within a chain
    never end it. Glycans trailing the last residue that belong to the same
    original chain (or are attached to it) are kept before the TER.

    :rtype: dict[int, Residue]
    """
    out, deferred = {}, 0
    if not opt.write_ter:
        return out
    n = len(res)
    poly = [k for k, r in enumerate(res) if r.kind in POLY]
    for a, b in zip(poly, poly[1:] + [None]):
        r = res[a]
        key = (r.segid, r.chain)
        if opt.ter_before_glycans:  # old behaviour
            if b is None or b != a + 1 or (res[b].segid, res[b].chain) != key:
                out[a] = r
            continue
        if b is not None and (res[b].segid, res[b].chain) == key:
            continue  # chain continues
        end, stop = a, (b if b is not None else n)
        for j in range(a + 1, stop):
            g = res[j]
            if g.kind == "glycan" and (
                g.orig["chain"] == r.orig["chain"] or key in attach.get(j, ())
            ):
                end = j
            else:
                break
        if end != a:
            deferred += 1
        out[end] = r
    if deferred:
        rep["info"].append(
            f"{deferred} TER record(s) placed after trailing glycans; "
            "TER names the terminal polymer residue"
        )
    return out


def _reorder_residues(res, t, pos, bonds, conect, link_pairs, attach, mode, rep):
    """
    Permute residues (and atoms) so glycans follow their polymer chain or the end.

    :param str mode: ``'after-chain'`` or ``'end'``.
    :return: The new ``(res, t, pos, bonds, conect, link_pairs, attach, perm)``;
        ``perm[new atom index] = old atom index``.
    :rtype: tuple
    """
    poly_keys = []
    for r in res:
        if r.kind in POLY and (r.segid, r.chain) not in poly_keys:
            poly_keys.append((r.segid, r.chain))
    glycans = [i for i, r in enumerate(res) if r.kind == "glycan"]
    others = [i for i, r in enumerate(res) if r.kind not in POLY and r.kind != "glycan"]

    # a tree bridging several chains goes after the first of them in output order
    anchor = {
        i: next((k for k in poly_keys if k in attach.get(i, ())), None) for i in glycans
    }

    order = []
    for k in poly_keys:
        order += [
            i for i, r in enumerate(res) if r.kind in POLY and (r.segid, r.chain) == k
        ]
        if mode == "after-chain":
            order += [i for i in glycans if anchor[i] == k]
    placed = set(order)
    order += [i for i in glycans if i not in placed]  # unattached / 'end' mode
    order += others  # solvent, ions, ligands
    assert sorted(order) == list(range(len(res)))

    perm = np.concatenate([np.arange(res[i].start, res[i].stop) for i in order])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))

    t = {k: [v[j] for j in perm] for k, v in t.items()}
    pos = pos[perm]
    new_res, s = [], 0
    for i in order:
        r = res[i]
        r.orig.setdefault("first_atom", r.start)
        n = r.stop - r.start
        r.start, r.stop = s, s + n
        s += n
        new_res.append(r)

    def remap(pairs):
        return {tuple(sorted((int(inv[a]), int(inv[b])))) for a, b in pairs}

    bonds = [(int(inv[a]), int(inv[b])) for a, b in bonds]
    new_index = {old: new for new, old in enumerate(order)}
    attach = {new_index[k]: v for k, v in attach.items()}

    moved = sum(1 for new, old in enumerate(order) if new != old)
    if moved:
        rep["atom_order_preserved"] = False
        rep["info"].append(
            f"glycan_placement={mode!r}: {moved} residue(s) moved; "
            "see <prefix>_atom_map.csv for the original atom indices"
        )
    return new_res, t, pos, bonds, remap(conect), remap(link_pairs), attach, perm
