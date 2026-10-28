"""Final chain, segid, resid, icode and record labels."""

import string
from collections import Counter, defaultdict

from glycokit.standardize.constants import CHAIN_POOL, POLY
from glycokit.standardize.model import _owner


def _next_free(letter, used):
    """
    First chain letter not in ``used``, starting at ``letter``.

    :raises ValueError: If all 62 chain IDs are in use.
    :rtype: str
    """
    pool = string.ascii_uppercase + string.ascii_lowercase + string.digits
    i = pool.index(letter) if letter in pool else 0
    for k in range(len(pool)):
        c = pool[(i + k) % len(pool)]
        if c not in used:
            return c
    raise ValueError("no free chain identifiers")


def _split_repeated_chains(res, used, opt, rep):
    """
    Give each repeated copy of a chain its own (chain, segid).

    A copy starts only when a *protein* residue's (resid, icode) recurs within
    the current copy of its original chain and is not the residue immediately
    before it. Ligands, ions, caps and split fragments never start a copy;
    they follow the current copy of their chain (any remaining collision is
    resolved with an insertion code in step 6 of :func:`_assign_ids`).

    New chain letters are used while available (the glycan chain letter is
    reserved). After that, ``chain_overflow='segid'`` reuses the original
    letter and gives the copy a unique segid; ``chain_overflow='error'`` raises.

    :return: The updated set of used chain letters.
    :rtype: set[str]
    """
    used = set(used) | {opt.glycan_chain}
    used_seg = {r.segid for r in res if r.segid} | {r.chain[:4] for r in res if r.chain}
    copy_no, assign, prev, seen = {}, {}, {}, set()
    copies, overflowed = [], False

    for r in res:
        if r.kind == "glycan" or not r.chain:
            continue
        orig = r.chain
        n = copy_no.setdefault(orig, 1)
        if r.kind == "protein" and not r.split:
            here = (r.resid, r.icode)
            if (orig, n) + here in seen and prev.get(orig) != here:
                n = copy_no[orig] = n + 1
                try:
                    ch = _next_free(orig, used)
                    used.add(ch)
                    seg = ch[:4]
                except ValueError:
                    if opt.chain_overflow == "error":
                        raise ValueError(
                            f"chain {orig!r} repeats at resid {r.resid}{r.icode} but all "
                            "62 PDB chain IDs are in use; use --chain-overflow segid "
                            "or write mmCIF"
                        ) from None
                    ch, seg, overflowed = orig, None, True
                if seg is None or seg in used_seg:
                    k = n
                    seg = f"{orig[:1]}{k}"
                    while seg in used_seg:
                        k += 1
                        seg = f"{orig[:1]}{k}"
                    if len(seg) > 4:
                        raise ValueError(
                            f"ran out of 4-character segids for chain {orig!r}"
                        )
                used_seg.add(seg)
                assign[(orig, n)] = (ch, seg)
                copies.append(
                    {
                        "orig_chain": orig,
                        "copy": n,
                        "chain": ch,
                        "segid": seg,
                        "first_resid": f"{r.resid}{r.icode}",
                    }
                )
            prev[orig] = here
            seen.add((orig, n) + here)
        if (orig, n) in assign:
            r.chain, r.segid = assign[(orig, n)]

    rep["chain_copies"] = copies
    if copies:
        shown = "; ".join(
            f"{c['orig_chain']} copy {c['copy']} -> chain {c['chain']}/"
            f"segid {c['segid']} (from resid {c['first_resid']})"
            for c in copies[:10]
        )
        more = f"; ... and {len(copies) - 10} more" if len(copies) > 10 else ""
        rep["warnings"].append(
            f"{len(copies)} repeated chain copies relabelled: "
            f"{shown}{more} (see 'chain_copies')"
        )
    if overflowed:
        rep["warnings"].append(
            "more than 62 chains: chain letters reused; copies are distinguished "
            "by segid (cols 73-76) only. Tools that read only the chain column "
            "will see duplicate chains."
        )
    return used


def _glycan_attachment(res, n_atoms, link_pairs):
    """
    Map glycan residue index -> frozenset of (segid, chain) polymer chains it is
    covalently attached to.

    Glycan-glycan links are followed, so whole trees inherit the anchor of
    their root. The anchors use the residues' CURRENT labels, so this must
    run after the polymer chains/segids are final (step 5 of
    :func:`_assign_ids`).

    :param list res: Residues.
    :param int n_atoms: Number of atoms.
    :param link_pairs: Inter-residue bonds involving glycans (atom pairs).
    :rtype: dict[int, frozenset[tuple[str, str]]]
    """
    owner = _owner(res, n_atoms)
    adj = defaultdict(set)
    for i, j in link_pairs:
        a, b = int(owner[i]), int(owner[j])
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    attach = {}
    for k, r in enumerate(res):
        if r.kind != "glycan" or k in attach:
            continue
        comp, anchors, stack = set(), set(), [k]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            for y in adj[x]:
                if res[y].kind == "glycan":
                    stack.append(y)
                elif res[y].kind in POLY:
                    anchors.add((res[y].segid, res[y].chain))
        anchors = frozenset(anchors)
        for x in comp:
            attach[x] = anchors
    return attach


def _unique_segid(base, used_seg):
    """
    A segid derived from ``base`` that is not in ``used_seg`` (added to it).

    :rtype: str
    """
    base = (base or "G")[:4]
    s, n = base, 2
    while s in used_seg:
        suf = str(n)
        s = base[: 4 - len(suf)] + suf
        n += 1
    used_seg.add(s)
    return s


def _glycan_chain_for(pchain, poly_used, glycan_owner):
    """
    Preferred glycan chain for protein chain ``pchain``: its case-swapped letter.

    :param str pchain: Polymer chain the glycans are attached to.
    :param set poly_used: Chain letters used by polymers.
    :param dict glycan_owner: Glycan chain letter -> polymer chain owning it.
    :return: ``(chain or None, preferred)``
    :rtype: tuple[str or None, bool]
    """
    c0 = pchain[:1]
    want = c0.lower() if c0.isupper() else c0.upper() if c0.islower() else None

    def ok(c):
        return c not in poly_used and glycan_owner.get(c, pchain) == pchain

    if want and ok(want):
        return want, True
    for c in CHAIN_POOL:
        if c not in poly_used and c not in glycan_owner:
            return c, False
    return None, False


def _assign_glycan_chains(res, opt, rep, attach):
    """
    Chain, segid and sequential resids for glycans (``glycan_chain_mode='attached'``).

    Glycans attached to a polymer chain get its case-swapped letter and a
    segid ``G<segid>``; unattached glycans go to ``opt.glycan_chain``. Trees
    bridging several chains are assigned to the first chain in input order.
    The result is listed in ``rep['glycan_chains']``.

    :param dict attach: Output of :func:`_glycan_attachment` (final labels).
    """
    gly = [k for k, r in enumerate(res) if r.kind == "glycan"]
    rep["glycan_chains"] = []
    if not gly:
        return
    poly_used = {r.chain for r in res if r.kind != "glycan" and r.chain}
    used_seg = {r.segid for r in res if r.kind != "glycan" and r.segid}

    if opt.glycan_chain_mode != "attached":
        attach = {}

    order = []
    for r in res:
        if r.kind in POLY and (r.segid, r.chain) not in order:
            order.append((r.segid, r.chain))
    rank = {k: i for i, k in enumerate(order)}

    groups, glycan_owner, next_resid = (
        {},
        {},
        defaultdict(lambda: opt.glycan_resid_start),
    )
    bridged, unattached = [], 0

    def new_group(anchor):
        if anchor is None:
            ch = opt.glycan_chain
            if ch in poly_used or glycan_owner.get(ch, None) not in (
                None,
                "<unattached>",
            ):
                ch = next(
                    (
                        c
                        for c in CHAIN_POOL
                        if c not in poly_used and c not in glycan_owner
                    ),
                    opt.glycan_chain,
                )
                rep["warnings"].append(
                    f"glycan chain {opt.glycan_chain!r} in use; "
                    f"unattached glycans placed in {ch!r}"
                )
            glycan_owner.setdefault(ch, "<unattached>")
            return ch, _unique_segid(opt.glycan_segid or ch, used_seg)
        pseg, pch = anchor
        ch, preferred = _glycan_chain_for(pch, poly_used, glycan_owner)
        if ch is None:
            ch = pch[:1].lower()
            rep["warnings"].append(
                f"no free chain ID for glycans of {pseg}/{pch}; "
                f"reusing {ch!r}, distinguished by segid only"
            )
        elif not preferred:
            rep["warnings"].append(
                f"glycans of chain {pch!r} placed in chain {ch!r} "
                "(case-swapped letter unavailable)"
            )
        glycan_owner.setdefault(ch, pch)
        return ch, _unique_segid("G" + (pseg or pch), used_seg)

    for k in gly:
        r = res[k]
        anchors = sorted(attach.get(k, ()), key=lambda a: rank.get(a, len(rank)))
        anchor = anchors[0] if anchors else None
        if len(anchors) > 1:
            bridged.append(f"{r.orig['chain']}:{r.resname}:{r.orig['resid']}")
        if anchor is None:
            unattached += 1
        if anchor not in groups:
            groups[anchor] = new_group(anchor)
        ch, seg = groups[anchor]
        r.chain, r.segid, r.icode = ch, seg, ""
        r.resid = next_resid[(seg, ch)]
        next_resid[(seg, ch)] += 1

    rep["glycan_chains"] = [
        {
            "protein": None if a is None else f"{a[0]}/{a[1]}",
            "chain": c,
            "segid": s,
            "n_residues": next_resid[(s, c)] - opt.glycan_resid_start,
        }
        for a, (c, s) in groups.items()
    ]
    if bridged:
        rep["warnings"].append(
            f"{len(bridged)} glycan residue(s) belong to trees linked to "
            f"more than one chain; assigned to the first, e.g. {bridged[:5]}"
        )
    if unattached and opt.glycan_chain_mode == "attached":
        rep["warnings"].append(
            f"{unattached} glycan residue(s) not linked to any polymer "
            f"chain; placed in {groups[None][0]!r} (check --link-cutoff)"
        )


def _assign_ids(res, opt, rep, link_pairs, n_atoms):
    """
    Assign final chain, segid, resid, icode and record labels.

    Steps:

    1. blank polymer chain IDs (GRO) get letters; a new chain starts only
       where the numbering restarts
    2. unchained solvent / ions / ligands get their own chain
    3. repeated chain copies get their own (chain, segid)
    4. blank segids follow the chain
    5. glycan attachment is computed from ``link_pairs`` with the now-final
       polymer labels; glycans get their chains and sequential resids
    6. remaining label collisions get a free insertion code
    7. record types

    :param list res: Residues (modified in place).
    :param link_pairs: Inter-residue glycan bonds from :func:`_link_pairs`.
    :param int n_atoms: Number of atoms.
    :return: Glycan attachment map (see :func:`_glycan_attachment`), keyed by
        residue index and labelled with the final polymer (segid, chain).
    :rtype: dict
    :raises ValueError: On duplicate labels or resids below -999.
    """
    used = {r.chain for r in res if r.chain}

    # 1. blank polymer chain IDs (GRO): new chain only where numbering restarts.
    #    Gaps (missing residues) do NOT start a new chain.
    poly = [r for r in res if r.kind in POLY]
    if poly and all(not r.chain for r in poly):
        letter = opt.protein_chain_default
        if letter in used:
            letter = _next_free(letter, used)
        prev = None
        for r in poly:
            if (
                prev is not None
                and r.kind == "protein"
                and not r.split
                and r.resid <= prev.resid
            ):
                used.add(letter)
                letter = _next_free(letter, used)
            r.chain, prev = letter, r
        used.add(letter)

    # 2. solvent / ions / ligands without chain IDs get their own chain
    blank_other = [r for r in res if r.kind == "other" and not r.chain]
    if blank_other:
        wchain = _next_free("W", used)
        used.add(wchain)
        for k, r in enumerate(blank_other):
            r.chain = wchain
            if opt.renumber_blank_other:
                r.resid = k + 1
        rep["info"].append(
            f"{len(blank_other)} unchained non-polymer residues placed in "
            f"chain {wchain!r}"
            + (" and renumbered" if opt.renumber_blank_other else "")
        )

    # 3. repeated copies with identical labels -> one (chain, segid) per copy
    _split_repeated_chains(res, used, opt, rep)

    # 4. blank segids follow the chain (MDAnalysis groups residues partly by segid)
    for r in res:
        if r.kind != "glycan" and not r.segid:
            r.segid = r.chain[:4]

    # 5. glycans: the polymer labels are final now, so the attachment anchors
    #    stay valid for TER placement and reordering. Chain = case-swapped
    #    chain of the attached polymer, sequential resids per glycan chain;
    #    unattached glycans -> opt.glycan_chain
    attach = _glycan_attachment(res, n_atoms, link_pairs)
    if opt.glycan_chain_mode == "single":
        used = {r.chain for r in res if r.kind != "glycan" and r.chain}
        gchain = (
            opt.glycan_chain
            if opt.glycan_chain not in used
            else _next_free(opt.glycan_chain, used)
        )
        gsegid = (opt.glycan_segid or gchain)[:4]
        for k, r in enumerate(x for x in res if x.kind == "glycan"):
            r.chain, r.resid, r.icode, r.segid = (
                gchain,
                opt.glycan_resid_start + k,
                "",
                gsegid,
            )
    else:
        _assign_glycan_chains(res, opt, rep, attach)

    # 6. any remaining label collision (split pieces, caps, microheterogeneity,
    #    ligands numbered like protein residues) gets a free insertion code
    taken, pending = defaultdict(set), []
    for r in res:
        if r.kind == "glycan":
            continue
        key = (r.segid, r.chain, r.resid)
        if r.split or r.kind == "cap" or r.icode in taken[key]:
            pending.append(r)
        else:
            taken[key].add(r.icode)
    moved = []
    for r in pending:
        s = taken[(r.segid, r.chain, r.resid)]
        if r.icode in s:
            free = [c for c in string.ascii_uppercase if c not in s]
            if not free:
                raise ValueError(
                    f"no free insertion code for {r.segid}/{r.chain}:{r.resid}"
                )
            r.icode = free[0]
            r.note = (r.note + "; " if r.note else "") + f"icode {r.icode!r} assigned"
            if not (r.split or r.kind == "cap"):
                moved.append(f"{r.chain}:{r.resname}:{r.resid}{r.icode}")
        s.add(r.icode)
    if moved:
        rep["warnings"].append(
            f"{len(moved)} residue(s) shared a label with another residue and got an "
            f"insertion code, e.g. {', '.join(moved[:10])}"
        )

    # 7. record types
    for r in res:
        if r.kind in POLY:
            r.record = "ATOM"
        elif r.kind == "glycan":
            r.record = (
                (r.orig["record"] or "ATOM")
                if opt.glycan_record == "keep"
                else opt.glycan_record
            )
        else:
            r.record = r.record or "HETATM"

    labels = Counter((r.segid, r.chain, r.resid, r.icode) for r in res)
    dups = [k for k, v in labels.items() if v > 1]
    if dups:
        raise ValueError(
            f"duplicate (segid, chain, resid, icode) labels after "
            f"standardization: {dups[:10]}"
        )
    bad = [r for r in res if r.resid < -999]
    if bad:
        raise ValueError(f"resid below -999 for {len(bad)} residues")
    return attach
