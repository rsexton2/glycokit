"""Glycan detection: name lists, PDB CCD and the sugar-ring structure test."""

import warnings
from collections import defaultdict
from functools import lru_cache

import numpy as np

from glycokit.standardize.constants import (
    _CCD_DATA_ERRORS,
    CAPS,
    CHARMM_GLYCANS,
    GLYCAM_PATTERN,
    HIS_ALIASES,
    KNOWN_GLYCANS,
    PROTEIN,
    SUGAR_ELEMENTS,
    SUGAR_HEAVY_RANGE,
    SUGAR_TEST_INSTANCES,
    WATER_IONS,
)
from glycokit.standardize.model import StandardizeOptions
from glycokit.standardize.reading import _atom_table, _segment


@lru_cache(maxsize=1)
def _ccd_carbohydrates():
    """
    Carbohydrate component names from the PDB CCD via biotite (optional).

    :return: Upper-case names; empty if biotite or its data is unavailable.
    :rtype: frozenset[str]
    """
    try:
        # optional dependency: imported on first use only
        # pylint: disable-next=import-outside-toplevel
        from biotite.structure.info import carbohydrate_names
    except ImportError:  # not installed, or too old for carbohydrate_names
        return frozenset()

    try:
        names = carbohydrate_names()
    except _CCD_DATA_ERRORS as exc:
        warnings.warn(
            f"biotite is installed but its CCD carbohydrate data could not "
            f"be read ({exc}); CCD-based glycan detection is disabled",
            stacklevel=2,
        )
        return frozenset()
    return frozenset(str(n).upper() for n in names)


def _base_kind(resname):
    """
    Kind decided by the residue name alone.

    :return: ``'protein'``, ``'cap'``, ``'solvent'``, or None if the name
        needs further classification (glycan or other).
    :rtype: str or None
    """
    r = resname.upper()
    if r in PROTEIN or r in HIS_ALIASES:
        return "protein"
    if r in CAPS:
        return "cap"
    if r in WATER_IONS:
        return "solvent"
    return None


def _el(t, i):
    """Element, falling back to the atom name (elements are filled in later)."""
    e = t["element"][i]
    if e:
        return e.upper()
    return t["name"][i].upper().lstrip("0123456789")[:1]


def _heavy_graph(r, t, pos, bond_adj):
    """
    Heavy-atom connectivity inside one residue: distance-based plus topology bonds.

    :return: ``(idx, el, adj)``: heavy-atom indices, their elements and the
        adjacency sets.
    :rtype: tuple[list[int], dict, dict]
    """
    idx = [i for i in range(r.start, r.stop) if _el(t, i) != "H"]
    el = {i: _el(t, i) for i in idx}
    adj = {i: set() for i in idx}
    if len(idx) > 1:
        p = pos[idx]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        big = np.array([el[i] in ("S", "P") for i in idx])
        cut = np.where(big[:, None] | big[None, :], 1.90, 1.70)
        for a, b in np.argwhere(np.triu((d > 0.9) & (d <= cut), 1)):
            i, j = idx[a], idx[b]
            adj[i].add(j)
            adj[j].add(i)
    for i in idx:
        for j in bond_adj.get(i, ()):
            if j in adj:
                adj[i].add(j)
                adj[j].add(i)
    return idx, el, adj


def _ring_ok(ring, adj, el, pos):
    """
    Whether a candidate ring looks like a saturated sugar ring.

    Rejects aromatic rings, lactones and ascorbate-like enols (short ring
    bonds) and nucleosides (base nitrogen next to the ring oxygen); requires
    at least two ring carbons with an O or N substituent.

    :param list ring: ``[ring O, C (next to O), ..., C (next to O)]``.
    :param dict adj: Heavy-atom adjacency.
    :param dict el: Element per atom.
    :param numpy.ndarray pos: Coordinates.
    :rtype: bool
    """
    o, carbons = ring[0], ring[1:]
    members = set(ring)

    def dist(x, y):
        return float(np.linalg.norm(pos[x] - pos[y]))

    # saturated ring only: rejects aromatic rings, lactones, ascorbate-like enols
    if any(dist(x, y) < 1.45 for x, y in zip(carbons, carbons[1:])):
        return False
    if min(dist(o, carbons[0]), dist(o, carbons[-1])) < 1.38:
        return False
    exo = {c: [n for n in adj[c] if n not in members] for c in carbons}
    # nucleosides/nucleotides: base N on a carbon next to the ring oxygen
    if any(el[n] == "N" for c in (carbons[0], carbons[-1]) for n in exo[c]):
        return False
    # at least two ring carbons carry an O or N substituent
    return sum(any(el[n] in ("O", "N") for n in exo[c]) for c in carbons) >= 2


def _sugar_ring(adj, el, pos):
    """
    Find a 5/6-membered ring with exactly one O (the rest carbon) that passes
    :func:`_ring_ok`.

    :return: The ring as ``[O, C, ..., C]``, or None.
    :rtype: list[int] or None
    """
    for o in sorted(adj):
        nbs = adj[o]
        if el[o] != "O" or len(nbs) != 2 or any(el[n] != "C" for n in nbs):
            continue
        a, b = sorted(nbs)
        stack = [[a]]
        while stack:
            path = stack.pop()
            for n in adj[path[-1]]:
                if n == b:
                    if len(path) in (3, 4):  # 5- or 6-membered ring
                        ring = [o, *path, b]
                        if _ring_ok(ring, adj, el, pos):
                            return ring
                elif n != o and n not in path and el[n] == "C" and len(path) < 4:
                    stack.append(path + [n])
    return None


def _looks_like_sugar(r, t, pos, bond_adj):
    """
    Structure test for one residue: size, elements and a sugar ring.

    :return: ``(passed, reason)``
    :rtype: tuple[bool, str]
    """
    idx, el, adj = _heavy_graph(r, t, pos, bond_adj)
    lo, hi = SUGAR_HEAVY_RANGE
    if not lo <= len(idx) <= hi:
        return False, f"{len(idx)} heavy atoms"
    odd = set(el.values()) - SUGAR_ELEMENTS
    if odd:
        return False, f"contains {','.join(sorted(odd))}"
    ring = _sugar_ring(adj, el, pos)
    if ring is None:
        return (
            False,
            "no saturated 5/6-ring with one ring O and >=2 substituted carbons",
        )
    return (
        True,
        f"{len(ring)}-membered sugar ring {'-'.join(t['name'][i] for i in ring)}",
    )


def _classify(res, t, pos, bonds, opt, rep):
    """
    Set ``r.kind`` for every residue. Decisions are made per residue name.

    Order: protein/cap/solvent names, ``--not-glycan``, name lists
    (``--glycan-resnames``, built-in, CHARMM, PDB CCD), then the structure
    test in ``glycan_guess='auto'``. Decisions are logged in
    ``rep['glycan_classification']``.

    :raises ValueError: If a name is given as both glycan and not-glycan.
    """
    explicit = {g.upper() for g in (opt.glycan_resnames or ())}
    excluded = {g.upper() for g in (opt.not_glycan or ())}
    both = explicit & excluded
    if both:
        raise ValueError(
            f"resnames given to both --glycan-resnames and --not-glycan: {sorted(both)}"
        )
    sources = {
        "--glycan-resnames": explicit,
        "built-in": {g.upper() for g in KNOWN_GLYCANS},
    }
    if opt.glycan_guess in ("names", "auto"):
        sources["CHARMM"] = CHARMM_GLYCANS
        sources["PDB CCD"] = _ccd_carbohydrates()
        if not sources["PDB CCD"]:
            rep["info"].append(
                "biotite not available: PDB CCD carbohydrate list not used"
            )

    bond_adj = defaultdict(set)
    for i, j in bonds:
        bond_adj[i].add(j)
        bond_adj[j].add(i)

    by_name = defaultdict(list)
    for r in res:
        by_name[r.resname].append(r)

    log, guessed, non_glycan, mixed = [], [], [], []
    for name, rs in sorted(by_name.items()):
        base = _base_kind(name)
        if base:
            for r in rs:
                r.kind = "other" if base == "solvent" else base
            continue
        up = name.upper()
        entry = {"resname": name, "n_residues": len(rs)}
        if up in excluded:
            kind, entry["reason"] = "other", "--not-glycan"
        else:
            src = next((s for s, names in sources.items() if up in names), None)
            if src:
                kind, entry["reason"] = "glycan", f"name ({src})"
            elif opt.glycan_guess == "auto":
                tested = rs[:SUGAR_TEST_INSTANCES]
                results = [_looks_like_sugar(r, t, pos, bond_adj) for r in tested]
                n_pass = sum(ok for ok, _ in results)
                kind = (
                    "glycan"
                    if n_pass / len(tested) >= opt.glycan_min_fraction
                    else "other"
                )
                why = next(w for ok, w in results if ok == (kind == "glycan"))
                entry.update(
                    reason=f"structure: {why}", tested=len(tested), passed=n_pass
                )
                if GLYCAM_PATTERN.match(name):
                    entry["reason"] += " (matches GLYCAM naming pattern)"
                if kind == "glycan":
                    guessed.append(name)
                if 0 < n_pass < len(tested):
                    mixed.append(f"{name} ({n_pass}/{len(tested)})")
            else:
                kind, entry["reason"] = "other", "not in name lists"
        if kind == "other":
            non_glycan.append(name)
        entry["kind"] = kind
        for r in rs:
            r.kind = kind
        log.append(entry)

    rep["glycan_classification"] = log
    if guessed:
        rep["warnings"].append(
            f"glycan resnames guessed from structure: {guessed} "
            "(see 'glycan_classification'; override with --not-glycan)"
        )
    if mixed:
        rep["warnings"].append(
            f"resnames with inconsistent sugar-test results (passed/tested): {mixed}"
        )
    if non_glycan:
        rep["warnings"].append(
            f"residues treated as non-glycan hetero groups: {non_glycan} "
            "(use --glycan-resnames if any are sugars)"
        )

    # CHARMM-GUI truncates e.g. BGLCNA -> BGLC, which looks like plain glucose
    trunc = sorted(
        {
            r.resname
            for r in res
            if r.kind == "glycan"
            and r.resname.upper() in {"AGLC", "BGLC", "AGAL", "BGAL"}
            and "N2" in t["name"][r.start : r.stop]
        }
    )
    if trunc:
        rep["warnings"].append(
            f"{trunc} contain an N2 atom: probably truncated CHARMM GlcNAc/GalNAc "
            "names (BGLCNA -> BGLC); still treated as glycans"
        )


def guess_glycans(universe, glycan_guess="auto", glycan_resnames=None, not_glycan=None):
    """
    Glycan residue names of ``universe``, decided like ``glyco standardize``.

    Uses the coordinates of the current frame for the sugar-ring test.

    :param universe: Loaded Universe.
    :param str glycan_guess: ``off``, ``names`` or ``auto``.
    :param glycan_resnames: Extra names always treated as glycans.
    :param not_glycan: Names never treated as glycans.
    :return: ``(names, report)``: sorted glycan residue names as written in
        the topology, and a report with ``warnings``, ``info`` and
        ``glycan_classification``.
    :rtype: tuple[list[str], dict]
    :raises ValueError: If a name is in both glycan_resnames and not_glycan.
    """
    rep = {"warnings": [], "info": []}
    if universe.atoms.n_atoms == 0:
        return [], rep
    t, pos, bonds = _atom_table(universe.atoms)
    res = _segment(t)
    opt = StandardizeOptions(
        glycan_guess=glycan_guess,
        glycan_resnames=tuple(glycan_resnames) if glycan_resnames else None,
        not_glycan=tuple(not_glycan) if not_glycan else None,
    )
    _classify(res, t, pos, bonds, opt, rep)
    names = sorted({r.resname for r in res if r.kind == "glycan"})
    return names, rep


def glycan_selection(names):
    """MDAnalysis selection of the residue names ``names`` ('' if none)."""
    return f"resname {' '.join(names)}" if names else ""
