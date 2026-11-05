"""
PDB output that keeps the chain and segment IDs of the Universe.

Written by MDAnalysis' PDBWriter; afterwards column 22 (chain ID) and
columns 73-76 (segment ID) of every ATOM/HETATM record are set from the
Universe, because the writer replaces blank or non-standard chain IDs
with 'X'. Only MDAnalysis is imported, so viewer_scripts.py can use it.
"""

import warnings
from pathlib import Path

from MDAnalysis.exceptions import NoDataError

PDB_SUFFIXES = (".pdb", ".ent")
#: largest atom serial a fixed-column PDB CONECT record can hold
PDB_MAX_SERIAL = 99999
#: MDAnalysis' segid for topologies without segments; written as blank
_DEFAULT_SEGID = "SYSTEM"
_RECORDS = ("ATOM  ", "HETATM")

_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: PDB segID columns 73-76
PDB_SEGID_WIDTH = 4


def _pdb_char(c):
    return c.isascii() and c.isalnum()


def _distinct_position(ids, taken):
    """Index (start first, then end) where every ID in ``ids`` has a distinct
    letter/digit that is not in ``taken``; None if there is none."""
    n = min(len(i) for i in ids)
    for k in [*range(n), *range(-1, -n - 1, -1)]:
        chars = [i[k] for i in ids]
        if (all(_pdb_char(c) and c not in taken for c in chars)
                and len(set(chars)) == len(chars)):
            return k
    return None


def _match_characters(ids, taken):
    """One distinct character of each ID (bipartite matching), or None."""
    options = {
        i: [c for c in dict.fromkeys(i) if _pdb_char(c) and c not in taken]
        for i in ids
    }
    owner = {}

    def assign(i, seen):
        for c in options[i]:
            if c in seen:
                continue
            seen.add(c)
            if c not in owner or assign(owner[c], seen):
                owner[c] = i
                return True
        return False

    for i in ids:
        if not assign(i, set()):
            return None
    return {i: c for c, i in owner.items()}


def pdb_chain_map(chains):
    """
    ``(mapping, message)``: one PDB chain character per chain ID.

    Single-character IDs are kept; longer IDs get the character that tells
    them apart (a common position, else one distinct character of each ID,
    else the first character). ``message`` is None if nothing changed.
    """
    ids = list(dict.fromkeys(str(c).strip() for c in chains))
    mapping = {c: c for c in ids if len(c) <= 1}
    long = [c for c in ids if len(c) > 1]
    if not long:
        return mapping, None
    taken = {c for c in mapping if c}

    k = _distinct_position(long, taken)
    if k is not None:
        chosen = {c: c[k] for c in long}
        how = (
            f"character {k + 1}" if k >= 0 else f"character {-k} from the end")
    else:
        chosen = _match_characters(long, taken)
        how = "a distinct character of each ID"
    merged = chosen is None
    if merged:
        chosen = {c: c[0] for c in long}
        how = "the first character"
    mapping.update(chosen)

    shown = ", ".join(f"{c}->{chosen[c]}" for c in long[:8])
    more = f" (+{len(long) - 8} more)" if len(long) > 8 else ""
    msg = (
        f"chain IDs longer than 1 character are written as {how}: "
        f"{shown}{more}" +
        (" (different chain IDs become equal)" if merged else ""))
    return mapping, msg


def has_bonds(atoms):
    """True if the topology defines at least one bond for ``atoms``."""
    try:
        return len(atoms.bonds) > 0
    except (AttributeError, NoDataError):
        return False


def _strings(atoms, name):
    try:
        values = getattr(atoms, name)
    except (AttributeError, NoDataError):
        return None
    return [str(v).strip() for v in values]


def pdb_ids(atoms):
    """``(chains, segids, messages)`` as written to the PDB columns."""
    n = atoms.n_atoms
    chains = _strings(atoms, "chainIDs") or [""] * n
    segids = _strings(atoms, "segids") or [""] * n
    segids = ["" if s == _DEFAULT_SEGID else s for s in segids]
    chain_map, m1 = pdb_chain_map(chains)
    segid_map, m2 = pdb_segid_map(segids)
    chains = [chain_map[c] for c in chains]
    segids = [segid_map[s] for s in segids]
    return chains, segids, [m for m in (m1, m2) if m]


def _patch_ids(path, chains, segids):
    """Set column 22 and columns 73-76 of every ATOM/HETATM record."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    k = 0
    for i, line in enumerate(lines):
        if not line.startswith(_RECORDS):
            continue
        if k == len(chains):
            raise RuntimeError(f"{path.name}: more atom records than atoms")
        body = line.rstrip("\r\n")
        end = line[len(body):]
        body = body.ljust(80)
        body = (
            body[:21] + (chains[k] or " ") + body[22:72] + f"{segids[k]:<4}" +
            body[76:])
        lines[i] = body + end
        k += 1
    if k != len(chains):
        raise RuntimeError(
            f"{path.name}: {k} atom records for {len(chains)} atoms")
    path.write_text("".join(lines), encoding="utf-8")


def write_pdb(atoms, output, *, bonds="auto"):
    """
    PDB of ``atoms`` (current frame) with the Universe's chain and segment
    IDs (see :func:`pdb_ids` for IDs that do not fit).

    :param bonds: ``"auto"``: every topology bond (read or guessed) as
        CONECT, if ``atoms`` is the whole Universe; otherwise as
        MDAnalysis' ``bonds`` (``"all"``, ``"conect"`` or None).
    :rtype: pathlib.Path
    """
    output = Path(output)
    if output.suffix.lower() not in PDB_SUFFIXES:
        raise ValueError(f"write_pdb writes PDB files, got {output.name}")
    whole = atoms.n_atoms == atoms.universe.atoms.n_atoms
    if bonds == "auto":
        bonds = "all" if whole and has_bonds(atoms) else None
    if bonds is not None and atoms.n_atoms > PDB_MAX_SERIAL:
        warnings.warn(
            f"{output.name}: {atoms.n_atoms} atoms exceed the PDB serial "
            f"limit ({PDB_MAX_SERIAL}); CONECT records are not written",
            stacklevel=2)
        bonds = None

    chains, segids, messages = pdb_ids(atoms)
    for msg in messages:
        warnings.warn(f"{output.name}: {msg}", stacklevel=2)
    with warnings.catch_warnings():
        # the chain IDs are set below, so the writer's 'X' warning is moot
        warnings.filterwarnings("ignore", message=r"(?s).*chainID")
        atoms.write(str(output), bonds=bonds)
    _patch_ids(output, chains, segids)
    return output


def _abbreviate(ident, width, k):
    """First ``width - k`` characters plus the last ``k``."""
    return ident[:width - k] + (ident[len(ident) - k:] if k else "")


def _base36(n, digits):
    out = ""
    for _ in range(digits):
        n, r = divmod(n, 36)
        out = _BASE36[r] + out
    return out


def _numbered(ids, width, taken):
    """Leading characters plus a base-36 counter, skipping ``taken``;
    None if ``width`` cannot hold enough distinct values."""
    digits = 1
    while 36**digits < len(ids) + len(taken):
        digits += 1
    if digits >= width:
        return None
    used, out, n = set(taken), {}, 0
    for ident in ids:
        while True:
            cand = ident[:width - digits] + _base36(n, digits)
            n += 1
            if cand not in used:
                break
        used.add(cand)
        out[ident] = cand
    return out


def _mapping_message(label, width, how, chosen, merged):
    long = list(chosen)
    shown = ", ".join(f"{i}->{chosen[i]}" for i in long[:8])
    more = f" (+{len(long) - 8} more)" if len(long) > 8 else ""
    return (
        f"{label} longer than {width} character(s) are written as "
        f"{how}: {shown}{more}" +
        (f" (different {label} become equal)" if merged else ""))


def pdb_segid_map(segids, width=PDB_SEGID_WIDTH):
    """
    ``(mapping, message)``: a PDB segID (at most ``width`` characters) for
    every segment ID.

    Short IDs are kept; longer IDs keep a prefix plus the shortest tail
    that tells them apart (GLYCAN1, GLYCAN2 -> GLY1, GLY2), else a
    numbered prefix, else plain truncation. ``message`` is None if
    nothing changed.
    """
    ids = list(dict.fromkeys(str(s).strip() for s in segids))
    mapping = {s: s for s in ids if len(s) <= width}
    long = [s for s in ids if len(s) > width]
    if not long:
        return mapping, None
    taken = {s for s in mapping if s}

    chosen, how = None, None
    for k in range(width + 1):
        cand = {s: _abbreviate(s, width, k) for s in long}
        values = list(cand.values())
        if len(set(values)) == len(values) and not taken & set(values):
            chosen = cand
            how = (
                "the first characters" if k == 0 else
                f"the first {width - k} plus the last {k} character(s)")
            break
    if chosen is None:
        chosen = _numbered(long, width, taken)
        how = "a numbered prefix"
    merged = chosen is None
    if merged:
        chosen = {s: s[:width] for s in long}
        how = "the first characters"
    mapping.update(chosen)
    msg = _mapping_message("segment IDs", width, how, chosen, merged)
    return mapping, msg
