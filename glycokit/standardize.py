"""glyco_standardize.py -- normalise structure inputs for GLYCO / glycokit.

Guarantees (default options):

* atom order and atom count are preserved (trajectories stay valid)
* every residue has a unique (segid, chain, resid, icode) label; repeated
  copies (e.g. multiple assemblies in one file) get new chain IDs, and once
  the 62 single-character PDB chain IDs are exhausted, chain letters are
  reused and copies are distinguished by segid (cols 73-76)
* glycans get their own chain/segid and sequential resids; glycans attached
  to a polymer chain use the case-swapped letter of that chain
* merged residues (repeated atom names, altloc-aware) are split and given
  distinct insertion codes; other label collisions (ligands numbered like
  protein residues, microheterogeneity) are resolved the same way
* chain gaps are allowed and reported; truncated side chains are an error
  unless incomplete_sidechains='warn'
* fixed-column PDB output (wwPDB v3.3), formal charges, CONECT for hetero
  groups / links, CRYST1 preserved (no placeholder written)

Outputs: <prefix>.pdb, <prefix>_residues.csv, <prefix>_report.json
(+ <prefix>_atom_map.csv and a reordered trajectory when atoms are moved).
"""

from __future__ import annotations

import csv
import json
import re
import string
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from glycokit.console import heading, info, set_color_mode, style, warning
from glycokit.parsers import standardize_parser

# ----------------------------------------------------------------- constants
# fmt: off
PROTEIN = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
           "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
           "SEC", "PYL", "CYX", "ASH", "GLH", "LYN"}
HIS_ALIASES = {"HSD", "HSE", "HSP", "HID", "HIE", "HIP"}
CAPS = {"ACE", "NME", "NMA", "NH2"}
WATER_IONS = {"HOH", "WAT", "SOL", "TIP3", "TIP", "T3P", "TIP4", "T4P", "SPC",
              "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "CD", "CO", "NI",
              "CU", "FE", "FE2", "LI", "RB", "CS", "SR", "BA", "HG", "IOD",
              "BR", "F", "NA+", "CL-", "SOD", "CLA", "POT", "CAL", "CES",
              "SO4", "PO4", "ACT", "GOL", "EDO", "PEG", "FMT", "NO3", "SCN"}

# Default glycan residue names (PDB CCD, CHARMM 4-char, a few GLYCAM). Extend with
# --glycan-resnames; anything else non-protein is treated as 'other' with a warning.
KNOWN_GLYCANS = {"NAG", "NDG", "BMA", "MAN", "GAL", "GLA", "GLC", "BGC", "FUC",
                 "FUL", "SIA", "SLB", "NGA", "A2G", "XYS", "XYP", "GCU", "IDR",
                 "FRU", "RAM", "KDN", "NGC", "BGLC", "AGLC", "BMAN", "AMAN",
                 "BGAL", "AGAL", "AFUC", "BFUC", "BXYL", "ANE5", "4YB", "0YB",
                 "VMB", "0MA", "2MA", "0fA"}
CAP_N = {"CAY": "CH3", "HY1": "H1", "HY2": "H2", "HY3": "H3", "CY": "C",
         "OY": "O"}
CAP_C = {"NT": "N", "HNT": "H", "CAT": "C", "HT1": "H1", "HT2": "H2",
         "HT3": "H3"}
TERM_O = {"OT1": "O", "OT2": "OXT"}
TWO_LETTER = {"CL", "NA", "MG", "ZN", "FE", "CA", "MN", "CU", "BR", "CO", "NI",
              "CD", "SE", "LI", "RB", "CS", "SR", "BA", "HG", "AL", "SI", "PT",
              "AU", "AG"}

# Heavy-atom templates; "A|B" means either name is accepted (CHARMM vs PDB naming).
BACKBONE = ["N", "CA", "C", "O|OT1|OC1"]
SIDECHAIN = {
    "ALA": ["CB"], "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"], "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"], "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"], "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1|CD"], "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"], "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"], "PRO": ["CB", "CG", "CD"],
    "SER": ["CB", "OG"], "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"], "SEC": ["CB", "SE"],
}
RES_ALIAS = {"CYX": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
             **{h: "HIS" for h in HIS_ALIASES}}
POLY = ("protein", "cap")

COMPRESSION_SUFFIXES = {".gz", ".bz2", ".xz"}

# ------------------------------------------------------------ glycan detection
# CHARMM36 carbohydrate residue names, truncated to the 4-character PDB field
CHARMM_SUGAR_CODES = ("GLC", "MAN", "GAL", "FUC", "XYL", "XYF", "RHM", "ARB",
                      "GUL", "ALT", "ALL", "TAL", "IDO", "LYF", "RIB", "FRU",
                      "NE5")
CHARMM_GLYCANS = {p + c for p in "AB" for c in CHARMM_SUGAR_CODES}

# GLYCAM: <linkage code><sugar letter><anomer/ring code>[P], e.g. 0GA, 4YB, 0fA, 0AD, 0ZBP.
# Also matches NAD/FAD etc., so it is only reported, never used on its own.
GLYCAM_PATTERN = re.compile(r"^[0-9A-Z][A-Za-z][ABDU]P?$")
SUGAR_ELEMENTS = {"C", "O", "N", "S", "P"}
SUGAR_HEAVY_RANGE = (7, 60)
SUGAR_TEST_INSTANCES = 10

# Chain letters tried for glycan chains (lowercase first)
CHAIN_POOL = string.ascii_lowercase + string.ascii_uppercase + string.digits

# Errors from reading biotite's bundled CCD data: missing or unreadable data
# files, or a data format this biotite version doesn't expect.
_CCD_DATA_ERRORS = (OSError, ValueError, KeyError)

# fmt: on


# ----------------------------------------------------------------- paths
def default_output_prefix(topology, tag="_std"):
    """
    Output prefix: the input path with ``tag`` before the file-type suffix.

    ::

        protein.pdb        -> protein_std
        /data/sys.gro      -> /data/sys_std
        model.pdb.gz       -> model_std        (compression suffix also removed)
        noext              -> noext_std

    :param topology: Input structure path.
    :type topology: str or pathlib.Path
    :param str tag: Text inserted before the suffix.
    :return: Output prefix without extension.
    :rtype: str
    """
    p = Path(topology)
    if p.suffix.lower() in COMPRESSION_SUFFIXES:
        p = p.with_suffix("")  # model.pdb.gz -> model.pdb
    return str(p.with_name(p.stem + tag))  # model.pdb    -> model_std


def normalise_output_prefix(out):
    """
    Accept ``-o`` with a ``.pdb`` extension without producing ``x.pdb.pdb``.

    :param str out: Prefix as given by the user.
    :return: Prefix without a trailing ``.pdb``.
    :rtype: str
    """
    p = Path(out)
    return str(p.with_suffix("")) if p.suffix.lower() == ".pdb" else str(p)


# ----------------------------------------------------------------- data
@dataclass
class StandardizeOptions:
    """
    Options of :func:`standardize`.

    :standardize: never modifies an options object; values resolved at run
        time (e.g. ``glycan_placement='auto'``) are only recorded in the report.
    """

    altloc: str = "keep"  # keep | first  ('first' drops atoms!)
    caps: str = "keep"  # keep | split  (CHARMM ACE/CT3 caps)
    chain_overflow: str = "segid"  # segid | error  (more copies than chain IDs)
    conect: str = "het"  # none | het | all
    cryst1: tuple | None = (
        None  # explicit (a, b, c[, alpha, beta, gamma]); written as P 1
    )
    frame: int = 0
    gap_cutoff: float = 2.0  # Angstrom, C(i)-N(i+1) distance reported as gap
    glycan_chain_mode: str = (
        "attached"  # attached: case-swapped protein chain | single: all in glycan_chain
    )
    glycan_chain: str = (
        "G"  # unattached glycans, or all with glycan_chain_mode="single"
    )
    glycan_guess: str = "auto"  # off | names | auto (names + structure test)
    glycan_min_fraction: float = 0.5  # auto: fraction of tested residues that must pass
    glycan_placement: str = "auto"  # auto | keep | after-chain | end
    glycan_record: str = "keep"  # keep | ATOM | HETATM
    glycan_resid_start: int = 1
    glycan_resnames: tuple | None = None  # extra glycan names, added to the name lists
    glycan_segid: str | None = None  # default: same as glycan_chain
    histidine: str | None = "HIS"  # None keeps HSD/HSE/...
    incomplete_sidechains: str = "error"  # error | warn
    keep_cryst1: bool = True
    link_cutoff: float = 2.0  # Angstrom, geometric glycan link detection
    long_resnames: str = "warn"  # allow | warn | error  (4-char resnames)
    not_glycan: tuple | None = None  # resnames never treated as glycans
    protein_chain_default: str = "A"  # used only when chain IDs are blank (e.g. GRO)
    renumber_blank_other: bool = True  # renumber solvent/ions that had no chain ID
    resname_map: dict[str, str] = field(default_factory=dict)
    ter_before_glycans: bool = (
        False  # True: old behaviour (TER before attached glycans)
    )
    terminal_oxygens: bool = True  # OT1/OT2 -> O/OXT
    write_ter: bool = True


@dataclass
class Residue:
    """
    One residue: a contiguous atom range ``[start, stop)`` and its labels.

    ``orig`` keeps the input labels; ``kind`` is ``protein``, ``cap``,
    ``glycan`` or ``other``; ``split`` marks pieces of a merged residue.
    """

    start: int
    stop: int
    resname: str
    chain: str
    resid: int
    icode: str
    segid: str
    record: str
    kind: str = "other"
    orig: dict = field(default_factory=dict)
    note: str = ""
    split: bool = False


# ----------------------------------------------------------------- hybrid-36
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


# ----------------------------------------------------------------- reading
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


def _read(topology, trajectory, topology_format, frame):
    """
    Load the input and extract plain per-atom data.

    :param topology: Structure / topology file.
    :param trajectory: Trajectory file, or None.
    :param topology_format: Explicit MDAnalysis format, or None.
    :param int frame: Frame whose coordinates are used.
    :return: ``(u, t, pos, bonds)``: the Universe, a dict of per-atom lists
        (name, resname, resid, chain, icode, segid, record, altloc, element,
        occ, bfac, charge), float64 positions rounded to 3 decimals, and the
        topology bonds as index pairs (empty if none).
    :rtype: tuple
    """
    kw = {"topology_format": topology_format} if topology_format else {}
    u = (
        mda.Universe(topology, trajectory, **kw)
        if trajectory
        else mda.Universe(topology, **kw)
    )
    u.trajectory[frame]  # pylint: disable=pointless-statement  # load the frame
    ag = u.atoms
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


# ----------------------------------------------------------------- steps
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


def _owner(res, n):
    """
    Residue index of every atom.

    :param list res: Residues covering atoms ``0 .. n-1``.
    :param int n: Number of atoms.
    :rtype: numpy.ndarray of int
    """
    owner = np.empty(n, dtype=int)
    for k, r in enumerate(res):
        owner[r.start : r.stop] = k
    return owner


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


# ----------------------------------------------------------------- writing
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


# ----------------------------------------------------------------- public API
def standardize(
    topology, out_prefix=None, trajectory=None, options=None, topology_format=None
):
    """
    Standardize a structure (and optionally its trajectory) for GLYCO / glycokit.

    Writes ``<prefix>.pdb``, ``<prefix>_residues.csv`` and
    ``<prefix>_report.json``; when atoms are reordered also
    ``<prefix>_atom_map.csv`` and, with a trajectory, a reordered trajectory.

    :param topology: Input structure / topology file.
    :param out_prefix: Output prefix; None uses :func:`default_output_prefix`.
    :param trajectory: Trajectory file, or None.
    :param options: Options; None uses the defaults. Never modified.
    :type options: StandardizeOptions or None
    :param topology_format: Explicit MDAnalysis topology format, or None.
    :return: The report (also written as JSON).
    :rtype: dict
    :raises ValueError: On invalid input (truncated residues, duplicate
        labels after standardization, conflicting options, ...).
    """
    out_prefix = (
        normalise_output_prefix(out_prefix)
        if out_prefix
        else default_output_prefix(topology)
    )
    opt = options or StandardizeOptions()
    rep = {
        "input": str(topology),
        "output_prefix": out_prefix,
        "trajectory": str(trajectory) if trajectory else None,
        "options": asdict(opt),
        "warnings": [],
        "info": [],
        "atom_order_preserved": True,
    }
    u, t, pos, bonds = _read(topology, trajectory, topology_format, opt.frame)
    rep["n_atoms_in"] = len(t["name"])
    rep["n_bonds_in"] = len(bonds)

    if opt.altloc == "first":
        if trajectory:
            raise ValueError(
                "altloc='first' drops atoms and would desynchronise the trajectory"
            )
        t, pos, keep = _drop_altlocs(t, pos, rep)
        remap = {old: new for new, old in enumerate(keep)}
        bonds = [(remap[i], remap[j]) for i, j in bonds if i in remap and j in remap]

    res = _segment(t)
    _classify(res, t, pos, bonds, opt, rep)

    if opt.caps == "split":
        res = _split_caps(res, t["name"], rep)
    _rename(res, t, opt, rep)
    _check_sidechains(res, t, opt, rep)
    _elements(res, t)

    # glycan links: atom pairs once (label-independent); the attachment map is
    # built inside _assign_ids once the polymer labels are final
    link_pairs, link_source = _link_pairs(res, t, pos, opt, bonds)
    attach = _assign_ids(res, opt, rep, link_pairs, len(t["name"]))

    rep["gaps"] = _find_gaps(res, t, pos, opt.gap_cutoff)
    if rep["gaps"]:
        rep["info"].append(
            f"{len(rep['gaps'])} chain gap(s) found (allowed; see 'gaps')"
        )
    rep["n_atoms_out"] = len(t["name"])
    rep["n_residues"] = len(res)
    rep["residues_split"] = [
        f"{r.orig['chain']}:{r.orig['resname']}:{r.orig['resid']} -> "
        f"{r.chain}:{r.resname}:{r.resid}{r.icode}"
        for r in res
        if r.split
    ]
    rep["glycans"] = [
        {
            "label": f"{r.chain}:{r.resname}:{r.resid}",
            "n_atoms": r.stop - r.start,
            "orig": r.orig,
        }
        for r in res
        if r.kind == "glycan"
    ]

    # resolve 'auto' locally: the caller's options object is never modified
    placement = opt.glycan_placement
    if placement == "auto":
        placement = "keep" if trajectory else "after-chain"
    rep["options"]["glycan_placement"] = placement

    rep["links"] = _link_records(res, t, pos, link_pairs, link_source)
    conect = _conect_pairs(res, t, bonds, link_pairs, opt.conect)

    if placement != "keep":
        res, t, pos, bonds, conect, link_pairs, attach, perm = _reorder_residues(
            res, t, pos, bonds, conect, link_pairs, attach, placement, rep
        )
        _write_atom_map(f"{out_prefix}_atom_map.csv", perm)
        if trajectory:
            traj_out = f"{out_prefix}{Path(trajectory).suffix}"
            _write_reordered_traj(u, perm, traj_out)
            rep["trajectory_out"] = traj_out

    ter = _ter_positions(res, attach, opt, rep)
    rep["unattached_glycans"] = [
        f"{r.chain}:{r.resname}:{r.resid}"
        for k, r in enumerate(res)
        if r.kind == "glycan" and not attach.get(k)
    ]
    rep["n_conect_bonds"] = len(conect)
    rep["info"].append("ANISOU records (if any) are not written")

    cryst = _cryst1_line(topology, trajectory, u, opt, rep)
    _write_pdb(f"{out_prefix}.pdb", res, t, pos, rep, cryst, conect, ter)
    _write_csv(f"{out_prefix}_residues.csv", res)
    with open(f"{out_prefix}_report.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=str)
    return rep


def verify(std_pdb, trajectory=None, report=None):
    """
    Re-read the output and check that it round-trips cleanly.

    Checks unique (segid, chain, resid, icode) labels, no repeated atom names
    per residue and, with a report, the atom and residue counts.

    :param std_pdb: Standardized PDB file.
    :param trajectory: Trajectory to load with it, or None.
    :param report: Report returned by :func:`standardize`, or None.
    :return: True
    :rtype: bool
    :raises AssertionError: If a check fails.
    """
    u = mda.Universe(std_pdb, trajectory) if trajectory else mda.Universe(std_pdb)
    labels = Counter(
        (r.segid, r.atoms.chainIDs[0], r.resid, r.icode) for r in u.residues
    )
    dups = [k for k, v in labels.items() if v > 1]
    assert not dups, (
        f"duplicate (segid, chain, resid, icode) labels on re-read: {dups[:10]}"
    )
    for r in u.residues:
        keys = list(zip(r.atoms.names, _get(r.atoms, "altLocs", "")))
        assert len(keys) == len(set(keys)), f"repeated atom names in {r}"
    if report:
        assert u.atoms.n_atoms == report["n_atoms_out"], "atom count mismatch"
        if not report.get(
            "hybrid36"
        ):  # hybrid-36 resids may not round-trip in every reader
            assert len(u.residues) == report["n_residues"], (
                "MDAnalysis residue count differs"
            )
    return True


def main(argv=None):
    """
    Command-line entry point: standardize, verify and print the report summary.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :type argv: list[str] or None
    """
    p = standardize_parser()
    a = p.parse_args(argv)
    set_color_mode(a.color)

    if a.cryst1 and len(a.cryst1) not in (3, 6):
        p.error(
            f"--cryst1 takes 3 or 6 values (A B C [ALPHA BETA GAMMA]), "
            f"got {len(a.cryst1)}"
        )

    out = normalise_output_prefix(a.out) if a.out else default_output_prefix(a.topology)
    opt = StandardizeOptions(
        altloc=a.altloc,
        caps=a.caps,
        chain_overflow=a.chain_overflow,
        conect=a.conect,
        cryst1=tuple(a.cryst1) if a.cryst1 else None,
        frame=a.frame,
        gap_cutoff=a.gap_cutoff,
        glycan_chain_mode=a.glycan_chain_mode,
        glycan_chain=a.glycan_chain,
        glycan_guess=a.glycan_guess,
        glycan_placement=a.glycan_placement,
        glycan_record=a.glycan_record,
        glycan_resid_start=a.glycan_resid_start,
        glycan_resnames=tuple(a.glycan_resnames) if a.glycan_resnames else None,
        glycan_segid=a.glycan_segid,
        histidine=None if a.keep_his_names else "HIS",
        incomplete_sidechains="warn" if a.allow_incomplete else "error",
        keep_cryst1=not a.no_keep_cryst1,
        link_cutoff=a.link_cutoff,
        long_resnames=a.long_resnames,
        not_glycan=tuple(a.not_glycan) if a.not_glycan else None,
        protein_chain_default=a.protein_chain_default,
        renumber_blank_other=not a.no_renumber_solvent,
        resname_map=(dict(a.resname_map),),
        ter_before_glycans=a.ter_before_glycans,
        terminal_oxygens=not a.no_terminal_oxygens,
    )
    rep = standardize(a.topology, out, a.traj, opt, a.topology_format)

    verify(f"{out}.pdb", a.traj, rep)

    for w in rep["warnings"]:
        warning(w)
    for i in rep["info"]:
        info(i)
    heading(
        f"{rep['n_residues']} residues, {len(rep['glycans'])} glycans, "
        f"{len(rep['links'])} glycan links, {len(rep['gaps'])} gaps, "
        f"{len(rep.get('chain_copies', []))} chain copies relabelled -> "
        f"{style(out + '.pdb', 'cyan')}"
    )


if __name__ == "__main__":
    main()
