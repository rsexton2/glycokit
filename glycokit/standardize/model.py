"""Data classes of glyco standardize."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
