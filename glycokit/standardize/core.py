"""Public API: output prefixes, the standardize pipeline and verification."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import MDAnalysis as mda

from glycokit.standardize.constants import COMPRESSION_SUFFIXES
from glycokit.standardize.glycans import _classify
from glycokit.standardize.labels import _assign_ids
from glycokit.standardize.layout import _reorder_residues, _ter_positions
from glycokit.standardize.links import _conect_pairs, _link_pairs, _link_records
from glycokit.standardize.model import StandardizeOptions
from glycokit.standardize.reading import _drop_altlocs, _get, _read, _segment
from glycokit.standardize.residues import (
    _check_sidechains,
    _elements,
    _find_gaps,
    _rename,
    _split_caps,
)
from glycokit.standardize.writing import (
    _cryst1_line,
    _write_atom_map,
    _write_csv,
    _write_pdb,
    _write_reordered_traj,
)


def default_output_prefix(topology, tag="_std"):
    """(original docstring)"""
    p = Path(topology)
    if p.suffix.lower() in COMPRESSION_SUFFIXES:
        p = p.with_suffix("")  # model.pdb.gz -> model.pdb
    return str(p.with_name(p.stem + tag))  # model.pdb -> model_std


def normalise_output_prefix(out):
    """(original docstring)"""
    p = Path(out)
    return str(p.with_suffix("")) if p.suffix.lower() == ".pdb" else str(p)


def standardize(
    topology, out_prefix=None, trajectory=None, options=None, topology_format=None
):
    """(original docstring)"""
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
    """(original docstring)"""
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
        if not report.get("hybrid36"):  # hybrid-36 resids may not round-trip
            assert len(u.residues) == report["n_residues"], (
                "MDAnalysis residue count differs"
            )
    return True
