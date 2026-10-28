"""Command-line entry point of ``glyco standardize``."""

from glycokit.console import heading, info, set_color_mode, style, warning
from glycokit.parsers import standardize_parser
from glycokit.standardize.core import (
    default_output_prefix,
    normalise_output_prefix,
    standardize,
    verify,
)
from glycokit.standardize.model import StandardizeOptions


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
        resname_map=dict(a.resname_map),  # was (dict(...),): see below
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
