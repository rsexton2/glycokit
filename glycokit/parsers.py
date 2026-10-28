"""Command-line parsers of ``glyco run`` and ``glyco standardize``."""

import argparse
from pathlib import Path

from glycokit._cli_common import (
    add_color,
    add_option,
    add_shared_option,
    chain_id,
    command_line,
    glyco_parser,
    name_list,
    non_negative_float,
    non_negative_int,
    positive_float,
    resname_pair,
    segid,
)
from glycokit.config import COMPARE_DEFAULTS, COMPARE_ENV_PREFIX, _compare_default


def _env_help(text, env):
    """Append the environment variable and built-in default to ``text``."""
    return f"{text} [{COMPARE_ENV_PREFIX}{env}; built-in {COMPARE_DEFAULTS[env]!r}]"


def _add(group, flags, text, env=None, **kwargs):
    """
    Add one option to ``group``.

    :param env: :data:`COMPARE_DEFAULTS` key. Sets ``dest`` to its lower-case
        form, the default to ``GLYCO_COMPARE_<env>`` or the built-in value,
        and appends both to the help text. None: a plain option (flags,
        options without an environment default).
    """
    if env is not None:
        kwargs.setdefault("dest", env.lower())
        kwargs["default"] = _compare_default(env)
        text = _env_help(text, env)
    add_option(group, flags, text, **kwargs)


def _add_shared(group, env, text):
    """Add a shared glyco option with its environment default."""
    add_shared_option(
        group,
        env,
        _env_help(text, env),
        dest=env.lower(),
        default=_compare_default(env),
    )


def base_parser():
    """
    Command-line parser of glyco.

    Every option has a default, so ``parse_args([])`` gives the full set of
    GlycoAnalysis parameters. Frame-slicing options use the dests
    frame_start/frame_stop/frame_step/frame_list, because AnalysisBase uses
    start/stop/step/frames for its own state. Defaults are appended to the
    help text by :class:`_GlycoHelpFormatter`.

    :rtype: argparse.ArgumentParser
    """
    parser = glyco_parser(
        "glyco run",
        (
            "glyco (GlyCo, GLYcan COverage): glycan coverage of protein "
            "surface residues from a topology/trajectory (or a directory of "
            "independent structures). FreeSASA selects the surface residues; a "
            "line-of-sight test (van der Waals spheres by default, a cylinder "
            "with --blocking cylinder) counts the glycan atoms each residue "
            "can see."
        ),
        (
            "examples:\n"
            "  glyco run -a top.pdb -b traj.xtc --gs 'resname NAG BMA'\n"
            "  glyco run -a top.pdb -b run1.xtc run2.xtc --gs glycan -j -1\n"
            "  glyco run --indir ./structures --gs glycan\n"
            "  glyco run -a top.pdb --gs glycan --vdw-scale 0.8 --write-blockers\n"
            "  glyco run -a top.pdb --gs glycan --blocking cylinder --cr 1.4\n"
            "  glyco run -a top.pdb --gs glycan --mode v1 --write-excluded\n"
            "  glyco run -a top.pdb --gs glycan --mode v2 --cr 1.4 "
            "--write-excluded\n"
        ),
    )

    inputs = parser.add_argument_group(
        "input",
        "either -a/--top (with optional -b/--traj) or --indir must be "
        "given; they are mutually exclusive",
    )
    add_option(
        inputs,
        ("-a", "--top"),
        "path to topology file",
        dest="top",
        type=str,
        default=None,
        metavar="PATH",
    )
    add_option(
        inputs,
        ("-b", "--traj"),
        "path(s) to trajectory file(s); several files are loaded via the "
        "MDAnalysis chain reader and must all match the atom count of -a/--top",
        dest="traj",
        type=str,
        nargs="+",
        default=None,
        metavar="PATH",
    )
    add_option(
        inputs,
        ("--indir",),
        "directory of independent structure files, each analyzed separately; "
        "mutually exclusive with -a/--top and -b/--traj",
        dest="indir",
        type=str,
        default=None,
        metavar="DIR",
    )
    add_option(
        inputs,
        ("--in-types",),
        "comma-separated file extensions used with --indir. --mode v1/v2 need "
        "PDB record types (.pdb files)",
        dest="in_types",
        type=str,
        default="pdb,gro,cif,mmcif",
        metavar="EXTS",
    )

    selections = parser.add_argument_group("atom selections")
    add_shared_option(
        selections,
        "PROTEIN_SELECTION",
        "MDAnalysis selection of the protein residues used as line-of-sight "
        "STARTING points (and reported in the output). Every heavy atom of a "
        "surface residue is a start; a residue's count is the number of UNIQUE "
        "glycan atoms visible from any of them. Each residue needs exactly one "
        "CA atom. With --mode v1/v2, restricted to the residue names of the "
        "original scripts. Does not restrict which atoms can block; see --bs",
        default="protein",
    )
    add_option(
        selections,
        ("--bs", "--block-sel"),
        "MDAnalysis selection of the atoms that can BLOCK a line of sight "
        "(heavy atoms only, not limited to surface residues). With --mode "
        "v1/v2, restricted to the residue names of the original scripts; with "
        "--mode v1 additionally to ATOM records",
        dest="block_sel",
        type=str,
        default="protein",
        metavar="SELECTION",
    )
    add_shared_option(
        selections,
        "GLYCAN_SELECTION",
        "MDAnalysis selection of the glycan atoms. None: the residue names "
        "detected by the glyco standardize glycan guesser (name lists and "
        "sugar-ring test, as --glycan-guess auto)",
        default=None,
    )

    mode = parser.add_argument_group("counting mode")
    add_option(
        mode,
        ("--mode",),
        "default: GlyCo. v1: reproduce the original GLYCO script (integer "
        "cutoff, legacy hydrogen filter, ATOM-record blockers, legacy FreeSASA "
        "set, int-truncated +-1 A box test; slower). v2: reproduce GLYCO-2 "
        "(legacy filters and FreeSASA set, float32 pair and cylinder "
        "arithmetic, the start residue can block)",
        dest="mode",
        choices=("default", "v1", "v2"),
        default="default",
    )
    add_option(
        mode,
        ("--blocking",),
        "blocking test. None: vdw in --mode default; v1 always uses the box "
        "test, v2 the cylinder test. vdw: a blocker blocks if the line touches "
        "its sphere of radius vdw_scale * r_vdW + vdw_probe. cylinder: a "
        "blocker blocks if it lies within --cr of the line (inclusive)",
        dest="blocking",
        choices=("vdw", "cylinder"),
        default=None,
    )

    geometry = parser.add_argument_group("cutoffs and geometry")
    add_shared_option(
        geometry,
        "DISTANCE_CUTOFF",
        "distance cutoff for a start-atom/glycan-atom pair (in Angstrom); a "
        "pair is considered if its distance is strictly below the cutoff. "
        "--mode v1 truncates it to an integer",
        default=30.0,
    )
    add_shared_option(
        geometry,
        "CYLINDER_RADIUS",
        "radius of the cylinder blocking test (in Angstrom); only used with "
        "--blocking cylinder or --mode v2. Use GLYCO-2's -cyl_radius value "
        "with --mode v2",
        default=1.4,
    )
    add_option(
        geometry,
        ("--vdw-scale",),
        "scale factor for the blocker van der Waals radii (looked up by "
        "element); only used by the vdw blocking test",
        dest="vdw_scale",
        type=positive_float,
        default=1.0,
        metavar="FACTOR",
    )
    add_option(
        geometry,
        ("--vdw-probe",),
        "added to every scaled blocker vdW radius (in Angstrom); only used by "
        "the vdw blocking test. Not the FreeSASA probe (--pr)",
        dest="vdw_probe",
        type=non_negative_float,
        default=0.0,
        metavar="ANGSTROM",
    )

    sasa = parser.add_argument_group("FreeSASA (surface residues)")
    add_shared_option(
        sasa,
        "AREA_CUTOFF",
        "per-residue SASA threshold (in Angstrom^2); residues below it are not "
        "surface residues and are not used as starting points",
        default=30.0,
    )
    add_shared_option(
        sasa,
        "PROBE",
        "probe radius passed to FreeSASA (in Angstrom)",
        default=1.4,
    )
    add_option(
        sasa,
        ("--sasa-env",),
        "MDAnalysis selection of extra atoms present as occluders in the "
        "FreeSASA calculation (areas are still reported for the protein only). "
        "With --mode v1/v2 this replaces the legacy atom set, which mirrors "
        "the freesasa executable on the whole PDB (heavy atoms of ATOM "
        "records, first alternate location only)",
        dest="sasa_env",
        type=str,
        default=None,
        metavar="SELECTION",
    )
    add_option(
        sasa,
        ("--sasa-hetatm",),
        "with --mode v1/v2: also include HETATM records in the legacy FreeSASA "
        "atom set (freesasa --hetatm). No effect with --sasa-env",
        dest="sasa_hetatm",
        action="store_true",
    )

    slicing = parser.add_argument_group(
        "trajectory slicing",
        "--frames is mutually exclusive with "
        "--start/--stop/--step; not applicable with "
        "--indir",
    )
    add_option(
        slicing,
        ("--frames",),
        "explicit list of non-negative frame indices to analyze",
        dest="frame_list",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
    )
    add_option(
        slicing,
        ("--start",),
        "first frame index to analyze",
        dest="frame_start",
        type=int,
        default=None,
        metavar="N",
    )
    add_option(
        slicing,
        ("--stop",),
        "frame index to stop before (exclusive)",
        dest="frame_stop",
        type=int,
        default=None,
        metavar="N",
    )
    add_option(
        slicing,
        ("--step",),
        "step between analyzed frames",
        dest="frame_step",
        type=int,
        default=None,
        metavar="N",
    )

    output = parser.add_argument_group("output")
    add_option(
        output,
        ("-o", "--outdir"),
        "output directory (created if missing). If omitted: glyco_<topology "
        "name>, or glyco_<directory name> with --indir (one subfolder per "
        "structure)",
        dest="outdir",
        type=str,
        default=None,
        metavar="DIR",
    )
    add_option(
        output,
        ("-y", "--force"),
        "overwrite existing output files without prompting",
        dest="force",
        action="store_true",
    )
    add_option(
        output,
        ("--bfactor-frame",),
        "index among the ANALYZED frames (negative counts from the end) used "
        "for b_factors.pdb, residue_summary.csv and the per-atom CSVs. None: "
        "the last analyzed frame",
        dest="bfactor_frame",
        type=int,
        default=None,
        metavar="N",
    )
    add_option(
        output,
        ("--write-vmd",),
        "also write a VMD Tcl script that loads the per-residue counts into "
        'the "user" atom attribute',
        dest="write_vmd",
        action="store_true",
    )
    add_option(
        output,
        ("--write-excluded",),
        "also write excluded_glycan_atoms.csv: one row per glycan atom missing "
        "from the visibility union of each surface residue with a nonzero "
        "count. Works in every mode",
        dest="write_excluded",
        action="store_true",
    )
    add_option(
        output,
        ("--write-blockers",),
        "also write blockers.csv: one row per blocking atom (from --bs) of "
        "every blocked start-atom/glycan-atom pair. Can be very large. Not "
        "available with --mode v1 (the box test does not identify blocking "
        "atoms)",
        dest="write_blockers",
        action="store_true",
    )
    add_option(
        output,
        ("--write-stats",),
        "also write trajectory statistics: residue_histograms.csv, "
        "residue_statistics.csv and b_factors_stats.pdb (B-factor = histogram "
        "maximum of the glycan count, occupancy = width of its 95%% interval)",
        dest="write_stats",
        action="store_true",
    )
    add_option(
        output,
        ("--write-chimerax",),
        "also write glyco_stats.cxc, a ChimeraX script that colors by the "
        "histogram maximum and draws a worm by the interval width (implies "
        "--write-stats)",
        dest="write_chimerax",
        action="store_true",
    )

    misc = parser.add_argument_group("misc")
    add_option(
        misc,
        ("-v", "--verbose"),
        "show progress bars and print diagnostic output",
        dest="verbose",
        action="store_true",
    )
    add_shared_option(
        misc,
        "N_WORKERS",
        "parallel workers; -1 uses all processors. Threads for the KD-tree "
        "queries (vdw/cylinder), worker processes for the --mode v1 box test "
        "(a new pool per analyzed frame)",
        default=1,
    )
    add_color(misc)
    return parser


def standardize_parser():
    """
    Command-line parser of ``glyco standardize``.

    Defaults must match :class:`glycokit.standardize.StandardizeOptions`
    (checked by the test suite); they are written out here because
    standardize.py imports this module. Defaults are appended to the help
    text by :class:`_GlycoHelpFormatter`.

    :rtype: argparse.ArgumentParser
    """
    parser = glyco_parser(
        "glyco standardize",
        (
            "Normalise a structure (and optionally its trajectory) for glyco / "
            "GLYCO: unique residue labels, glycans in their own chains with "
            "sequential resids, split merged residues, fixed-column PDB output. "
            "Atom order and count are preserved unless glycans are moved "
            "(see --glycan-placement)."
        ),
        (
            "outputs: <prefix>.pdb, <prefix>_residues.csv, <prefix>_report.json "
            "(+ <prefix>_atom_map.csv and a reordered trajectory when atoms are "
            "moved)\n\n"
            "examples:\n"
            "  glyco standardize protein.pdb\n"
            "  glyco standardize system.gro --traj md.xtc -o system_std\n"
            "  glyco standardize model.pdb --glycan-resnames 0GA 4YB --not-glycan NAD\n"
            "  glyco standardize model.pdb --glycan-chain-mode single --glycan-chain G\n"
            "  glyco standardize model.pdb --resname-map HSD=HIS CYX=CYS\n"
        ),
    )

    # --- input and output ---------------------------------------------------
    io_group = parser.add_argument_group("input and output")
    add_option(
        io_group,
        ("topology",),
        "input structure / topology file (any MDAnalysis format; PDB input "
        "keeps record types and CRYST1)",
        metavar="TOPOLOGY",
    )
    add_option(
        io_group,
        ("-o", "--out"),
        "output prefix; a trailing .pdb is removed. None: <input>_std next "
        "to the input file",
        dest="out",
        type=str,
        default=None,
        metavar="PREFIX",
    )
    add_option(
        io_group,
        ("--traj",),
        "trajectory file loaded with the topology. Its --frame gives the "
        "output coordinates; if atoms are reordered, a reordered copy "
        "<prefix><suffix> is written",
        dest="traj",
        type=str,
        default=None,
        metavar="PATH",
    )
    add_option(
        io_group,
        ("--topology-format",),
        "explicit MDAnalysis topology format (e.g. PDB, PSF, GRO) when the "
        "file extension does not identify it",
        dest="topology_format",
        type=str,
        default=None,
        metavar="FORMAT",
    )
    add_option(
        io_group,
        ("--frame",),
        "trajectory frame whose coordinates are written to the output PDB",
        dest="frame",
        type=non_negative_int,
        default=0,
        metavar="N",
    )

    # --- glycan detection ---------------------------------------------------
    detect = parser.add_argument_group("glycan detection")
    add_option(
        detect,
        ("--glycan-guess",),
        "off: built-in names and --glycan-resnames only; names: also CHARMM "
        "and PDB CCD carbohydrate names; auto: also detect sugar rings from "
        "the structure (decisions are listed in the report)",
        dest="glycan_guess",
        choices=("off", "names", "auto"),
        default="auto",
    )
    add_option(
        detect,
        ("--glycan-resnames",),
        "extra residue names always treated as glycans (added to the built-in lists)",
        dest="glycan_resnames",
        nargs="+",
        default=None,
        metavar="NAME",
    )
    add_option(
        detect,
        ("--not-glycan",),
        "residue names never treated as glycans; overrides every name list "
        "and the structure test",
        dest="not_glycan",
        nargs="+",
        default=None,
        metavar="NAME",
    )

    # --- glycan labels and placement ----------------------------------------
    glycans = parser.add_argument_group("glycan labels and placement")
    add_option(
        glycans,
        ("--glycan-chain-mode",),
        "attached: glycans linked to a protein chain get its case-swapped "
        "letter (A -> a); single: all glycans in --glycan-chain (old "
        "behaviour)",
        dest="glycan_chain_mode",
        choices=("attached", "single"),
        default="attached",
    )
    add_option(
        glycans,
        ("--glycan-chain",),
        "chain for unattached glycans, or for all glycans with "
        "--glycan-chain-mode single",
        dest="glycan_chain",
        type=chain_id,
        default="G",
        metavar="ID",
    )
    add_option(
        glycans,
        ("--glycan-segid",),
        "segid of the --glycan-chain glycans. None: same as the chain",
        dest="glycan_segid",
        type=segid,
        default=None,
        metavar="SEGID",
    )
    add_option(
        glycans,
        ("--glycan-resid-start",),
        "first resid of every glycan chain (glycans are numbered sequentially)",
        dest="glycan_resid_start",
        type=int,
        default=1,
        metavar="N",
    )
    add_option(
        glycans,
        ("--glycan-record",),
        "record type written for glycan atoms; keep: as in the input (ATOM if unknown)",
        dest="glycan_record",
        choices=("keep", "ATOM", "HETATM"),
        default="keep",
    )
    add_option(
        glycans,
        ("--glycan-placement",),
        "keep: input atom order; after-chain: attached glycans follow their "
        "chain (before its TER); end: all glycans after all polymer chains. "
        "auto: after-chain without --traj, keep with --traj",
        dest="glycan_placement",
        choices=("auto", "keep", "after-chain", "end"),
        default="auto",
    )
    add_option(
        glycans,
        ("--ter-before-glycans",),
        "write TER between a protein chain and its attached glycans (old "
        "behaviour); by default TER follows the trailing glycans",
        dest="ter_before_glycans",
        action="store_true",
    )
    add_option(
        glycans,
        ("--link-cutoff",),
        "maximum glycan C1 - acceptor distance (in Angstrom) for a covalent "
        "link when the input has no bonds",
        dest="link_cutoff",
        type=positive_float,
        default=2.0,
        metavar="ANGSTROM",
    )

    # --- chains, residues and atom names ------------------------------------
    names = parser.add_argument_group("chains, residues and atom names")
    add_option(
        names,
        ("--protein-chain-default",),
        "first chain letter used when the input has no chain IDs (e.g. GRO); "
        "a new chain starts where the residue numbering restarts",
        dest="protein_chain_default",
        type=chain_id,
        default="A",
        metavar="ID",
    )
    add_option(
        names,
        ("--chain-overflow",),
        "when repeated chain copies exceed the 62 PDB chain IDs: segid: reuse "
        "letters and distinguish copies by segid (cols 73-76); error: fail",
        dest="chain_overflow",
        choices=("segid", "error"),
        default="segid",
    )
    add_option(
        names,
        ("--no-renumber-solvent",),
        "keep the input resids of solvent / ions / ligands without a chain ID "
        "(by default they are placed in their own chain and renumbered from 1)",
        dest="no_renumber_solvent",
        action="store_true",
    )
    add_option(
        names,
        ("--keep-his-names",),
        "keep histidine protonation names (HSD, HSE, HSP, HID, HIE, HIP) "
        "instead of renaming them to HIS",
        dest="keep_his_names",
        action="store_true",
    )
    add_option(
        names,
        ("--no-terminal-oxygens",),
        "keep CHARMM terminal oxygen names OT1/OT2 instead of renaming them to O/OXT",
        dest="no_terminal_oxygens",
        action="store_true",
    )
    add_option(
        names,
        ("--caps",),
        "keep: CHARMM terminal cap atoms stay in their residue; split: move "
        "them into separate ACE / NME residues",
        dest="caps",
        choices=("keep", "split"),
        default="keep",
    )
    add_option(
        names,
        ("--resname-map",),
        "residue renames applied after all other renames, as OLD=NEW pairs "
        "(NEW at most 4 characters)",
        dest="resname_map",
        nargs="*",
        type=resname_pair,
        default=[],
        metavar="OLD=NEW",
    )
    add_option(
        names,
        ("--long-resnames",),
        "4-character residue names (nonstandard in PDB columns 18-21): allow "
        "silently, warn, or error",
        dest="long_resnames",
        choices=("allow", "warn", "error"),
        default="warn",
    )

    # --- structure checks ---------------------------------------------------
    checks = parser.add_argument_group("structure checks")
    add_option(
        checks,
        ("--allow-incomplete",),
        "warn instead of failing when protein residues have missing backbone "
        "or side-chain heavy atoms",
        dest="allow_incomplete",
        action="store_true",
    )
    add_option(
        checks,
        ("--gap-cutoff",),
        "C(i)-N(i+1) distance (in Angstrom) above which a chain gap is "
        "reported (gaps are allowed)",
        dest="gap_cutoff",
        type=positive_float,
        default=2.0,
        metavar="ANGSTROM",
    )
    add_option(
        checks,
        ("--altloc",),
        "keep: write all alternate locations; first: keep only the first "
        "altloc per residue (drops atoms; not allowed with --traj)",
        dest="altloc",
        choices=("keep", "first"),
        default="keep",
    )

    # --- PDB records --------------------------------------------------------
    records = parser.add_argument_group("PDB records")
    add_option(
        records,
        ("--conect",),
        "CONECT records: none; het: hetero groups, glycan links and "
        "non-peptide polymer links (e.g. disulfides); all: every topology bond",
        dest="conect",
        choices=("none", "het", "all"),
        default="het",
    )
    add_option(
        records,
        ("--no-keep-cryst1",),
        "do not write the input box as CRYST1",
        dest="no_keep_cryst1",
        action="store_true",
    )
    add_option(
        records,
        ("--cryst1",),
        "explicit box written as CRYST1 with space group P 1: A B C "
        "[ALPHA BETA GAMMA]; angles default to 90",
        dest="cryst1",
        nargs="+",
        type=positive_float,
        default=None,
        metavar="X",
    )

    # --- misc ---------------------------------------------------------------
    misc = parser.add_argument_group("misc")
    add_color(misc)
    return parser


def compare_parser():
    """
    Command-line parser of compare.py.

    Value options with an ``env`` name default to ``GLYCO_COMPARE_<NAME>``
    when that variable is set, otherwise to :data:`COMPARE_DEFAULTS`. The
    analysis options share their flags with ``glyco run``
    (:data:`glycokit._cli_common.SHARED_OPTIONS`); their dests stay the
    lower-case environment names.

    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="compare.py",
        description=(
            "Compare glycokit's reproduction modes with the legacy tools:\n"
            "  glyco --mode v1  versus  GLYCO v1 (glyco_mod.py)\n"
            "  glyco --mode v2  versus  GLYCO-2  (glycomod.py)\n\n"
            "Gating checks: residue counts, residue SASA and excluded glycan "
            "atoms for both pairs.\nThe v1 versus v2 cross-comparison is "
            "informational only."
        ),
        epilog=(
            f"Value options default to the environment variable "
            f"{COMPARE_ENV_PREFIX}<NAME> (shown in brackets) when it is set; "
            "command-line options override it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )

    inputs = parser.add_argument_group("input")
    _add(inputs, ("--pdb",), "input PDB file", env="PDB", type=Path)
    _add(
        inputs,
        ("--glycans",),
        "glycan residue names for the legacy tools (comma/space separated, "
        "at most 4 characters each)",
        env="GLYCANS",
        type=name_list,
    )

    params = parser.add_argument_group(
        "analysis parameters (shared by all four programs)"
    )
    _add_shared(params, "PROTEIN_SELECTION", "glyco --ps selection")
    _add_shared(
        params,
        "GLYCAN_SELECTION",
        "MDAnalysis selection of the glycan atoms. None: the residue names "
        "detected by the glyco standardize glycan guesser (name lists and "
        "sugar-ring test, as --glycan-guess auto)",
    )
    _add_shared(params, "DISTANCE_CUTOFF", "glycan distance cutoff (A)")
    _add_shared(params, "AREA_CUTOFF", "surface SASA cutoff (A^2)")
    _add_shared(params, "PROBE", "FreeSASA probe radius (A)")
    _add_shared(
        params,
        "CYLINDER_RADIUS",
        "cylinder radius (A): glyco --cr (--mode v2) and GLYCO-2 -cyl_radius",
    )
    _add(
        params,
        ("--sasa-tolerance",),
        "allowed per-residue SASA difference (A^2)",
        env="SASA_TOLERANCE",
        type=positive_float,
    )

    tools = parser.add_argument_group("programs")
    _add(
        tools,
        ("--glyco-command",),
        "command that runs glycokit, e.g. 'python -m glycokit.glyco'",
        env="GLYCO_COMMAND",
        type=command_line,
    )
    _add_shared(tools, "N_WORKERS", "glyco -j/--n-workers (-1 = all processors)")
    _add(tools, ("--v1-root",), "GLYCO v1 source directory", env="V1_ROOT", type=Path)
    _add(tools, ("--v2-root",), "GLYCO-2 source directory", env="V2_ROOT", type=Path)
    _add(tools, ("--v1-env",), "conda environment of GLYCO v1", env="V1_ENV")
    _add(tools, ("--v2-env",), "conda environment of GLYCO-2", env="V2_ENV")

    run = parser.add_argument_group("run and output")
    _add(
        run,
        ("--no-normalize",),
        "run all tools on the original PDB instead of a normalized copy "
        "(first MODEL, no ANISOU)",
        action="store_true",
    )
    _add(
        run,
        ("--keep-output",),
        "keep the temporary output directories after the run",
        action="store_true",
    )
    _add(
        run,
        ("--json-summary",),
        "write the configuration and check results as JSON",
        type=Path,
        default=None,
        metavar="PATH",
    )
    _add(
        run,
        ("--print-config",),
        "print the resolved configuration and exit",
        action="store_true",
    )
    return parser
