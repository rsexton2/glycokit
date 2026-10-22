import argparse
import os
import re
import shlex
import sys
from pathlib import Path

COMPARE_ENV_PREFIX = "GLYCO_COMPARE_"

# Built-in defaults. GLYCO_COMPARE_<NAME> overrides them; command-line
# options override both. String defaults go through the option's ``type``.
COMPARE_DEFAULTS = {
    "PDB": "../data/gper_std.pdb",
    "PROBE": 1.4,
    "AREA_CUTOFF": 30.0,
    "DISTANCE_CUTOFF": 30.0,
    "CYLINDER_RADIUS": 1.4,
    "PROTEIN_SELECTION": "protein",
    "GLYCANS": "BGLC",  # legacy -glycan / -glycans argument
    "GLYCAN_SELECTION": "",  # empty: "resname <GLYCANS>"
    "N_WORKERS": 8,
    "SASA_TOLERANCE": 0.006,  # legacy RSA output is rounded to 2 decimals
    "V1_ROOT": "/home/r2/opt/GLYCO",
    "V2_ROOT": "/home/r2/opt/GLYCO-2",
    "V1_ENV": "GLYCO",
    "V2_ENV": "GLYCO2",
    "GLYCO_COMMAND": "glyco run",  # split like a shell command line
}


class _HelpOnNoArgsParser(argparse.ArgumentParser):
    """
    Print help and exit if invoked with NO arguments via sys.argv.

    Does NOT intervene when an explicit args list is passed in, even an
    empty one: ``parser.parse_args([])`` is used by GlycoAnalysis to read
    the defaults.
    """

    def parse_args(self, args=None, namespace=None):
        if args is None:
            argv = sys.argv[1:]
            if not argv:
                self.print_help(sys.stderr)
                self.exit(1)
            args = argv
        return super().parse_args(args, namespace)


class _GlycoHelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Keep description/epilog formatting and append "(default: ...)"."""


def non_negative_int(text):
    """argparse type: an integer >= 0."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def chain_id(text):
    """argparse type: a single-character PDB chain ID (A-Z, a-z, 0-9)."""
    text = str(text).strip()
    if len(text) != 1 or not text.isalnum() or not text.isascii():
        raise argparse.ArgumentTypeError(
            f"must be one character A-Z, a-z or 0-9, got {text!r}"
        )
    return text


def segid(text):
    """argparse type: a PDB segment ID (1-4 characters, columns 73-76)."""
    text = str(text).strip()
    if not 1 <= len(text) <= 4:
        raise argparse.ArgumentTypeError(f"must be 1-4 characters, got {text!r}")
    return text


def resname_pair(text):
    """argparse type: 'OLD=NEW' -> ('OLD', 'NEW'); NEW has at most 4 characters."""
    old, sep, new = str(text).partition("=")
    old, new = old.strip(), new.strip()
    if not sep or not old or not new:
        raise argparse.ArgumentTypeError(f"expected OLD=NEW, got {text!r}")
    if len(new) > 4:
        raise argparse.ArgumentTypeError(
            f"new residue name {new!r} is longer than 4 characters"
        )
    return old, new


def positive_float(text):
    """argparse type: a float > 0 (rejects nan)."""
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text!r}")
    return value


def non_negative_float(text):
    """argparse type: a float >= 0 (rejects nan)."""
    value = float(text)
    if not value >= 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text!r}")
    return value


def n_workers_int(text):
    """argparse type: -1 (all processors) or an integer >= 1."""
    value = int(text)
    if value != -1 and value < 1:
        raise argparse.ArgumentTypeError(f"must be -1 or >= 1, got {value}")
    return value


def name_list(text):
    """argparse type: 'BGL,BMA AMA' -> ('BGL', 'BMA', 'AMA'), duplicates removed."""
    names = tuple(dict.fromkeys(n for n in re.split(r"[,\s]+", str(text)) if n))
    if not names:
        raise argparse.ArgumentTypeError("no names given")
    return names


def command_line(text):
    """argparse type: split a command like a shell does -> tuple of words."""
    words = tuple(shlex.split(str(text)))
    if not words:
        raise argparse.ArgumentTypeError("empty command")
    return words


def _compare_default(name):
    """GLYCO_COMPARE_<name> if set, else the built-in default."""
    value = os.environ.get(COMPARE_ENV_PREFIX + name, "").strip()
    return value if value else COMPARE_DEFAULTS[name]


def compare_parser():
    """
    Command-line parser of compare.py.

    Value options with an ``env`` name default to ``GLYCO_COMPARE_<NAME>``
    when that variable is set, otherwise to :data:`COMPARE_DEFAULTS`. Long
    option names are used by the glycokit test suite (conftest.py) and must
    not change; the short aliases match the glyco flags.

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

    def add(group, flags, text, env=None, **kwargs):
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
            text = (
                f"{text} [{COMPARE_ENV_PREFIX}{env}; "
                f"built-in {COMPARE_DEFAULTS[env]!r}]"
            )
        group.add_argument(*flags, help=text, **kwargs)

    inputs = parser.add_argument_group("input")
    add(inputs, ("--pdb",), "input PDB file", env="PDB", type=Path)
    add(
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
    add(
        params,
        ("--protein-selection", "--ps"),
        "glyco --ps selection",
        env="PROTEIN_SELECTION",
    )
    add(
        params,
        ("--glycan-selection", "--gs"),
        "glyco --gs selection; empty means 'resname <glycans>'",
        env="GLYCAN_SELECTION",
    )
    add(
        params,
        ("--distance-cutoff", "--dc"),
        "glycan distance cutoff (A)",
        env="DISTANCE_CUTOFF",
        type=positive_float,
    )
    add(
        params,
        ("--area-cutoff", "--ac"),
        "surface SASA cutoff (A^2)",
        env="AREA_CUTOFF",
        type=positive_float,
    )
    add(
        params,
        ("--probe", "--pr"),
        "FreeSASA probe radius (A)",
        env="PROBE",
        type=positive_float,
    )
    add(
        params,
        ("--cylinder-radius", "--cr"),
        "cylinder radius (A): glyco --cr (--mode v2) and GLYCO-2 -cyl_radius",
        env="CYLINDER_RADIUS",
        type=positive_float,
    )
    add(
        params,
        ("--sasa-tolerance",),
        "allowed per-residue SASA difference (A^2)",
        env="SASA_TOLERANCE",
        type=positive_float,
    )

    tools = parser.add_argument_group("programs")
    add(
        tools,
        ("--glyco-command",),
        "command that runs glycokit, e.g. 'python -m glycokit.glyco'",
        env="GLYCO_COMMAND",
        type=command_line,
    )
    add(
        tools,
        ("--n-workers", "-j"),
        "glyco -j/--n-workers (-1 = all processors)",
        env="N_WORKERS",
        type=n_workers_int,
    )
    add(tools, ("--v1-root",), "GLYCO v1 source directory", env="V1_ROOT", type=Path)
    add(tools, ("--v2-root",), "GLYCO-2 source directory", env="V2_ROOT", type=Path)
    add(tools, ("--v1-env",), "conda environment of GLYCO v1", env="V1_ENV")
    add(tools, ("--v2-env",), "conda environment of GLYCO-2", env="V2_ENV")

    run = parser.add_argument_group("run and output")
    add(
        run,
        ("--no-normalize",),
        "run all tools on the original PDB instead of a normalized copy "
        "(first MODEL, no ANISOU)",
        action="store_true",
    )
    add(
        run,
        ("--keep-output",),
        "keep the temporary output directories after the run",
        action="store_true",
    )
    add(
        run,
        ("--json-summary",),
        "write the configuration and check results as JSON",
        type=Path,
        default=None,
        metavar="PATH",
    )
    add(
        run,
        ("--print-config",),
        "print the resolved configuration and exit",
        action="store_true",
    )
    return parser


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
    parser = _HelpOnNoArgsParser(
        prog="glyco run",
        description=(
            "glyco (GlyCo, GLYcan COverage): glycan coverage of protein "
            "surface residues from a topology/trajectory (or a directory of "
            "independent structures). FreeSASA selects the surface residues; a "
            "line-of-sight test (van der Waals spheres by default, a cylinder "
            "with --blocking cylinder) counts the glycan atoms each residue "
            "can see."
        ),
        epilog=(
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
        formatter_class=_GlycoHelpFormatter,
        allow_abbrev=False,
    )

    def add(group, flags, text, **kwargs):
        """Add one option to ``group``; ``dest`` is always given explicitly."""
        group.add_argument(*flags, help=text, **kwargs)

    inputs = parser.add_argument_group(
        "input",
        "either -a/--top (with optional -b/--traj) or --indir must be "
        "given; they are mutually exclusive",
    )
    add(
        inputs,
        ("-a", "--top"),
        "path to topology file",
        dest="top",
        type=str,
        default=None,
        metavar="PATH",
    )
    add(
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
    add(
        inputs,
        ("--indir",),
        "directory of independent structure files, each analyzed separately; "
        "mutually exclusive with -a/--top and -b/--traj",
        dest="indir",
        type=str,
        default=None,
        metavar="DIR",
    )
    add(
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
    add(
        selections,
        ("--ps", "--prot-sel"),
        "MDAnalysis selection of the protein residues used as line-of-sight "
        "STARTING points (and reported in the output). Every heavy atom of a "
        "surface residue is a start; a residue's count is the number of UNIQUE "
        "glycan atoms visible from any of them. Each residue needs exactly one "
        "CA atom. With --mode v1/v2, restricted to the residue names of the "
        "original scripts. Does not restrict which atoms can block; see --bs",
        dest="prot_sel",
        type=str,
        default="protein",
        metavar="SELECTION",
    )
    add(
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
    add(
        selections,
        ("--gs", "--gly-sel"),
        "MDAnalysis selection of the glycan atoms (required)",
        dest="gly_sel",
        type=str,
        default=None,
        metavar="SELECTION",
    )

    mode = parser.add_argument_group("counting mode")
    add(
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
    add(
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
    add(
        geometry,
        ("--dc", "--distance-cutoff"),
        "distance cutoff for a start-atom/glycan-atom pair (in Angstrom); a "
        "pair is considered if its distance is strictly below the cutoff. "
        "--mode v1 truncates it to an integer",
        dest="distance_cutoff",
        type=positive_float,
        default=30.0,
        metavar="ANGSTROM",
    )
    add(
        geometry,
        ("--cr", "--cylinder-radius"),
        "radius of the cylinder blocking test (in Angstrom); only used with "
        "--blocking cylinder or --mode v2. Use GLYCO-2's -cyl_radius value "
        "with --mode v2",
        dest="cylinder_radius",
        type=positive_float,
        default=1.4,
        metavar="ANGSTROM",
    )
    add(
        geometry,
        ("--vdw-scale",),
        "scale factor for the blocker van der Waals radii (looked up by "
        "element); only used by the vdw blocking test",
        dest="vdw_scale",
        type=positive_float,
        default=1.0,
        metavar="FACTOR",
    )
    add(
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
    add(
        sasa,
        ("--ac", "--area-cutoff"),
        "per-residue SASA threshold (in Angstrom^2); residues below it are not "
        "surface residues and are not used as starting points",
        dest="area_cutoff",
        type=positive_float,
        default=30.0,
        metavar="ANGSTROM^2",
    )
    add(
        sasa,
        ("--pr", "--probe-radius"),
        "probe radius passed to FreeSASA (in Angstrom)",
        dest="sasa_probe",
        type=positive_float,
        default=1.4,
        metavar="ANGSTROM",
    )
    add(
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
    add(
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
    add(
        slicing,
        ("--frames",),
        "explicit list of non-negative frame indices to analyze",
        dest="frame_list",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
    )
    add(
        slicing,
        ("--start",),
        "first frame index to analyze",
        dest="frame_start",
        type=int,
        default=None,
        metavar="N",
    )
    add(
        slicing,
        ("--stop",),
        "frame index to stop before (exclusive)",
        dest="frame_stop",
        type=int,
        default=None,
        metavar="N",
    )
    add(
        slicing,
        ("--step",),
        "step between analyzed frames",
        dest="frame_step",
        type=int,
        default=None,
        metavar="N",
    )

    output = parser.add_argument_group("output")
    add(
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
    add(
        output,
        ("-y", "--force"),
        "overwrite existing output files without prompting",
        dest="force",
        action="store_true",
    )
    add(
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
    add(
        output,
        ("--write-vmd",),
        "also write a VMD Tcl script that loads the per-residue counts into "
        'the "user" atom attribute',
        dest="write_vmd",
        action="store_true",
    )
    add(
        output,
        ("--write-excluded",),
        "also write excluded_glycan_atoms.csv: one row per glycan atom missing "
        "from the visibility union of each surface residue with a nonzero "
        "count. Works in every mode",
        dest="write_excluded",
        action="store_true",
    )
    add(
        output,
        ("--write-blockers",),
        "also write blockers.csv: one row per blocking atom (from --bs) of "
        "every blocked start-atom/glycan-atom pair. Can be very large. Not "
        "available with --mode v1 (the box test does not identify blocking "
        "atoms)",
        dest="write_blockers",
        action="store_true",
    )

    misc = parser.add_argument_group("misc")
    add(
        misc,
        ("-v", "--verbose"),
        "show progress bars and print diagnostic output",
        dest="verbose",
        action="store_true",
    )
    add(
        misc,
        ("-j", "--n-workers"),
        "parallel workers; -1 uses all processors. Threads for the KD-tree "
        "queries (vdw/cylinder), worker processes for the --mode v1 box test "
        "(a new pool per analyzed frame)",
        dest="n_workers",
        type=n_workers_int,
        default=1,
        metavar="N",
    )
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
    parser = _HelpOnNoArgsParser(
        prog="glyco standardize",
        description=(
            "Normalise a structure (and optionally its trajectory) for glyco / "
            "GLYCO: unique residue labels, glycans in their own chains with "
            "sequential resids, split merged residues, fixed-column PDB output. "
            "Atom order and count are preserved unless glycans are moved "
            "(see --glycan-placement)."
        ),
        epilog=(
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
        formatter_class=_GlycoHelpFormatter,
        allow_abbrev=False,
    )

    def add(group, flags, text, **kwargs):
        """Add one argument to ``group``; options give ``dest`` explicitly."""
        group.add_argument(*flags, help=text, **kwargs)

    # --- input and output ---------------------------------------------------
    io_group = parser.add_argument_group("input and output")
    add(
        io_group,
        ("topology",),
        "input structure / topology file (any MDAnalysis format; PDB input "
        "keeps record types and CRYST1)",
        metavar="TOPOLOGY",
    )
    add(
        io_group,
        ("-o", "--out"),
        "output prefix; a trailing .pdb is removed. None: <input>_std next "
        "to the input file",
        dest="out",
        type=str,
        default=None,
        metavar="PREFIX",
    )
    add(
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
    add(
        io_group,
        ("--topology-format",),
        "explicit MDAnalysis topology format (e.g. PDB, PSF, GRO) when the "
        "file extension does not identify it",
        dest="topology_format",
        type=str,
        default=None,
        metavar="FORMAT",
    )
    add(
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
    add(
        detect,
        ("--glycan-guess",),
        "off: built-in names and --glycan-resnames only; names: also CHARMM "
        "and PDB CCD carbohydrate names; auto: also detect sugar rings from "
        "the structure (decisions are listed in the report)",
        dest="glycan_guess",
        choices=("off", "names", "auto"),
        default="auto",
    )
    add(
        detect,
        ("--glycan-resnames",),
        "extra residue names always treated as glycans (added to the built-in lists)",
        dest="glycan_resnames",
        nargs="+",
        default=None,
        metavar="NAME",
    )
    add(
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
    add(
        glycans,
        ("--glycan-chain-mode",),
        "attached: glycans linked to a protein chain get its case-swapped "
        "letter (A -> a); single: all glycans in --glycan-chain (old "
        "behaviour)",
        dest="glycan_chain_mode",
        choices=("attached", "single"),
        default="attached",
    )
    add(
        glycans,
        ("--glycan-chain",),
        "chain for unattached glycans, or for all glycans with "
        "--glycan-chain-mode single",
        dest="glycan_chain",
        type=chain_id,
        default="G",
        metavar="ID",
    )
    add(
        glycans,
        ("--glycan-segid",),
        "segid of the --glycan-chain glycans. None: same as the chain",
        dest="glycan_segid",
        type=segid,
        default=None,
        metavar="SEGID",
    )
    add(
        glycans,
        ("--glycan-resid-start",),
        "first resid of every glycan chain (glycans are numbered sequentially)",
        dest="glycan_resid_start",
        type=int,
        default=1,
        metavar="N",
    )
    add(
        glycans,
        ("--glycan-record",),
        "record type written for glycan atoms; keep: as in the input (ATOM if unknown)",
        dest="glycan_record",
        choices=("keep", "ATOM", "HETATM"),
        default="keep",
    )
    add(
        glycans,
        ("--glycan-placement",),
        "keep: input atom order; after-chain: attached glycans follow their "
        "chain (before its TER); end: all glycans after all polymer chains. "
        "auto: after-chain without --traj, keep with --traj",
        dest="glycan_placement",
        choices=("auto", "keep", "after-chain", "end"),
        default="auto",
    )
    add(
        glycans,
        ("--ter-before-glycans",),
        "write TER between a protein chain and its attached glycans (old "
        "behaviour); by default TER follows the trailing glycans",
        dest="ter_before_glycans",
        action="store_true",
    )
    add(
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
    add(
        names,
        ("--protein-chain-default",),
        "first chain letter used when the input has no chain IDs (e.g. GRO); "
        "a new chain starts where the residue numbering restarts",
        dest="protein_chain_default",
        type=chain_id,
        default="A",
        metavar="ID",
    )
    add(
        names,
        ("--chain-overflow",),
        "when repeated chain copies exceed the 62 PDB chain IDs: segid: reuse "
        "letters and distinguish copies by segid (cols 73-76); error: fail",
        dest="chain_overflow",
        choices=("segid", "error"),
        default="segid",
    )
    add(
        names,
        ("--no-renumber-solvent",),
        "keep the input resids of solvent / ions / ligands without a chain ID "
        "(by default they are placed in their own chain and renumbered from 1)",
        dest="no_renumber_solvent",
        action="store_true",
    )
    add(
        names,
        ("--keep-his-names",),
        "keep histidine protonation names (HSD, HSE, HSP, HID, HIE, HIP) "
        "instead of renaming them to HIS",
        dest="keep_his_names",
        action="store_true",
    )
    add(
        names,
        ("--no-terminal-oxygens",),
        "keep CHARMM terminal oxygen names OT1/OT2 instead of renaming them to O/OXT",
        dest="no_terminal_oxygens",
        action="store_true",
    )
    add(
        names,
        ("--caps",),
        "keep: CHARMM terminal cap atoms stay in their residue; split: move "
        "them into separate ACE / NME residues",
        dest="caps",
        choices=("keep", "split"),
        default="keep",
    )
    add(
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
    add(
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
    add(
        checks,
        ("--allow-incomplete",),
        "warn instead of failing when protein residues have missing backbone "
        "or side-chain heavy atoms",
        dest="allow_incomplete",
        action="store_true",
    )
    add(
        checks,
        ("--gap-cutoff",),
        "C(i)-N(i+1) distance (in Angstrom) above which a chain gap is "
        "reported (gaps are allowed)",
        dest="gap_cutoff",
        type=positive_float,
        default=2.0,
        metavar="ANGSTROM",
    )
    add(
        checks,
        ("--altloc",),
        "keep: write all alternate locations; first: keep only the first "
        "altloc per residue (drops atoms; not allowed with --traj)",
        dest="altloc",
        choices=("keep", "first"),
        default="keep",
    )

    # --- PDB records ----------------------------------------------------------
    records = parser.add_argument_group("PDB records")
    add(
        records,
        ("--conect",),
        "CONECT records: none; het: hetero groups, glycan links and "
        "non-peptide polymer links (e.g. disulfides); all: every topology bond",
        dest="conect",
        choices=("none", "het", "all"),
        default="het",
    )
    add(
        records,
        ("--no-keep-cryst1",),
        "do not write the input box as CRYST1",
        dest="no_keep_cryst1",
        action="store_true",
    )
    add(
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
    return parser
