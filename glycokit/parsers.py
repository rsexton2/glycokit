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
    "GLYCANS": "BGLC",                 # legacy -glycan / -glycans argument
    "GLYCAN_SELECTION": "",            # empty: "resname <GLYCANS>"
    "N_WORKERS": 8,
    "SASA_TOLERANCE": 0.006,           # legacy RSA output is rounded to 2 decimals
    "V1_ROOT": "/home/r2/opt/GLYCO",
    "V2_ROOT": "/home/r2/opt/GLYCO-2",
    "V1_ENV": "GLYCO",
    "V2_ENV": "GLYCO2",
    "GLYCO_COMMAND": "glyco",          # split like a shell command line
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


class _GlycoHelpFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
    """Keep description/epilog formatting and append "(default: ...)"."""


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
            "informational only."),
        epilog=(f"Value options default to the environment variable "
                f"{COMPARE_ENV_PREFIX}<NAME> (shown in brackets) when it is set; "
                "command-line options override it."),
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
            text = (f"{text} [{COMPARE_ENV_PREFIX}{env}; "
                    f"built-in {COMPARE_DEFAULTS[env]!r}]")
        group.add_argument(*flags, help=text, **kwargs)

    inputs = parser.add_argument_group("input")
    add(inputs, ("--pdb",), "input PDB file", env="PDB", type=Path)
    add(inputs, ("--glycans",),
        "glycan residue names for the legacy tools (comma/space separated, "
        "at most 4 characters each)", env="GLYCANS", type=name_list)

    params = parser.add_argument_group("analysis parameters (shared by all four programs)")
    add(params, ("--protein-selection", "--ps"), "glyco --ps selection",
        env="PROTEIN_SELECTION")
    add(params, ("--glycan-selection", "--gs"),
        "glyco --gs selection; empty means 'resname <glycans>'",
        env="GLYCAN_SELECTION")
    add(params, ("--distance-cutoff", "--dc"), "glycan distance cutoff (A)",
        env="DISTANCE_CUTOFF", type=positive_float)
    add(params, ("--area-cutoff", "--ac"), "surface SASA cutoff (A^2)",
        env="AREA_CUTOFF", type=positive_float)
    add(params, ("--probe", "--pr"), "FreeSASA probe radius (A)",
        env="PROBE", type=positive_float)
    add(params, ("--cylinder-radius", "--cr"),
        "cylinder radius (A): glyco --cr (--mode v2) and GLYCO-2 -cyl_radius",
        env="CYLINDER_RADIUS", type=positive_float)
    add(params, ("--sasa-tolerance",), "allowed per-residue SASA difference (A^2)",
        env="SASA_TOLERANCE", type=positive_float)

    tools = parser.add_argument_group("programs")
    add(tools, ("--glyco-command",),
        "command that runs glycokit, e.g. 'python -m glycokit.glyco'",
        env="GLYCO_COMMAND", type=command_line)
    add(tools, ("--n-workers", "-j"), "glyco -j/--n-workers (-1 = all processors)",
        env="N_WORKERS", type=n_workers_int)
    add(tools, ("--v1-root",), "GLYCO v1 source directory", env="V1_ROOT",
        type=Path)
    add(tools, ("--v2-root",), "GLYCO-2 source directory", env="V2_ROOT",
        type=Path)
    add(tools, ("--v1-env",), "conda environment of GLYCO v1", env="V1_ENV")
    add(tools, ("--v2-env",), "conda environment of GLYCO-2", env="V2_ENV")

    run = parser.add_argument_group("run and output")
    add(run, ("--no-normalize",),
        "run all tools on the original PDB instead of a normalized copy "
        "(first MODEL, no ANISOU)", action="store_true")
    add(run, ("--keep-output",),
        "keep the temporary output directories after the run",
        action="store_true")
    add(run, ("--json-summary",),
        "write the configuration and check results as JSON",
        type=Path, default=None, metavar="PATH")
    add(run, ("--print-config",),
        "print the resolved configuration and exit", action="store_true")
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
        prog="glyco",
        description=(
            "glyco (GlyCo, GLYcan COverage): glycan coverage of protein "
            "surface residues from a topology/trajectory (or a directory of "
            "independent structures). FreeSASA selects the surface residues; a "
            "line-of-sight test (van der Waals spheres by default, a cylinder "
            "with --blocking cylinder) counts the glycan atoms each residue "
            "can see."),
        epilog=(
            "examples:\n"
            "  glyco -a top.pdb -b traj.xtc --gs 'resname NAG BMA'\n"
            "  glyco -a top.pdb -b run1.xtc run2.xtc --gs glycan -j -1\n"
            "  glyco --indir ./structures --gs glycan\n"
            "  glyco -a top.pdb --gs glycan --vdw-scale 0.8 --write-blockers\n"
            "  glyco -a top.pdb --gs glycan --blocking cylinder --cr 1.4\n"
            "  glyco -a top.pdb --gs glycan --mode v1 --write-excluded\n"
            "  glyco -a top.pdb --gs glycan --mode v2 --cr 1.4 "
            "--write-excluded\n"),
        formatter_class=_GlycoHelpFormatter,
        allow_abbrev=False,
    )

    def add(group, flags, text, **kwargs):
        """Add one option to ``group``; ``dest`` is always given explicitly."""
        group.add_argument(*flags, help=text, **kwargs)

    inputs = parser.add_argument_group(
        "input", "either -a/--top (with optional -b/--traj) or --indir must be "
                 "given; they are mutually exclusive")
    add(inputs, ("-a", "--top"), "path to topology file",
        dest="top", type=str, default=None, metavar="PATH")
    add(inputs, ("-b", "--traj"),
        "path(s) to trajectory file(s); several files are loaded via the "
        "MDAnalysis chain reader and must all match the atom count of -a/--top",
        dest="traj", type=str, nargs="+", default=None, metavar="PATH")
    add(inputs, ("--indir",),
        "directory of independent structure files, each analyzed separately; "
        "mutually exclusive with -a/--top and -b/--traj",
        dest="indir", type=str, default=None, metavar="DIR")
    add(inputs, ("--in-types",),
        "comma-separated file extensions used with --indir. --mode v1/v2 need "
        "PDB record types (.pdb files)",
        dest="in_types", type=str, default="pdb,gro,cif,mmcif", metavar="EXTS")

    selections = parser.add_argument_group("atom selections")
    add(selections, ("--ps", "--prot-sel"),
        "MDAnalysis selection of the protein residues used as line-of-sight "
        "STARTING points (and reported in the output). Every heavy atom of a "
        "surface residue is a start; a residue's count is the number of UNIQUE "
        "glycan atoms visible from any of them. Each residue needs exactly one "
        "CA atom. With --mode v1/v2, restricted to the residue names of the "
        "original scripts. Does not restrict which atoms can block; see --bs",
        dest="prot_sel", type=str, default="protein", metavar="SELECTION")
    add(selections, ("--bs", "--block-sel"),
        "MDAnalysis selection of the atoms that can BLOCK a line of sight "
        "(heavy atoms only, not limited to surface residues). With --mode "
        "v1/v2, restricted to the residue names of the original scripts; with "
        "--mode v1 additionally to ATOM records",
        dest="block_sel", type=str, default="protein", metavar="SELECTION")
    add(selections, ("--gs", "--gly-sel"),
        "MDAnalysis selection of the glycan atoms (required)",
        dest="gly_sel", type=str, default=None, metavar="SELECTION")

    mode = parser.add_argument_group("counting mode")
    add(mode, ("--mode",),
        "default: GlyCo. v1: reproduce the original GLYCO script (integer "
        "cutoff, legacy hydrogen filter, ATOM-record blockers, legacy FreeSASA "
        "set, int-truncated +-1 A box test; slower). v2: reproduce GLYCO-2 "
        "(legacy filters and FreeSASA set, float32 pair and cylinder "
        "arithmetic, the start residue can block)",
        dest="mode", choices=("default", "v1", "v2"), default="default")
    add(mode, ("--blocking",),
        "blocking test. None: vdw in --mode default; v1 always uses the box "
        "test, v2 the cylinder test. vdw: a blocker blocks if the line touches "
        "its sphere of radius vdw_scale * r_vdW + vdw_probe. cylinder: a "
        "blocker blocks if it lies within --cr of the line (inclusive)",
        dest="blocking", choices=("vdw", "cylinder"), default=None)

    geometry = parser.add_argument_group("cutoffs and geometry")
    add(geometry, ("--dc", "--distance-cutoff"),
        "distance cutoff for a start-atom/glycan-atom pair (in Angstrom); a "
        "pair is considered if its distance is strictly below the cutoff. "
        "--mode v1 truncates it to an integer",
        dest="distance_cutoff", type=positive_float, default=30.0,
        metavar="ANGSTROM")
    add(geometry, ("--cr", "--cylinder-radius"),
        "radius of the cylinder blocking test (in Angstrom); only used with "
        "--blocking cylinder or --mode v2. Use GLYCO-2's -cyl_radius value "
        "with --mode v2",
        dest="cylinder_radius", type=positive_float, default=1.4,
        metavar="ANGSTROM")
    add(geometry, ("--vdw-scale",),
        "scale factor for the blocker van der Waals radii (looked up by "
        "element); only used by the vdw blocking test",
        dest="vdw_scale", type=positive_float, default=1.0, metavar="FACTOR")
    add(geometry, ("--vdw-probe",),
        "added to every scaled blocker vdW radius (in Angstrom); only used by "
        "the vdw blocking test. Not the FreeSASA probe (--pr)",
        dest="vdw_probe", type=non_negative_float, default=0.0,
        metavar="ANGSTROM")

    sasa = parser.add_argument_group("FreeSASA (surface residues)")
    add(sasa, ("--ac", "--area-cutoff"),
        "per-residue SASA threshold (in Angstrom^2); residues below it are not "
        "surface residues and are not used as starting points",
        dest="area_cutoff", type=positive_float, default=30.0,
        metavar="ANGSTROM^2")
    add(sasa, ("--pr", "--probe-radius"),
        "probe radius passed to FreeSASA (in Angstrom)",
        dest="sasa_probe", type=positive_float, default=1.4, metavar="ANGSTROM")
    add(sasa, ("--sasa-env",),
        "MDAnalysis selection of extra atoms present as occluders in the "
        "FreeSASA calculation (areas are still reported for the protein only). "
        "With --mode v1/v2 this replaces the legacy atom set, which mirrors "
        "the freesasa executable on the whole PDB (heavy atoms of ATOM "
        "records, first alternate location only)",
        dest="sasa_env", type=str, default=None, metavar="SELECTION")
    add(sasa, ("--sasa-hetatm",),
        "with --mode v1/v2: also include HETATM records in the legacy FreeSASA "
        "atom set (freesasa --hetatm). No effect with --sasa-env",
        dest="sasa_hetatm", action="store_true")

    slicing = parser.add_argument_group(
        "trajectory slicing", "--frames is mutually exclusive with "
                              "--start/--stop/--step; not applicable with "
                              "--indir")
    add(slicing, ("--frames",),
        "explicit list of non-negative frame indices to analyze",
        dest="frame_list", nargs="+", type=int, default=None, metavar="N")
    add(slicing, ("--start",), "first frame index to analyze",
        dest="frame_start", type=int, default=None, metavar="N")
    add(slicing, ("--stop",), "frame index to stop before (exclusive)",
        dest="frame_stop", type=int, default=None, metavar="N")
    add(slicing, ("--step",), "step between analyzed frames",
        dest="frame_step", type=int, default=None, metavar="N")

    output = parser.add_argument_group("output")
    add(output, ("-o", "--outdir"),
        "output directory (created if missing). If omitted: glyco_<topology "
        "name>, or glyco_<directory name> with --indir (one subfolder per "
        "structure)",
        dest="outdir", type=str, default=None, metavar="DIR")
    add(output, ("-y", "--force"),
        "overwrite existing output files without prompting",
        dest="force", action="store_true")
    add(output, ("--bfactor-frame",),
        "index among the ANALYZED frames (negative counts from the end) used "
        "for b_factors.pdb, residue_summary.csv and the per-atom CSVs. None: "
        "the last analyzed frame",
        dest="bfactor_frame", type=int, default=None, metavar="N")
    add(output, ("--write-vmd",),
        "also write a VMD Tcl script that loads the per-residue counts into "
        "the \"user\" atom attribute",
        dest="write_vmd", action="store_true")
    add(output, ("--write-excluded",),
        "also write excluded_glycan_atoms.csv: one row per glycan atom missing "
        "from the visibility union of each surface residue with a nonzero "
        "count. Works in every mode",
        dest="write_excluded", action="store_true")
    add(output, ("--write-blockers",),
        "also write blockers.csv: one row per blocking atom (from --bs) of "
        "every blocked start-atom/glycan-atom pair. Can be very large. Not "
        "available with --mode v1 (the box test does not identify blocking "
        "atoms)",
        dest="write_blockers", action="store_true")

    misc = parser.add_argument_group("misc")
    add(misc, ("-v", "--verbose"),
        "show progress bars and print diagnostic output",
        dest="verbose", action="store_true")
    add(misc, ("-j", "--n-workers"),
        "parallel workers; -1 uses all processors. Threads for the KD-tree "
        "queries (vdw/cylinder), worker processes for the --mode v1 box test "
        "(a new pool per analyzed frame)",
        dest="n_workers", type=n_workers_int, default=1, metavar="N")
    return parser