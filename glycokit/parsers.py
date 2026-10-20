import argparse
import sys
import textwrap


class _HelpOnNoArgsParser(argparse.ArgumentParser):
    """
    Prints help and exits if invoked with NO arguments at all via
    sys.argv (args=None, the normal case for a real CLI invocation).

    Does NOT intervene when an explicit args list is passed in, even
    an empty one -- that's a deliberate programmatic call (e.g.
    parser.parse_args([]) to introspect defaults, as GlycoAnalysis
    does), not a user forgetting to type any arguments.
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
    """
    Preserves literal newlines/formatting in description and epilog text
    (RawDescriptionHelpFormatter), while still appending "(default: ...)"
    to each argument's help text (ArgumentDefaultsHelpFormatter).
    Neither formatter alone does both.
    """


def base_parser():
    """
    Create and return the command-line argument parser for glyco.

    Note on destination names: the frame-slicing options use the dests
    frame_start/frame_stop/frame_step/frame_list rather than
    start/stop/step/frames, because AnalysisBase uses the latter names for
    its own internal state and would silently overwrite them.
    """
    description = (
        "glyco: identify glycan-accessible protein surface residues from "
        "a topology/trajectory (or a directory of independent structures), "
        "using FreeSASA for surface exposure and a line-of-sight blocking "
        "test to determine glycan accessibility."
    )
    epilog = textwrap.dedent(
        "examples:\n"
        "  glyco -a top.pdb -b traj.xtc --ps protein --gs 'resname NAG BMA'\n"
        "  glyco -a top.pdb -b run1.xtc run2.xtc --ps protein --gs glycan\n"
        "  glyco --indir ./structures --ps protein --gs glycan\n"
        "  glyco -a top.pdb --gs glycan --compat --write-excluded\n"
        "  glyco -a top.pdb --gs glycan --compat-v2 --cr 1.4 --write-excluded\n"
    )

    parser = _HelpOnNoArgsParser(
        prog='glyco',
        description=description,
        epilog=epilog,
        formatter_class=_GlycoHelpFormatter,
        allow_abbrev=False,
    )

    # --- input structure(s) -------------------------------------------------
    input_group = parser.add_argument_group(
        'input',
        'either -a/--top (with optional -b/--traj) or --indir must be '
        'given; they are mutually exclusive'
    )
    input_group.add_argument(
        '-a', '--top', type=str, dest='top', metavar='PATH',
        help='path to topology file',
    )
    input_group.add_argument(
        '-b', '--traj', type=str, nargs='+', dest='traj', metavar='PATH',
        help='path(s) to trajectory file(s); if more than one is given, '
             'they are loaded via the MDAnalysis chain reader and so must '
             'all match the atom count of -a/--top',
    )
    input_group.add_argument(
        '--indir', type=str, default=None, dest='indir', metavar='DIR',
        help='directory of independent structure files, each loaded and '
             'analyzed separately (not combined via the chain reader); '
             'mutually exclusive with -a/--top and -b/--traj',
    )
    input_group.add_argument(
        '--in-types', type=str, default='pdb,gro,cif,mmcif', dest='in_types',
        metavar='EXTS',
        help='comma-separated file extensions to treat as independent '
             'structures when using --indir',
    )

    # --- atom selections ------------------------------------------------
    selection_group = parser.add_argument_group('atom selections')
    selection_group.add_argument(
        '--ps', '--prot-sel', type=str, default='protein', dest='prot_sel',
        metavar='SELECTION',
        help='MDAnalysis atom selection for the protein residues used as '
             'line-of-sight STARTING points (and reported in the output). '
             'Does not restrict which atoms can block; see --bs',
    )
    selection_group.add_argument(
        '--bs', '--block-sel', type=str, default='protein', dest='block_sel',
        metavar='SELECTION',
        help='MDAnalysis atom selection for the atoms that can BLOCK a '
             'line of sight (heavy atoms only, not limited to surface '
             'residues). Independent of --ps. With --compat, restricted '
             'additionally to ATOM records, as in the original GLYCO',
    )
    selection_group.add_argument(
        '--gs', '--gly-sel', type=str, dest='gly_sel', metavar='SELECTION',
        help='MDAnalysis atom selection string for the glycan(s)',
    )

    # --- cutoffs and geometry --------------------------------------------
    geometry_group = parser.add_argument_group('cutoffs and geometry')
    geometry_group.add_argument(
        '--dc', '--distance-cutoff', type=float, default=30,
        dest='distance_cutoff', metavar='ANGSTROM',
        help='distance cutoff for considering a protein/glycan atom pair '
             '(in Angstrom)',
    )
    geometry_group.add_argument(
        '--ac', '--area-cutoff', type=float, default=30, dest='area_cutoff',
        metavar='ANGSTROM^2',
        help='per-residue SASA threshold (in Angstrom^2); residues with a '
             'SASA below this are not surface residues and are not used as '
             'line-of-sight starting points',
    )
    geometry_group.add_argument(
        '--cr', '--cylinder-radius', type=float, default=1.4, dest='radius',
        metavar='ANGSTROM',
        help='exclusion radius used when testing whether a protein atom '
             'blocks a protein-glycan line of sight (in Angstrom). Ignored '
             'with --compat (box test). Use the same value as GLYCO-2\'s '
             '-cyl_radius (GLYCO-2 default 1.4) with --compat-v2',
    )
    geometry_group.add_argument(
        '--pr', '--probe-radius', type=float, default=1.4, dest='probe',
        metavar='ANGSTROM',
        help='probe radius passed to FreeSASA (in Angstrom)',
    )

    # --- trajectory slicing -----------------------------------------------
    slicing_group = parser.add_argument_group(
        'trajectory slicing',
        '--frames is mutually exclusive with --start/--stop/--step; not '
        'applicable when using --indir'
    )
    slicing_group.add_argument(
        '--frames', nargs='*', type=int, default=None, dest='frame_list',
        metavar='N',
        help='explicit list of frame indices to analyze',
    )
    slicing_group.add_argument(
        '--start', default=None, type=int, dest='frame_start', metavar='N',
        help='first frame index to analyze',
    )
    slicing_group.add_argument(
        '--stop', default=None, type=int, dest='frame_stop', metavar='N',
        help='frame index to stop before (exclusive)',
    )
    slicing_group.add_argument(
        '--step', default=None, type=int, dest='frame_step', metavar='N',
        help='step between analyzed frames',
    )

    # --- output -------------------------------------------------------------
    output_group = parser.add_argument_group('output')
    output_group.add_argument(
        '-o', '--outdir', type=str, default=None, dest='outdir',
        metavar='DIR',
        help='output directory (created if missing). If omitted, defaults '
             'to glyco_<topology name> (or glyco_<directory name> with '
             '--indir); with --indir each structure gets its own subfolder',
    )
    output_group.add_argument(
        '-y', '--force', action='store_true',
        help='overwrite existing output files without prompting',
    )
    output_group.add_argument(
        '--bfactor-frame', type=int, default=None, dest='bfactor_frame',
        metavar='N',
        help='index (among the ANALYZED frames, negative counts from the '
             'end) used for b_factors.pdb, residue_summary.csv and the '
             'per-atom CSVs (--write-excluded, --write-blockers); '
             'defaults to the last analyzed frame',
    )
    output_group.add_argument(
        '--write-vmd', action='store_true', dest='write_vmd',
        help='additionally write a VMD Tcl script that loads per-residue '
             'results into the "user" atom attribute',
    )
    output_group.add_argument(
        '--write-excluded', action='store_true', dest='write_excluded',
        help='additionally write excluded_glycan_atoms.csv: one row per '
             'glycan atom absent from the final visibility union of each '
             'surface residue with a nonzero count (zero-count residues '
             'are omitted; frame chosen by --bfactor-frame). Works in '
             'every mode',
    )
    output_group.add_argument(
        '--write-blockers', action='store_true', dest='write_blockers',
        help='additionally write blockers.csv: one row per blocking protein '
             'atom for every blocked start-atom/glycan-atom pair (frame '
             'chosen by --bfactor-frame). Can be very large. Not supported '
             'with --compat, whose box test does not identify blocking atoms',
    )

    # --- compatibility ---------------------------------------------------
    compat_group = parser.add_argument_group(
        'compatibility with the original GLYCO scripts',
        '--compat and --compat-v2 are mutually exclusive (checked after '
        'parsing)'
    )
    compat_group.add_argument(
        '--compat', '--compat-v1', '--compat-box', action='store_true',
        dest='compat',
        help='reproduce the ORIGINAL GLYCO script (v1): every heavy atom of '
             'a surface residue is a line-of-sight start and a residue\'s '
             'count is the number of UNIQUE glycan atoms visible from any of '
             'them; integer distance cutoff; legacy hydrogen filter (drops '
             'any atom with an H in its name, e.g. NH1, OH); blockers from '
             'ATOM records only; the original blocking test (1 A sampling '
             'along the line, int-truncated sample points and a +-1 A '
             'axis-aligned box instead of a cylinder; --cr is ignored); '
             'FreeSASA on the atoms the freesasa executable uses on the '
             'whole PDB (see --sasa-env, --sasa-hetatm). Slower than the '
             'default. --compat-box is a deprecated alias',
    )
    compat_group.add_argument(
        '--compat-v2', action='store_true', dest='compat_v2',
        help='reproduce GLYCO-2: the same residue-union counting, legacy '
             'hydrogen filter and FreeSASA atom set as --compat, but with '
             'a float distance cutoff and the CYLINDER blocking test using '
             '--cr. Atoms of the start residue can block, the radius test '
             'is inclusive (d <= r) and the pair/cylinder arithmetic follows '
             'GLYCO-2\'s float32 precision, as in GLYCO-2',
    )
    compat_group.add_argument(
        '--sasa-env', type=str, default=None, dest='sasa_env',
        metavar='SELECTION',
        help='MDAnalysis selection of extra atoms present as occluders in '
             'the FreeSASA calculation (areas are still reported for the '
             'protein only). With --compat/--compat-v2, giving this '
             'replaces the default legacy atom set, which mirrors the '
             'freesasa executable on the whole PDB: heavy atoms of ATOM '
             'records, first alternate location only',
    )
    compat_group.add_argument(
        '--sasa-hetatm', action='store_true', dest='sasa_hetatm',
        help='with --compat/--compat-v2: also include HETATM records in the '
             'legacy FreeSASA atom set, like running the freesasa executable '
             'with --hetatm. Has no effect when --sasa-env is given',
    )
    compat_group.add_argument(
        '--box-workers', type=int, default=1, dest='box_workers',
        metavar='N',
        help='number of worker processes for the --compat box test; use 1 '
             'for serial execution. A new process pool is started for every '
             'analyzed frame, so large values mainly pay off for big systems '
             'or few frames',
    )

    # --- misc -----------------------------------------------------------
    misc_group = parser.add_argument_group('misc')
    misc_group.add_argument(
        '-v', '--verbose', action='store_true',
        help='print progress output during the run',
    )

    return parser