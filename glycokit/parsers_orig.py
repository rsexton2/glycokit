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
    )

    parser = _HelpOnNoArgsParser(
        prog='glyco',
        description=description,
        epilog=epilog,
        formatter_class=_GlycoHelpFormatter,
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
        metavar='EXT [,EXT...]',
        help='comma-separated file extensions to treat as independent '
             'structures when using --indir',
    )

    # --- atom selections ------------------------------------------------
    selection_group = parser.add_argument_group('atom selections')
    selection_group.add_argument(
        '--ps', '--prot-sel', type=str, default='protein', dest='prot_sel',
        metavar='SELECTION',
        help='MDAnalysis atom selection string for the protein',
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
        help='per-residue SASA threshold (in Angstrom^2) below which a '
             'residue is excluded from surface/blocker calculations',
    )
    geometry_group.add_argument(
        '--cr', '--cylinder-radius', type=float, default=1.4, dest='cyl_radius',
        metavar='ANGSTROM',
        help='exclusion radius used when testing whether a protein atom '
             'blocks a protein-glycan line of sight (in Angstrom)',
    )
    geometry_group.add_argument(
        '--pr', '--probe-radius', type=float, default=1.4, dest='probe_radius', metavar='ANGSTROM',
        help='probe radius passed to FreeSASA (in Angstrom)',
    )

    # --- trajectory slicing -----------------------------------------------
    slicing_group = parser.add_argument_group(
        'trajectory slicing',
        '--frames is mutually exclusive with --start/--stop/--step; not '
        'applicable when using --indir'
    )
    slicing_group.add_argument(
        '--frames', nargs='*', default=None, metavar='N',
        help='explicit list of frame indices to analyze',
    )
    slicing_group.add_argument(
        '--start', default=None, type=int, metavar='N',
        help='first frame index to analyze',
    )
    slicing_group.add_argument(
        '--stop', default=None, type=int, metavar='N',
        help='frame index to stop before (exclusive)',
    )
    slicing_group.add_argument(
        '--step', default=None, type=int, metavar='N',
        help='step between analyzed frames',
    )

    # --- output -------------------------------------------------------------
    output_group = parser.add_argument_group('output')
    output_group.add_argument(
        '-o', '--outdir', type=str, default='.', dest='outdir',
        metavar='DIR',
        help='output directory where results will be saved',
    )
    output_group.add_argument(
        '-y', '--force', action='store_true',
        help='overwrite outdir without prompting if it already exists',
    )
    output_group.add_argument(
        '--write-vmd', action='store_true', dest='write_vmd',
        help='additionally write a VMD Tcl script that loads per-residue '
             'results into the "user" atom attribute',
    )

    # --- misc -----------------------------------------------------------
    misc_group = parser.add_argument_group('misc')
    misc_group.add_argument(
        '-v', '--verbose', action='store_true',
        help='print progress output during the run',
    )

    return parser

# class _HelpOnNoArgsParser(argparse.ArgumentParser):
#     """
#     ArgumentParser that prints help and exits if invoked with no
#     arguments at all, rather than falling through to argparse's normal
#     behavior (which would just apply every default silently).

#     This is centralized here, rather than left to each call site, so
#     that ANY code calling .parse_args() with no explicit args --
#     whether a script's own __main__ block or something importing this
#     parser elsewhere (e.g. GlycoAnalysis.__init__, which also calls
#     parser.parse_args()) -- gets the same behavior automatically.
#     """

#     def parse_args(self, args=None, namespace=None):
#         if args is None:
#             args = sys.argv[1:]
#         if not args:
#             self.print_help(sys.stderr)
#             self.exit(1)
#         return super().parse_args(args, namespace)


# def base_parser():
#     """
#     Create and return the command-line argument parser for glyco.
#     """
#     description = (
#         "glyco: identify glycan-accessible protein surface residues from "
#         "a topology/trajectory (or a directory of independent structures), "
#         "using FreeSASA for surface exposure and a line-of-sight blocking "
#         "test to determine glycan accessibility."
#     )
#     epilog = (
#         "examples:\n"
#         "  glyco -a top.pdb -b traj.xtc --ps protein --gs 'resname NAG BMA'\n"
#         "  glyco -a top.pdb -b run1.xtc run2.xtc --ps protein --gs glycan\n"
#         "  glyco --indir ./structures --ps protein --gs glycan\n"
#     )

#     parser = argparse.ArgumentParser(
#         prog='glyco',
#         description=description,
#         epilog=epilog,
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     # --- input structure(s) -------------------------------------------------
#     input_group = parser.add_argument_group(
#         'input',
#         'either -a/--top (with optional -b/--traj) or --indir must be '
#         'given; they are mutually exclusive'
#     )
#     input_group.add_argument(
#         '-a', '--top', type=str, dest='top', metavar='PATH',
#         help='path to topology file',
#     )
#     input_group.add_argument(
#         '-b', '--traj', type=str, nargs='+', dest='traj', metavar='PATH',
#         help='path(s) to trajectory file(s); if more than one is given, '
#              'they are loaded via the MDAnalysis chain reader and so must '
#              'all match the atom count of -a/--top',
#     )
#     input_group.add_argument(
#         '--indir', type=str, default=None, dest='indir', metavar='DIR',
#         help='directory of independent structure files, each loaded and '
#              'analyzed separately (not combined via the chain reader); '
#              'mutually exclusive with -a/--top and -b/--traj',
#     )
#     input_group.add_argument(
#         '--in-types', type=str, default='pdb,gro,cif,mmcif', dest='in_types',
#         metavar='EXT[,EXT...]',
#         help='comma-separated file extensions to treat as independent '
#              'structures when using --indir',
#     )

#     # --- atom selections ------------------------------------------------
#     selection_group = parser.add_argument_group('atom selections')
#     selection_group.add_argument(
#         '--ps', '--prot-sel', type=str, default='protein', dest='prot_sel',
#         metavar='SELECTION',
#         help='MDAnalysis atom selection string for the protein',
#     )
#     selection_group.add_argument(
#         '--gs', '--gly-sel', type=str, dest='gly_sel', metavar='SELECTION',
#         help='MDAnalysis atom selection string for the glycan(s)',
#     )

#     # --- cutoffs and geometry --------------------------------------------
#     geometry_group = parser.add_argument_group('cutoffs and geometry')
#     geometry_group.add_argument(
#         '--gc', '--glycan-cutoff', type=float, default=30, dest='distance_cutoff',
#         metavar='ANGSTROM',
#         help='distance cutoff for considering a protein/glycan atom pair '
#              '(in Angstrom)',
#     )
#     geometry_group.add_argument(
#         '--sc', '--sasa-cutoff', type=float, default=30, dest='area_cutoff',
#         metavar='ANGSTROM^2',
#         help='per-residue SASA threshold (in Angstrom^2) below which a '
#              'residue is excluded from surface/blocker calculations',
#     )
#     geometry_group.add_argument(
#         '-r', '--radius', type=float, default=1.4, dest='radius',
#         metavar='ANGSTROM',
#         help='exclusion radius used when testing whether a protein atom '
#              'blocks a protein-glycan line of sight (in Angstrom)',
#     )
#     geometry_group.add_argument(
#         '--probe', type=float, default=1.4, dest='probe', metavar='ANGSTROM',
#         help='probe radius passed to FreeSASA (in Angstrom)',
#     )

#     # --- trajectory slicing -----------------------------------------------
#     slicing_group = parser.add_argument_group(
#         'trajectory slicing',
#         '--frames is mutually exclusive with --start/--stop/--step; not '
#         'applicable when using --indir'
#     )
#     slicing_group.add_argument(
#         '--frames', nargs='*', default=None, metavar='N',
#         help='explicit list of frame indices to analyze',
#     )
#     slicing_group.add_argument(
#         '--start', default=None, type=int, metavar='N',
#         help='first frame index to analyze',
#     )
#     slicing_group.add_argument(
#         '--stop', default=None, type=int, metavar='N',
#         help='frame index to stop before (exclusive)',
#     )
#     slicing_group.add_argument(
#         '--step', default=None, type=int, metavar='N',
#         help='step between analyzed frames',
#     )

#     # --- output -------------------------------------------------------------
#     output_group = parser.add_argument_group('output')
#     output_group.add_argument(
#         '--out-folder', '--outdir', type=str, default='.', dest='outdir',
#         metavar='DIR',
#         help='output folder where results will be saved',
#     )
#     output_group.add_argument(
#         '-y', '--force', action='store_true',
#         help='overwrite out-folder without prompting if it already exists',
#     )
#     output_group.add_argument(
#         '--write-vmd', action='store_true', dest='write_vmd',
#         help='additionally write a VMD Tcl script that loads per-residue '
#              'results into the "user" atom attribute',
#     )

#     # --- misc -----------------------------------------------------------
#     misc_group = parser.add_argument_group('misc')
#     misc_group.add_argument(
#         '--verbose', action='store_true',
#         help='print progress output during the run',
#     )

#     return parser


# def base_parser():
#     """
#     Create and return command-line argument parser.
#     """
#     description = 'TestDesc'
#     prog = 'glyco'
#     epilog = 'testEPI'

#     parser = argparse.ArgumentParser(add_help=True, description=description,
#                                     prog=prog, epilog=epilog,
#                                     )
#     parser.add_argument('-a', '--top', type=str, help='path to topology',
#                         dest='top')
#     parser.add_argument('-b', '--traj', type=str, nargs='+',
#                         help='path(s) to trajectory; if more than one is given, '
#                             'they are loaded via the MDAnalysis chain reader and '
#                             'so must all match the atom count of -a/--top',
#                         dest='traj')
#     parser.add_argument('--indir', type=str, default=None,
#                         help='directory of independent structure files, each loaded '
#                             'and analyzed separately (not combined via the chain reader); '
#                             'mutually exclusive with -a/--top, -b/--traj, and --dir',
#                         dest='indir')
#     parser.add_argument('--in-types', type=str, default='pdb,gro,cif,mmcif',
#                         help='comma-separated file extensions to treat as independent '
#                             'structures when using --indir',
#                         dest='in_types')
#     parser.add_argument('--ps', '--prot-sel', type=str, default='protein',
#                         help='',
#                         dest='prot_sel')
#     parser.add_argument('--gs', '--gly-sel', type=str,
#                         help='',
#                         dest='gly_sel')
#     parser.add_argument('--gc', '--glycan-cutoff', type=float,
#                         help='cutoff distance for glycans (in Angstrom)',
#                         dest='distance_cutoff', default=30)
#     parser.add_argument('--sc', '--sasa-cutoff', type=float, help='',
#                         dest='area_cutoff', default=30)
#     parser.add_argument('-r', '--radius', type=float, help='',
#                         dest='radius', default=1.4)
#     parser.add_argument('--probe', type=float, default=1.4,
#                         help='Enter probe radius of FreeSASA in Angstrom')
#     parser.add_argument('--frames', nargs='*')
#     parser.add_argument('--start', default=None, type=int)
#     parser.add_argument('--stop', default=None, type=int)
#     parser.add_argument('--step', default=None, type=int)
#     parser.add_argument('--verbose', action='store_true')
#     parser.add_argument('--outdir', type=str, help='Output folder where '\
#                         'results will be saved.', default='.')
#     parser.add_argument('--write-vmd', action='store_true')
#     parser.add_argument('-y', '--force', action='store_true',
#                         help='overwrite outdir without prompting')

#     return parser
