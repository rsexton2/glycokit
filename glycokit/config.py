import os
import re

import MDAnalysis as mda

# ============= Configuration Variables =============

VMD_SCRIPT_NAME = "load_user_attributes.vmd"
DEFAULT_OUTDIR_NAME = "glyco_output"

# Blocking tests allowed per counting mode; the first one is the default.
_BLOCKING_TESTS = {"default": ("vdw", "cylinder"), "v1": ("box",), "v2": ("cylinder",)}

COMPARE_ENV_PREFIX = "GLYCO_COMPARE_"

# Hydrogen atom names: optional leading digits then 'H' (H1, HB2, 1HB, HO2).
# Anchored at the START so heavy atoms like NH1, OH, CH2 are not matched.
_HYDROGEN_NAME = re.compile(r"^\d*H")

# Protein residue names accepted by the original scripts
_V1_RESNAMES = (
    "ALA ARG ASN ASP CYS GLN GLU GLY HSD HIS HIE HID "
    "ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL"
)
_V2_RESNAMES = _V1_RESNAMES + " HSE HSP"

# Safety margin of the vectorized distance prefilter in _box_chunk. The exact
# GLYCO v1 test follows, so this only has to exceed float64 rounding.
_BOX_PREFILTER_EPS = 1e-6

# Padding of the KD-tree prefilters (pair search and the cylinder / vdW
# candidate searches). The exact tests follow, so this only guards rounding.
_KD_PAD = 1e-3

# "file.py:123: UserWarning: message" -> category and message
_WARNING_LINE = re.compile(r"^.*?:\d+: (?P<cat>\w+): (?P<msg>.*)$")

OUTPUT_FILES = {
    "b_factors.pdb": None,
    "b_factors.npy": None,
    "frame_summary.csv": None,
    "residue_summary.csv": None,
    "run_config.json": None,
    "excluded_glycan_atoms.csv": "write_excluded",
    "blockers.csv": "write_blockers",
    "residue_histograms.csv": "write_stats",
    "residue_statistics.csv": "write_stats",
    "b_factors_stats.pdb": "write_stats",
    "glyco_stats.cxc": "write_chimerax",
    VMD_SCRIPT_NAME: "write_vmd",
}

# Built-in defaults. GLYCO_COMPARE_<NAME> overrides them; command-line
# options override both. String defaults go through the option's ``type``.
COMPARE_DEFAULTS = {
    "PDB": "../data/gper_std.pdb",
    "PROBE": 1.4,
    "AREA_CUTOFF": 30.0,
    "DISTANCE_CUTOFF": 30.0,
    "CYLINDER_RADIUS": 1.4,
    "PROTEIN_SELECTION": "protein",
    "GLYCANS": None,  # legacy -glycan / -glycans argument
    "GLYCAN_SELECTION": "",  # empty: "resname <GLYCANS>"
    "N_WORKERS": 8,
    "SASA_TOLERANCE": 0.006,  # legacy RSA output is rounded to 2 decimals
    "V1_ROOT": "/home/r2/opt/GLYCO",
    "V2_ROOT": "/home/r2/opt/GLYCO-2",
    "V1_ENV": "GLYCO",
    "V2_ENV": "GLYCO2",
    "GLYCO_COMMAND": "glyco run",  # split like a shell command line
}


# Errors MDAnalysis raises for an invalid or unsupported selection string:
# SelectionError for parse errors / unknown keywords, NoDataError (a subclass
# of ValueError and AttributeError) for missing topology attributes such as
# record_type on non-PDB topologies.
_SELECTION_ERRORS = (
    mda.exceptions.SelectionError,
    mda.exceptions.NoDataError,
    ValueError,
    TypeError,
    AttributeError,
)

# Errors from building a Universe: missing/unreadable files, unknown formats,
# malformed records, atom-count mismatches between topology and trajectory.
_LOAD_ERRORS = (OSError, ValueError, TypeError, IndexError, KeyError, EOFError)

# Errors one structure of an --indir batch may raise without aborting the
# batch. GlycoAnalysis reports input problems as ValueError, TypeError,
# IndexError, RuntimeError or FileNotFoundError (an OSError).
_JOB_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    IndexError,
    KeyError,
    RuntimeError,
    mda.exceptions.SelectionError,
)


def _compare_default(name):
    """GLYCO_COMPARE_<name> if set, else the built-in default."""
    value = os.environ.get(COMPARE_ENV_PREFIX + name, "").strip()
    return value if value else COMPARE_DEFAULTS[name]
