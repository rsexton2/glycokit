"""Residue, glycan and atom-name tables used by glyco standardize."""

import re
import string

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
