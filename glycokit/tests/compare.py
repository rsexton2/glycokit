#!/usr/bin/env python3
"""
Compare glycokit (GLYCO v3) compatibility modes with the legacy tools:

    glyco --compat     versus  GLYCO v1 (glyco_mod.py)
    glyco --compat-v2  versus  GLYCO-2  (glycomod.py)

Gating checks: residue counts, residue SASA and per-atom excluded glycan
atoms for both pairs. The --compat versus --compat-v2 cross-comparison is
informational only (box and cylinder tests are expected to differ).
"""
import argparse
import ast
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

# =====================================================================
# Configuration
#   precedence: command-line option > GLYCO_COMPARE_<NAME> > DEFAULTS
# =====================================================================

ENV_PREFIX = "GLYCO_COMPARE_"

DEFAULTS = {
    "PDB": "../data/gper_std.pdb",
    "PROBE": 1.4,
    "AREA_CUTOFF": 30.0,
    "DISTANCE_CUTOFF": 30.0,
    "CYLINDER_RADIUS": 1.4,
    "PROTEIN_SELECTION": "protein",
    "GLYCANS": "BGLC",                 # legacy -glycan / -glycans argument
    "GLYCAN_SELECTION": "",            # empty: "resname <GLYCANS>"
    "BOX_WORKERS": 8,
    "SASA_TOLERANCE": 0.006,           # CLI RSA output is rounded to 2 decimals
    "V1_ROOT": "/home/r2/opt/GLYCO",
    "V2_ROOT": "/home/r2/opt/GLYCO-2",
    "V1_ENV": "GLYCO",
    "V2_ENV": "GLYCO2",
    "GLYCO_COMMAND": "glyco",          # split like a shell command line
}


def env_default(name):
    """GLYCO_COMPARE_<name> if set (argparse applies `type`), else the built-in default."""
    value = os.environ.get(ENV_PREFIX + name, "").strip()
    return value if value else DEFAULTS[name]


def positive_float(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


@dataclass(frozen=True)
class CompareConfig:
    pdb: Path
    probe: float
    area_cutoff: float
    distance_cutoff: float
    cylinder_radius: float
    protein_selection: str
    glycan_names: tuple
    glycan_selection: str
    box_workers: int
    sasa_tolerance: float
    v1_root: Path
    v2_root: Path
    v1_env: str
    v2_env: str
    glyco_command: tuple

    @property
    def v1_script(self):
        return self.v1_root / "glyco_mod.py"

    @property
    def v2_script(self):
        return self.v2_root / "glycomod.py"

    @property
    def glycan_arg(self):
        return ",".join(self.glycan_names)

    def as_dict(self):
        out = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            out[key] = value
        return out


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Each setting defaults to the environment variable {ENV_PREFIX}<NAME> "
               "(shown in brackets) when it is set; command-line options override it.",
    )

    def add(flag, name, text, **kwargs):
        parser.add_argument(
            flag, default=env_default(name),
            help=f"{text} [{ENV_PREFIX}{name}; built-in {DEFAULTS[name]!r}]", **kwargs)

    add("--pdb", "PDB", "input PDB file", type=Path)
    add("--probe", "PROBE", "FreeSASA probe radius (A)", type=positive_float)
    add("--area-cutoff", "AREA_CUTOFF", "surface SASA cutoff (A^2)", type=positive_float)
    add("--distance-cutoff", "DISTANCE_CUTOFF", "glycan distance cutoff (A)",
        type=positive_float)
    add("--cylinder-radius", "CYLINDER_RADIUS", "cylinder radius (A), glyco --cr and "
        "GLYCO-2 -cyl_radius", type=positive_float)
    add("--protein-selection", "PROTEIN_SELECTION", "glycokit --ps selection")
    add("--glycans", "GLYCANS", "comma-separated glycan residue names (legacy tools)")
    add("--glycan-selection", "GLYCAN_SELECTION",
        "glycokit --gs selection; empty means 'resname <glycans>'")
    add("--box-workers", "BOX_WORKERS", "glyco --box-workers", type=positive_int)
    add("--sasa-tolerance", "SASA_TOLERANCE", "SASA tolerance (A^2)", type=positive_float)
    add("--v1-root", "V1_ROOT", "GLYCO v1 source directory", type=Path)
    add("--v2-root", "V2_ROOT", "GLYCO-2 source directory", type=Path)
    add("--v1-env", "V1_ENV", "conda environment of GLYCO v1")
    add("--v2-env", "V2_ENV", "conda environment of GLYCO-2")
    add("--glyco-command", "GLYCO_COMMAND",
        "command that runs glycokit, e.g. 'python -m glycokit.glyco'")

    parser.add_argument("--keep-output", action="store_true",
                        help="keep temporary output directories after the run")
    parser.add_argument("--no-normalize", action="store_true",
                        help="run all tools on the original PDB instead of a "
                             "normalized copy (first MODEL, no ANISOU)")
    parser.add_argument("--print-config", action="store_true",
                        help="print the resolved configuration and exit")
    parser.add_argument("--json-summary", type=Path, default=None,
                        help="write the configuration and check results as JSON")
    return parser


def config_from_args(args, parser):
    names = tuple(n for n in re.split(r"[,\s]+", str(args.glycans).strip()) if n)
    if not names:
        parser.error("--glycans: no glycan residue names given")
    too_long = [n for n in names if len(n) > 4]
    if too_long:
        print(f"WARNING: the legacy tools read 4-character residue names; "
              f"{too_long} will not match", file=sys.stderr)

    selection = str(args.glycan_selection).strip() or "resname " + " ".join(names)
    command = tuple(shlex.split(str(args.glyco_command)))
    if not command:
        parser.error("--glyco-command is empty")

    return CompareConfig(
        pdb=Path(args.pdb), probe=args.probe, area_cutoff=args.area_cutoff,
        distance_cutoff=args.distance_cutoff, cylinder_radius=args.cylinder_radius,
        protein_selection=str(args.protein_selection), glycan_names=names,
        glycan_selection=selection, box_workers=args.box_workers,
        sasa_tolerance=args.sasa_tolerance, v1_root=Path(args.v1_root),
        v2_root=Path(args.v2_root), v1_env=str(args.v1_env), v2_env=str(args.v2_env),
        glyco_command=command,
    )

# =====================================================================
# Utilities
# =====================================================================


def canonical_resname(value):
    """
    Canonicalize residue names for comparisons.

    The legacy PDB-based programs use three-character residue names,
    so BGLC and BGL are treated as the same residue name.
    """
    return str(value).strip().upper()[:3]


def check_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.resolve()


def resolve_freesasa(conda_env):
    """Find the absolute FreeSASA executable inside a conda environment."""
    result = subprocess.run(
        [
            "conda", "run", "-n", conda_env, "python", "-c",
            "import shutil; path = shutil.which('freesasa'); print(path or '')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Could not query FreeSASA in conda environment "
            f"{conda_env!r}:\n{result.stderr}"
        )

    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not paths:
        raise RuntimeError(
            f"FreeSASA was not found in conda environment {conda_env!r}"
        )

    path = Path(paths[-1]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resolved FreeSASA path does not exist: {path}")

    return str(path)


def run_command(command, label, cwd=None):
    """Run a command and print stdout/stderr."""
    command = [str(x) for x in command]

    print("\n" + "=" * 80)
    print(label)
    print("$", " ".join(command))
    print("=" * 80)

    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)

    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print("--- stderr ---", file=sys.stderr)
        print(result.stderr.rstrip(), file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")

    return result


def prepare_legacy_output_directory(path):
    """Both legacy programs require their output directory not to exist."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Legacy output directory already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_pdb(src, dst):
    """
    Write a copy of `src` that all four programs read identically:
    ATOM/HETATM/TER (and CRYST1) records of the FIRST model only.

    Removes ANISOU records (which the legacy column parsers would read as
    atoms), additional MODELs (which the legacy scripts merge, while
    FreeSASA uses the first model only) and all other record types.
    Returns counts for reporting.
    """
    stats = {"atoms": 0, "anisou": 0, "models": 0, "other": 0}

    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            record = line[:6]
            if not line.endswith("\n"):
                line += "\n"

            if record.startswith("MODEL"):
                stats["models"] += 1
                continue
            if record.startswith("ENDMDL"):
                break
            if record.startswith(("ATOM", "HETATM")):
                stats["atoms"] += 1
                fout.write(line)
            elif record.startswith(("TER", "CRYST1")):
                fout.write(line)
            elif record.startswith("ANISOU"):
                stats["anisou"] += 1
            else:
                stats["other"] += 1

        fout.write("END\n")

    # Count the MODEL records of the whole file (the loop stops early).
    with open(src) as fin:
        stats["models"] = sum(1 for line in fin if line.startswith("MODEL"))

    return stats


# =====================================================================
# Commands
# =====================================================================


def glycoanalysis_command(pdb, output_dir, compat, cfg):
    if compat == "v1":
        mode_flag = "--compat"
    elif compat == "v2":
        mode_flag = "--compat-v2"
    else:
        raise ValueError(f"Unknown compatibility mode: {compat!r}")

    return [
        *cfg.glyco_command,
        "-a", str(pdb),
        "--ps", cfg.protein_selection,
        "--gs", cfg.glycan_selection,
        "--pr", str(cfg.probe),
        "--ac", str(cfg.area_cutoff),
        "--dc", str(cfg.distance_cutoff),
        "--cr", str(cfg.cylinder_radius),
        "--outdir", str(output_dir),
        "--force",
        "--write-excluded",
        "--box-workers", str(cfg.box_workers),
        mode_flag,
    ]


def legacy_v1_command(pdb, output_dir, freesasa_path, cfg):
    return [
        "conda", "run", "-n", cfg.v1_env, "python", str(cfg.v1_script),
        "-pdb", str(pdb),
        "-cutoff", str(cfg.distance_cutoff),
        "-sur_cutoff", str(cfg.area_cutoff),
        "-probe", str(cfg.probe),
        "-module", "all",
        "-glycan", cfg.glycan_arg,
        "-freesasa", str(freesasa_path),
        "-num_proc_in", "1",
        "-num_parallel", "1",
        "-out_folder", str(output_dir),
    ]


def legacy_v2_command(pdb, output_dir, freesasa_path, cfg):
    return [
        "conda", "run", "-n", cfg.v2_env, "python", str(cfg.v2_script),
        "-pdb", str(pdb),
        "-cutoff", str(cfg.distance_cutoff),
        "-sur_cutoff", str(cfg.area_cutoff),
        "-probe_radius", str(cfg.probe),
        "-cyl_radius", str(cfg.cylinder_radius),
        "-module", "all_atom",
        "-glycans", cfg.glycan_arg,
        "-freesasa", str(freesasa_path),
        "-ncpu", "1",
        "-npar", "1",
        "-out_folder", str(output_dir),
    ]


# =====================================================================
# Output locators
# =====================================================================


def find_file(directory, pattern, label):
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"{label} not found: {Path(directory) / pattern}")
    if len(files) > 1:
        print(f"Multiple {label} files found, using the first:")
        for path in files:
            print(" ", path)
    return files[0]


# =====================================================================
# Readers
# =====================================================================

RESIDUE_COLUMNS = {"Chain", "ResNum", "ResName", "Glycan_density", "SASA"}


def read_residue_csv(path):
    """
    Read Chain,ResNum,ResName,Glycan_density,SASA (glycokit's
    residue_summary.csv and GLYCO-2's final_df_*.csv). An empty or NaN
    SASA field (GLYCO-2 residues missing from its RSA data) becomes None.
    """
    values = {}

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)

        if not RESIDUE_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")

        for row in reader:
            key = (
                str(row["Chain"]).strip(),
                str(row["ResNum"]).strip(),
                canonical_resname(row["ResName"]),
            )
            if key in values:
                raise ValueError(f"Duplicate residue key {key} in {path}")

            sasa_text = str(row["SASA"]).strip()
            values[key] = {
                "count": float(row["Glycan_density"]),
                "sasa": None if sasa_text.lower() in ("", "nan") else float(sasa_text),
            }

    return values


def read_v1_count_file(path):
    """
    Read GLYCO v1 *_all_glycount.txt lines such as:

        ('ALA', 'R', '1')   12

    The file contains counts only (no SASA).
    """
    values = {}

    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                tuple_text, count_text = line.rsplit(maxsplit=1)
                residue = ast.literal_eval(tuple_text)
                count = float(count_text)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"Could not parse {path}:{line_number}: {line!r}"
                ) from exc

            if not isinstance(residue, tuple) or len(residue) != 3:
                raise ValueError(
                    f"Unexpected residue key in {path}:{line_number}: {residue!r}"
                )

            resname, chain, resid = residue
            key = (str(chain).strip(), str(resid).strip(), canonical_resname(resname))

            if key in values:
                raise ValueError(f"Duplicate residue key {key} in {path}")

            values[key] = {"count": count, "sasa": None}

    return values


def read_v1_rsa(path):
    """Read GLYCO v1's *_all_outer.rsa with glyco_mod.py's column layout."""
    values = {}

    with open(path) as handle:
        for line in handle:
            if line[0:3] != "RES":
                continue
            try:
                area = float(line[14:22])
            except ValueError:
                continue

            key = (
                line[8].strip(),
                line[9:14].strip(),
                canonical_resname(line[3:8]),
            )
            values[key] = {"count": None, "sasa": area}

    return values


def read_excluded_csv(path, serial_column):
    """
    Read an excluded-glycan-atom CSV as a set of
    (protein chain, protein resid, glycan atom serial).

    serial_column: 'glycan_atom_number' (GLYCO v1) or
                   'glycan_atom_index'  (GLYCO-2, glycokit).
    Returns (keys, n_rows, n_duplicate_rows).
    """
    keys = set()
    n_rows = 0
    n_duplicates = 0

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_chain", "protein_resid", serial_column}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")

        for row in reader:
            n_rows += 1
            key = (
                str(row["protein_chain"]).strip(),
                str(row["protein_resid"]).strip(),
                str(row[serial_column]).strip(),
            )
            if key in keys:
                n_duplicates += 1
            keys.add(key)

    return keys, n_rows, n_duplicates


# =====================================================================
# Comparisons
# =====================================================================


def compare_counts(name_a, data_a, name_b, data_b, missing_in_b_means_zero=False):
    """
    Compare glycan counts.

    missing_in_b_means_zero=True: compare the UNION of both residue sets,
    treating a residue missing from either side as count 0 (glycokit compat
    modes and GLYCO-2 omit zero-count residues, GLYCO v1 lists them).

    missing_in_b_means_zero=False: compare only residues present in both;
    residues present on one side only cause a FAIL.
    """
    print("\n" + "-" * 80)
    print(f"COUNT COMPARISON: {name_a} versus {name_b}")
    print("-" * 80)

    keys_a = set(data_a)
    keys_b = set(data_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    comparison_keys = sorted(keys_a | keys_b) if missing_in_b_means_zero \
        else sorted(keys_a & keys_b)

    def count_of(data, key):
        return data[key]["count"] if key in data else 0.0

    differences = []
    zero_status_differences = []

    for key in comparison_keys:
        count_a = count_of(data_a, key)
        count_b = count_of(data_b, key)
        difference = count_a - count_b

        if difference != 0:
            differences.append((key, count_a, count_b, difference))
        if (count_a != 0) != (count_b != 0):
            zero_status_differences.append(key)

    differences.sort(key=lambda row: -abs(row[3]))

    nonzero_only_a = [k for k in only_a if data_a[k]["count"] != 0]
    nonzero_only_b = [k for k in only_b if data_b[k]["count"] != 0]

    print(f"{name_a} residues:", len(keys_a))
    print(f"{name_b} residues:", len(keys_b))
    print("Residues compared:", len(comparison_keys))
    print(f"Only in {name_a}:", len(only_a), f"({len(nonzero_only_a)} with nonzero count)")
    print(f"Only in {name_b}:", len(only_b), f"({len(nonzero_only_b)} with nonzero count)")

    if missing_in_b_means_zero:
        print("Residues missing on either side are treated as count zero.")

    print("Different counts:", len(differences))

    if missing_in_b_means_zero:
        for label, keys, data in ((name_a, nonzero_only_a, data_a),
                                  (name_b, nonzero_only_b, data_b)):
            if keys:
                print(f"\nNonzero residues present only in {label}:")
                for key in keys[:20]:
                    print(" ", key, f"{label}={data[key]['count']:g}")
    else:
        for label, keys, data in ((name_a, only_a, data_a), (name_b, only_b, data_b)):
            if keys:
                print(f"\nExamples only in {label}:")
                for key in keys[:20]:
                    print(" ", key, f"count={data[key]['count']:g}")

    if differences:
        print("\nLargest count differences:")
        for key, count_a, count_b, difference in differences[:20]:
            print(key, f"{name_a}={count_a:g}", f"{name_b}={count_b:g}",
                  f"diff={difference:+g}")

    print("Different zero/nonzero count status:", len(zero_status_differences))

    if missing_in_b_means_zero:
        passed = not differences
    else:
        passed = not differences and not only_a and not only_b

    print("RESULT:", "PASS" if passed else "FAIL")

    return {
        "passed": passed,
        "only_a": only_a,
        "only_b": only_b,
        "differences": differences,
        "zero_status_differences": zero_status_differences,
    }


def compare_sasa(name_a, data_a, name_b, data_b, cfg):
    print("\n" + "-" * 80)
    print(f"SASA COMPARISON: {name_a} versus {name_b}")
    print("-" * 80)

    tol, cutoff = cfg.sasa_tolerance, cfg.area_cutoff
    common = sorted(set(data_a) & set(data_b))
    with_sasa = [k for k in common
                 if data_a[k]["sasa"] is not None and data_b[k]["sasa"] is not None]

    differences = []
    for key in with_sasa:
        diff = data_a[key]["sasa"] - data_b[key]["sasa"]
        if abs(diff) > tol:
            differences.append((key, data_a[key]["sasa"], data_b[key]["sasa"], diff))
    differences.sort(key=lambda row: -abs(row[3]))

    surface_status_differences = [
        k for k in with_sasa
        if (data_a[k]["sasa"] >= cutoff) != (data_b[k]["sasa"] >= cutoff)
    ]

    print("Common residues:", len(common))
    print("Common residues with SASA on both sides:", len(with_sasa))
    print(f"Differences greater than {tol} A^2:", len(differences))
    if differences:
        print("\nLargest SASA differences:")
        for key, sasa_a, sasa_b, diff in differences[:20]:
            print(key, f"{name_a}={sasa_a:.6f}", f"{name_b}={sasa_b:.6f}",
                  f"diff={diff:+.6f}")
    print(f"Different SASA status at {cutoff} A^2:", len(surface_status_differences))
    if not with_sasa:
        print("No residue has SASA on both sides; nothing was compared.")

    passed = bool(with_sasa) and not differences
    print("RESULT:", "PASS" if passed else "FAIL")
    return {"passed": passed, "differences": differences,
            "surface_status_differences": surface_status_differences}


def compare_excluded(name_a, excluded_a, name_b, excluded_b):
    """Compare (chain, resid, glycan serial) sets of excluded glycan atoms."""
    keys_a, rows_a, dup_a = excluded_a
    keys_b, rows_b, dup_b = excluded_b

    print("\n" + "-" * 80)
    print(f"EXCLUDED-ATOM COMPARISON: {name_a} versus {name_b}")
    print("-" * 80)

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print(f"{name_a}: {rows_a} rows, {len(keys_a)} unique, {dup_a} duplicate rows")
    print(f"{name_b}: {rows_b} rows, {len(keys_b)} unique, {dup_b} duplicate rows")
    print("Common (chain, resid, glycan serial):", len(keys_a & keys_b))
    print(f"Only in {name_a}:", len(only_a))
    print(f"Only in {name_b}:", len(only_b))

    for label, keys in ((name_a, only_a), (name_b, only_b)):
        if keys:
            print(f"\nExamples only in {label} (chain, resid, glycan serial):")
            for key in keys[:20]:
                print(" ", key)

    passed = not only_a and not only_b and not dup_a and not dup_b
    print("RESULT:", "PASS" if passed else "FAIL")

    return {"passed": passed, "only_a": only_a, "only_b": only_b}


# =====================================================================
# Main
# =====================================================================


def check_details(result, limit=20):
    """JSON-friendly summary of one comparison: pass flag, counts and examples."""
    out = {"passed": bool(result["passed"])}
    for key, value in result.items():
        if key == "passed":
            continue
        items = list(value)
        out[f"n_{key}"] = len(items)
        out[key] = items[:limit]
    return out


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args, parser)

    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(json.dumps(cfg.as_dict(), indent=2))
    if args.print_config:
        return 0

    source_pdb = check_file(cfg.pdb)
    for script in (cfg.v1_script, cfg.v2_script):
        if not script.is_file():
            raise FileNotFoundError(script)

    freesasa_v1 = resolve_freesasa(cfg.v1_env)
    freesasa_v2 = resolve_freesasa(cfg.v2_env)
    print("FreeSASA for GLYCO v1:", freesasa_v1)
    print("FreeSASA for GLYCO-2:", freesasa_v2)

    temporary_root = Path(tempfile.mkdtemp(prefix="glyco_compare_"))
    print("Temporary root:", temporary_root)

    try:
        # ---------------- input ----------------
        if args.no_normalize:
            pdb = source_pdb
        else:
            pdb = temporary_root / "input" / source_pdb.name
            pdb.parent.mkdir(parents=True)
            stats = normalize_pdb(source_pdb, pdb)
            print(f"Normalized input: {pdb} ({stats['atoms']} atom records; "
                  f"removed {stats['anisou']} ANISOU and {stats['other']} other "
                  f"records; MODEL records in source: {stats['models']})")
            if stats["models"] > 1:
                print("WARNING: only the first MODEL is used by all programs.")

        prefix = pdb.name.replace(".pdb", "")
        new_v1_dir = temporary_root / "glycoanalysis_compat"
        new_v2_dir = temporary_root / "glycoanalysis_compat_v2"
        legacy_v1_dir = prepare_legacy_output_directory(temporary_root / "GLYCO_v1")
        legacy_v2_dir = prepare_legacy_output_directory(temporary_root / "GLYCO_v2")

        # ---------------- runs ----------------
        run_command(glycoanalysis_command(pdb, new_v1_dir, "v1", cfg),
                    "glycoanalysis --compat")
        run_command(glycoanalysis_command(pdb, new_v2_dir, "v2", cfg),
                    "glycoanalysis --compat-v2")
        run_command(legacy_v1_command(pdb, legacy_v1_dir, freesasa_v1, cfg),
                    "original GLYCO implementation")
        run_command(legacy_v2_command(pdb, legacy_v2_dir, freesasa_v2, cfg),
                    "GLYCO-2 implementation")

        # ---------------- outputs ----------------
        new_v1_csv = find_file(new_v1_dir, "residue_summary.csv",
                               "glycoanalysis --compat residue summary")
        new_v2_csv = find_file(new_v2_dir, "residue_summary.csv",
                               "glycoanalysis --compat-v2 residue summary")
        new_v1_excl = find_file(new_v1_dir, "excluded_glycan_atoms.csv",
                                "glycoanalysis --compat excluded atoms")
        new_v2_excl = find_file(new_v2_dir, "excluded_glycan_atoms.csv",
                                "glycoanalysis --compat-v2 excluded atoms")
        legacy_v1_counts = find_file(legacy_v1_dir, f"{prefix}_all_glycount.txt",
                                     "GLYCO v1 count file")
        legacy_v1_rsa = find_file(legacy_v1_dir, f"{prefix}_all_outer.rsa",
                                  "GLYCO v1 RSA file")
        legacy_v1_excl = find_file(legacy_v1_dir, f"{prefix}_excluded_glycan_atoms.csv",
                                   "GLYCO v1 excluded atoms")
        legacy_v2_csv = find_file(legacy_v2_dir / "result", "final_df_*.csv",
                                  "GLYCO-2 final CSV")
        legacy_v2_excl = find_file(legacy_v2_dir / "result",
                                   f"{prefix}_excluded_glycan_atoms.csv",
                                   "GLYCO-2 excluded atoms")

        new_v1 = read_residue_csv(new_v1_csv)
        new_v2 = read_residue_csv(new_v2_csv)
        legacy_v1 = read_v1_count_file(legacy_v1_counts)
        legacy_v1_sasa = read_v1_rsa(legacy_v1_rsa)
        legacy_v2 = read_residue_csv(legacy_v2_csv)

        # ---------------- gating checks ----------------
        gating = [
            ("compat versus original GLYCO counts", compare_counts(
                "glycoanalysis --compat", new_v1, "original GLYCO", legacy_v1,
                missing_in_b_means_zero=True)),
            ("compat versus original GLYCO SASA", compare_sasa(
                "glycoanalysis --compat", new_v1, "original GLYCO", legacy_v1_sasa, cfg)),
            ("compat versus original GLYCO excluded atoms", compare_excluded(
                "glycoanalysis --compat",
                read_excluded_csv(new_v1_excl, "glycan_atom_index"),
                "original GLYCO",
                read_excluded_csv(legacy_v1_excl, "glycan_atom_number"))),
            ("compat-v2 versus GLYCO-2 counts", compare_counts(
                "glycoanalysis --compat-v2", new_v2, "GLYCO-2", legacy_v2,
                missing_in_b_means_zero=True)),
            ("compat-v2 versus GLYCO-2 SASA", compare_sasa(
                "glycoanalysis --compat-v2", new_v2, "GLYCO-2", legacy_v2, cfg)),
            ("compat-v2 versus GLYCO-2 excluded atoms", compare_excluded(
                "glycoanalysis --compat-v2",
                read_excluded_csv(new_v2_excl, "glycan_atom_index"),
                "GLYCO-2",
                read_excluded_csv(legacy_v2_excl, "glycan_atom_index"))),
        ]

        # ---------------- informational ----------------
        print("\n" + "=" * 80)
        print("CROSS-COMPARISON: --compat versus --compat-v2 (informational)")
        print("=" * 80)
        compare_counts("glycoanalysis --compat", new_v1,
                       "glycoanalysis --compat-v2", new_v2,
                       missing_in_b_means_zero=True)
        print("(Informational: box and cylinder tests are expected to differ.)")
        compare_sasa("glycoanalysis --compat", new_v1,
                     "glycoanalysis --compat-v2", new_v2, cfg)

        # ---------------- result ----------------
        overall_pass = all(result["passed"] for _, result in gating)
        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)
        for label, result in gating:
            print(f"{label}:", "PASS" if result["passed"] else "FAIL")
        print("OVERALL:", "PASS" if overall_pass else "FAIL")

        if args.json_summary:
            args.json_summary.parent.mkdir(parents=True, exist_ok=True)
            args.json_summary.write_text(json.dumps({
                "schema": 2,
                "config": cfg.as_dict(),
                "checks": {label: check_details(result) for label, result in gating},
                "overall": bool(overall_pass),
                "output_dir": str(temporary_root) if args.keep_output else None,
            }, indent=2, default=str))

        return 0 if overall_pass else 1

    finally:
        if args.keep_output:
            print("\nOutput retained at:")
            print(temporary_root)
        else:
            shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(2)
