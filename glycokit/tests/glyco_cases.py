"""
Per-structure settings for the glycokit tests.

Sources, later wins:
1. global settings (pytest options > GLYCO_COMPARE_* > built-in)
2. every [[case]] entry of the --glyco-cases file whose `match` fits, in file order
3. a sidecar file <structure>.glyco.toml
"""
import fnmatch
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import tomllib

# Test mode -> (glyco --mode, glyco --blocking). None: the mode's only test.
MODE_SPECS = {
    "default": ("default", "vdw"),
    "default-cylinder": ("default", "cylinder"),
    "v2": ("v2", None),
    "v1": ("v1", None),
}
MODES = tuple(MODE_SPECS)
SIDECAR_SUFFIX = ".glyco.toml"


class CaseConfigError(ValueError):
    pass


def split_names(value):
    if isinstance(value, str):
        items = re.split(r"[,\s]+", value)
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        raise TypeError(f"expected a string or a list, got {type(value).__name__}")
    names = tuple(dict.fromkeys(n.strip() for n in items if n.strip()))
    if not names:
        raise ValueError("no names given")
    return names


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a non-empty string")
    return value.strip()


def _positive(value):
    if isinstance(value, bool):
        raise TypeError("expected a number")
    value = float(value)
    if value <= 0:
        raise ValueError("must be positive")
    return value


def _modes(value):
    modes = split_names(value)
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"unknown mode(s) {unknown}; choose from {list(MODES)}")
    return modes


KEYS = {
    "gs": _text, "ps": _text, "bs": _text, "sasa_env": _text,
    "dc": _positive, "ac": _positive, "cr": _positive, "pr": _positive,
    "legacy_glycans": split_names,
    "modes": _modes,
    "xfail": _text, "skip": _text,
    "note": _text,
}


def validate(raw, where):
    if not isinstance(raw, dict):
        raise CaseConfigError(f"{where}: expected a table")
    out = {}
    for key, value in raw.items():
        norm = key.replace("-", "_")
        if norm not in KEYS:
            raise CaseConfigError(f"{where}: unknown key {key!r}; allowed: {sorted(KEYS)}")
        try:
            out[norm] = KEYS[norm](value)
        except (TypeError, ValueError) as exc:
            raise CaseConfigError(f"{where}: {key} = {value!r}: {exc}") from exc
    out.pop("note", None)
    return out


def _read(path):
    path = Path(path)
    try:
        text = path.read_text()
        return json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
    except (OSError, ValueError) as exc:     # TOML/JSON decode errors are ValueErrors
        raise CaseConfigError(f"{path}: {exc}") from exc


@dataclass
class Entry:
    patterns: tuple
    values: dict
    label: str
    base: Path
    used: bool = False

    def matches(self, structure):
        structure = Path(structure).resolve()
        for pattern in self.patterns:
            if "/" in pattern or os.sep in pattern:
                full = Path(pattern) if os.path.isabs(pattern) else self.base / pattern
                if fnmatch.fnmatchcase(str(structure), os.path.normpath(str(full))):
                    return True
            elif fnmatch.fnmatchcase(structure.name, pattern):
                return True
        return False


def load_manifest(path):
    path = Path(path).resolve()
    data = _read(path)
    if not isinstance(data, dict) or set(data) - {"case"}:
        raise CaseConfigError(f"{path}: expected only [[case]] entries")
    cases = data.get("case", [])
    if not isinstance(cases, list):
        raise CaseConfigError(f"{path}: 'case' must be a list of tables ([[case]])")
    entries = []
    for n, raw in enumerate(cases, start=1):
        label = f"{path.name} case #{n}"
        if not isinstance(raw, dict):
            raise CaseConfigError(f"{label}: expected a table")
        raw = dict(raw)
        match = raw.pop("match", None)
        if not match:
            raise CaseConfigError(f"{label}: missing 'match'")
        patterns = (match,) if isinstance(match, str) else tuple(str(m) for m in match)
        entries.append(Entry(patterns, validate(raw, label), label, path.parent))
    return entries


def overrides_for(structure, entries):
    """(merged overrides, labels of the sources used) for one structure file."""
    merged, sources = {}, []
    for entry in entries:
        if entry.matches(structure):
            merged.update(entry.values)
            entry.used = True
            sources.append(entry.label)
    sidecar = Path(str(structure) + SIDECAR_SUFFIX)
    if sidecar.is_file():
        merged.update(validate(_read(sidecar), sidecar.name))
        sources.append(sidecar.name)
    return merged, sources


_RESNAME_ONLY = re.compile(r"\s*resname\s+([\w\s]+?)\s*")


def legacy_names_from_selection(selection):
    """'resname NAG BMA' -> ('NAG', 'BMA'); () when not a plain resname list."""
    m = _RESNAME_ONLY.fullmatch(selection or "")
    return tuple(m.group(1).split()) if m else ()