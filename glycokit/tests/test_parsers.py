"""
Tests for the argparse type functions in glycokit.parsers and their use in
base_parser() and compare_parser().
"""

import argparse
import math

import pytest

from glycokit.parsers import (
    base_parser,
    compare_parser,
    n_workers_int,
    non_negative_float,
    positive_float,
    standardize_parser,
)
from glycokit.standardize import StandardizeOptions

NAN_SPELLINGS = ("nan", "NaN", "-nan")


# --------------------------------------------------------------------------
# type functions called directly
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1", 1.0),
        ("0.5", 0.5),
        ("30", 30.0),
        ("1e-6", 1e-6),
        (" 2.5 ", 2.5),
    ],
)
def test_positive_float_accepts(text, expected):
    value = positive_float(text)
    assert isinstance(value, float)
    assert value == expected


@pytest.mark.parametrize("text", ["0", "0.0", "-0.0", "-1", "-1e-9", *NAN_SPELLINGS])
def test_positive_float_rejects_non_positive(text):
    with pytest.raises(argparse.ArgumentTypeError, match="must be positive"):
        positive_float(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0", 0.0),
        ("-0.0", 0.0),
        ("0.0", 0.0),
        ("1.4", 1.4),
        ("1e-9", 1e-9),
    ],
)
def test_non_negative_float_accepts(text, expected):
    value = non_negative_float(text)
    assert isinstance(value, float)
    assert value == expected


@pytest.mark.parametrize("text", ["-1", "-1e-9", *NAN_SPELLINGS])
def test_non_negative_float_rejects(text):
    with pytest.raises(argparse.ArgumentTypeError, match="must be >= 0"):
        non_negative_float(text)


@pytest.mark.parametrize("func", [positive_float, non_negative_float])
@pytest.mark.parametrize("text", ["", "abc", "1,5", "1.2.3"])
def test_float_types_reject_non_numbers(func, text):
    # ValueError (not ArgumentTypeError): argparse reports it as
    # "invalid <type name> value"
    with pytest.raises(ValueError):
        func(text)


@pytest.mark.parametrize("func", [positive_float, non_negative_float])
def test_float_types_accept_infinity(func):
    """inf passes both types (inf > 0); documented here so a change is deliberate."""
    assert math.isinf(func("inf"))


@pytest.mark.parametrize("text, expected", [("-1", -1), ("1", 1), ("8", 8), (" 4 ", 4)])
def test_n_workers_int_accepts(text, expected):
    value = n_workers_int(text)
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value == expected


@pytest.mark.parametrize("text", ["0", "-2", "-100"])
def test_n_workers_int_rejects_out_of_range(text):
    with pytest.raises(argparse.ArgumentTypeError, match="must be -1 or >= 1"):
        n_workers_int(text)


@pytest.mark.parametrize("text", ["", "2.5", "1e3", "two", "nan"])
def test_n_workers_int_rejects_non_integers(text):
    with pytest.raises(ValueError):
        n_workers_int(text)


# --------------------------------------------------------------------------
# base_parser: values are checked at parse time
# --------------------------------------------------------------------------


def _parse(argv):
    return base_parser().parse_args(argv)


def _parse_error(argv, capsys):
    """Parse ``argv``, expect an argparse error, return its stderr text."""
    with pytest.raises(SystemExit) as exc:
        _parse(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


POSITIVE_OPTIONS = [
    ("--dc", "distance_cutoff"),
    ("--distance-cutoff", "distance_cutoff"),
    ("--cr", "cylinder_radius"),
    ("--vdw-scale", "vdw_scale"),
    ("--ac", "area_cutoff"),
    ("--pr", "sasa_probe"),
]


@pytest.mark.parametrize("flag, dest", POSITIVE_OPTIONS)
def test_base_parser_positive_options_accept(flag, dest):
    assert getattr(_parse([flag, "2.5"]), dest) == 2.5


@pytest.mark.parametrize("flag, dest", POSITIVE_OPTIONS)
@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_base_parser_positive_options_reject(flag, dest, value, capsys):
    err = _parse_error([flag, value], capsys)
    assert "must be positive" in err
    assert flag in err or dest.replace("_", "-") in err


def test_base_parser_vdw_probe(capsys):
    assert _parse(["--vdw-probe", "0"]).vdw_probe == 0.0
    assert _parse(["--vdw-probe", "0.3"]).vdw_probe == 0.3
    assert "must be >= 0" in _parse_error(["--vdw-probe", "-0.1"], capsys)


@pytest.mark.parametrize("flag", ["-j", "--n-workers"])
def test_base_parser_n_workers_accepts(flag):
    assert _parse([flag, "-1"]).n_workers == -1
    assert _parse([flag, "4"]).n_workers == 4


@pytest.mark.parametrize(
    "value, message",
    [
        ("0", "must be -1 or >= 1"),
        ("-2", "must be -1 or >= 1"),
        ("2.5", "invalid n_workers_int value"),
    ],
)
def test_base_parser_n_workers_rejects(value, message, capsys):
    assert message in _parse_error(["-j", value], capsys)


def test_base_parser_defaults_unchanged():
    """Numeric defaults are not passed through ``type`` and keep their values."""
    d = vars(_parse([]))
    assert d["distance_cutoff"] == 30.0
    assert d["cylinder_radius"] == 1.4
    assert d["vdw_scale"] == 1.0
    assert d["vdw_probe"] == 0.0
    assert d["area_cutoff"] == 30.0
    assert d["sasa_probe"] == 1.4
    assert d["n_workers"] == 1
    assert isinstance(d["n_workers"], int)


def test_base_parser_option_types():
    """The checked options really use the custom types."""
    expected = {
        "distance_cutoff": positive_float,
        "cylinder_radius": positive_float,
        "vdw_scale": positive_float,
        "area_cutoff": positive_float,
        "sasa_probe": positive_float,
        "vdw_probe": non_negative_float,
        "n_workers": n_workers_int,
    }
    # pylint: disable=protected-access
    types = {a.dest: a.type for a in base_parser()._actions}
    for dest, func in expected.items():
        assert types[dest] is func, f"{dest}: type is {types[dest]!r}"


# --------------------------------------------------------------------------
# compare_parser: environment defaults go through the same types
# --------------------------------------------------------------------------


def test_compare_parser_cli_values(capsys):
    args = compare_parser().parse_args(["--dc", "26.5", "-j", "-1"])
    assert args.distance_cutoff == 26.5
    assert args.n_workers == -1
    with pytest.raises(SystemExit):
        compare_parser().parse_args(["--dc", "0"])
    assert "must be positive" in capsys.readouterr().err


@pytest.mark.parametrize(
    "env, value, dest, expected",
    [
        ("GLYCO_COMPARE_DISTANCE_CUTOFF", "26.5", "distance_cutoff", 26.5),
        ("GLYCO_COMPARE_N_WORKERS", "-1", "n_workers", -1),
    ],
)
def test_compare_parser_env_defaults_converted(monkeypatch, env, value, dest, expected):
    monkeypatch.setenv(env, value)
    assert getattr(compare_parser().parse_args([]), dest) == expected


@pytest.mark.parametrize(
    "env, value, message",
    [
        ("GLYCO_COMPARE_DISTANCE_CUTOFF", "0", "must be positive"),
        ("GLYCO_COMPARE_N_WORKERS", "0", "must be -1 or >= 1"),
    ],
)
def test_compare_parser_env_defaults_rejected(monkeypatch, capsys, env, value, message):
    """Invalid environment defaults fail like invalid options."""
    monkeypatch.setenv(env, value)
    with pytest.raises(SystemExit):
        compare_parser().parse_args([])
    assert message in capsys.readouterr().err


# --------------------------------------------------------------------------
# standardize_parser
# --------------------------------------------------------------------------


def test_standardize_parser_defaults_match_options():
    a = standardize_parser().parse_args(["in.pdb"])
    o = StandardizeOptions()
    assert a.glycan_guess == o.glycan_guess
    assert a.glycan_chain == o.glycan_chain
    assert a.glycan_chain_mode == o.glycan_chain_mode
    assert a.glycan_segid == o.glycan_segid
    assert a.glycan_resid_start == o.glycan_resid_start
    assert a.glycan_record == o.glycan_record
    assert a.glycan_placement == o.glycan_placement
    assert a.ter_before_glycans == o.ter_before_glycans
    assert a.link_cutoff == o.link_cutoff
    assert a.gap_cutoff == o.gap_cutoff
    assert a.protein_chain_default == o.protein_chain_default
    assert a.chain_overflow == o.chain_overflow
    assert a.caps == o.caps
    assert a.long_resnames == o.long_resnames
    assert a.altloc == o.altloc
    assert a.frame == o.frame
    assert a.conect == o.conect
    assert (not a.no_keep_cryst1) == o.keep_cryst1
    assert (not a.no_renumber_solvent) == o.renumber_blank_other
    assert (not a.no_terminal_oxygens) == o.terminal_oxygens
    assert (None if a.keep_his_names else "HIS") == o.histidine
    assert ("warn" if a.allow_incomplete else "error") == o.incomplete_sidechains
