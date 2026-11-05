import MDAnalysis as mda
import numpy as np
import pytest

from glycokit.residue_output import ResidueOutputMixin
from glycokit.structure_io import pdb_chain_map, pdb_segid_map, write_pdb
from glycokit.tests.conftest import LEGACY_PDB
from glycokit.tests.test_misc import _run

# pylint: disable=protected-access


def _fresh():
    return mda.Universe(str(LEGACY_PDB))


def _columns(path):
    rows = [
        l.rstrip("\n").ljust(80) for l in path.read_text().splitlines()
        if l.startswith(("ATOM  ", "HETATM"))
    ]
    return [r[21].strip() for r in rows], [r[72:76].strip() for r in rows]


def test_chain_and_segids_round_trip(tmp_path):
    u = _fresh()
    u.segments.segids = [f"S{i:03d}" for i in range(len(u.segments))]
    back = mda.Universe(str(write_pdb(u.atoms, tmp_path / "x.pdb")))
    assert list(back.atoms.chainIDs) == list(u.atoms.chainIDs)
    assert list(back.atoms.segids) == list(u.atoms.segids)


def test_blank_chain_is_not_x(tmp_path):
    u = _fresh()
    u.atoms.chainIDs = np.full(u.atoms.n_atoms, "", dtype=object)
    chains, _ = _columns(write_pdb(u.atoms, tmp_path / "x.pdb"))
    assert set(chains) == {""}


def test_default_segid_written_blank(tmp_path):
    u = _fresh()
    u.segments.segids = ["SYSTEM"] * len(u.segments)
    _, segids = _columns(write_pdb(u.atoms, tmp_path / "x.pdb"))
    assert set(segids) == {""}


def test_long_segid_warns_and_is_cut(tmp_path):
    u = _fresh()
    u.segments.segids = ["GLYCAN"] * len(u.segments)
    with pytest.warns(UserWarning, match="segment IDs"):
        out = write_pdb(u.atoms, tmp_path / "x.pdb")
    assert set(_columns(out)[1]) == {"GLYC"}


@pytest.mark.parametrize(
    "cls", ResidueOutputMixin._registry, ids=lambda c: c.__name__)
def test_host_outputs_keep_ids(cls, tmp_path):
    u = _fresh()
    u.segments.segids = [f"P{i}" for i in range(len(u.segments))]
    obj = _run(cls, u)
    for out in (obj.write_bfactors(output=tmp_path / "b.pdb"),
                obj.write_statistics_pdb(output=tmp_path / "s.pdb")):
        chains, segids = _columns(out)
        assert chains == [str(c).strip() for c in u.atoms.chainIDs]
        assert segids == [str(s).strip() for s in u.atoms.segids]


@pytest.mark.parametrize(
    "chains, expected", [
        (["PROA", "PROB", "GLYC"], {
            "PROA": "A",
            "PROB": "B",
            "GLYC": "C"
        }), (["A", "AB"], {
            "A": "A",
            "AB": "B"
        }),  # single chains keep theirs
        (["CHA", "CHB", ""], {
            "CHA": "A",
            "CHB": "B",
            "": ""
        }),
    ])
def test_chain_map_distinct_position(chains, expected):
    mapping, msg = pdb_chain_map(chains)
    assert mapping == expected
    assert msg is None or "become equal" not in msg


def test_chain_map_matching():
    mapping, msg = pdb_chain_map(["A1", "A2", "B1"])  # no common position
    assert len(set(mapping.values())) == 3
    assert all(mapping[c] in c for c in mapping)
    assert "distinct character" in msg


def test_chain_map_fallback_warns_about_merge():
    mapping, msg = pdb_chain_map(["AB", "BA", "AAB"])  # 3 chains, 2 letters
    assert mapping == {"AB": "A", "BA": "B", "AAB": "A"}
    assert "become equal" in msg


def test_single_char_chains_unchanged():
    assert pdb_chain_map(["A", "B", ""]) == ({"A": "A", "B": "B", "": ""}, None)


def test_write_pdb_uses_distinguishing_character(tmp_path):
    u = _fresh()
    n = len(u.residues)
    per_res = np.where(np.arange(n) < n // 2, "PROA", "PROB").astype(object)
    u.atoms.chainIDs = per_res[u.atoms.resindices]
    with pytest.warns(UserWarning, match="chain IDs longer"):
        out = write_pdb(u.atoms, tmp_path / "x.pdb")
    chains, _ = _columns(out)
    assert set(chains) == {"A", "B"}


@pytest.mark.parametrize(
    "segids, expected", [
        (["GLYCAN1", "GLYCAN2"], {
            "GLYCAN1": "GLY1",
            "GLYCAN2": "GLY2"
        }),
        (
            ["GLYCAN1", "GLYCAN10", "GLYCAN11"], {
                "GLYCAN1": "GLN1",
                "GLYCAN10": "GL10",
                "GLYCAN11": "GL11"
            }), (["SEGAXX", "SEGBXX"], {
                "SEGAXX": "SEGA",
                "SEGBXX": "SEGB"
            }),
        (
            ["GLY1", "GLYCAN1", "GLYCAN2"],  # GLY1 is reserved
            {
                "GLY1": "GLY1",
                "GLYCAN1": "GLN1",
                "GLYCAN2": "GLN2"
            }),
    ])
def test_segid_map_prefix_and_tail(segids, expected):
    mapping, msg = pdb_segid_map(segids)
    assert mapping == expected
    assert "become equal" not in msg


def test_segid_map_numbered_when_only_middle_differs():
    mapping, msg = pdb_segid_map(["AAAAQAAAA", "AAAARAAAA"])
    assert mapping == {"AAAAQAAAA": "AAA0", "AAAARAAAA": "AAA1"}
    assert "numbered" in msg


def test_short_segids_unchanged():
    assert pdb_segid_map(["PROA", "",
                          "W"]) == ({
                              "PROA": "PROA",
                              "": "",
                              "W": "W"
                          }, None)


def test_write_pdb_distinguishes_long_segids(tmp_path):
    u = _fresh()
    n = len(u.segments)
    u.segments.segids = [f"GLYCAN{i + 1}" for i in range(n)]
    with pytest.warns(UserWarning, match="segment IDs longer"):
        out = write_pdb(u.atoms, tmp_path / "x.pdb")
    _, segids = _columns(out)
    assert len(set(segids)) == n  # still one segid per segment
    assert all(len(s) <= 4 for s in segids)
