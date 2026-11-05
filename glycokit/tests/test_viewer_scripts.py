import numpy as np
import pytest

from glycokit import viewer_scripts as vs


def test_vmd_any_values(universe, tmp_path):
    ca = universe.select_atoms("protein and name CA")[:5]
    atoms, cols = vs.residue_atom_rows(ca)
    vals = np.arange(5, dtype=float)[None] / 2  # 0 .. 2
    text = vs.write_vmd_user_script(
        tmp_path / "x.vmd", atoms, vals, [0],
        columns=cols).read_text(encoding="utf-8")
    set_line = next(
        l for l in text.splitlines() if l.startswith("$sel set user"))
    assert len(set_line.split("{")[1].rstrip("}").split()) == len(atoms)
    assert "mol scaleminmax $molid 0 0 2\n" in text
    assert "$sel frame 0" in text


def test_vmd_rejects_bad_shapes(universe, tmp_path):
    ca = universe.select_atoms("protein and name CA")[:5]
    with pytest.raises(ValueError, match="columns"):
        vs.write_vmd_user_script(tmp_path / "x.vmd", ca, np.zeros((1, 3)), [0])
    with pytest.raises(ValueError, match="frames"):
        vs.write_vmd_user_script(tmp_path / "x.vmd", ca, np.zeros((2, 5)), [0])


def test_vmd_needs_files(small_universe, tmp_path):
    ca = small_universe.select_atoms("name CA")  # Merge()d universe
    with pytest.raises(ValueError, match="topology"):
        vs.write_vmd_user_script(
            tmp_path / "x.vmd", ca, np.zeros((1, len(ca))), [0])


def test_chimerax_with_defattr(universe, tmp_path):
    ca = universe.select_atoms("protein and name CA")[:5]
    attr = vs.write_defattr(
        tmp_path / "v.defattr", ca, np.arange(5.0), "my_value")
    cxc = vs.write_chimerax_script(
        tmp_path / "v.cxc", universe.filename, attribute_files=[attr],
        color_attribute="r:my_value").read_text(encoding="utf-8").splitlines()
    assert cxc[0] == "close session"
    assert any(l.startswith("open ") and l.endswith('v.defattr"') for l in cxc)
    assert any(l.startswith("color byattribute r:my_value") for l in cxc)
