import sys


def test_ccd_carbohydrates_without_biotite(monkeypatch):
    """Without biotite, CCD-based detection is disabled, not an error."""
    from glycokit import standardize  # pylint: disable=import-outside-toplevel

    monkeypatch.setitem(sys.modules, "biotite.structure.info", None)
    standardize._ccd_carbohydrates.cache_clear()  # pylint: disable=protected-access
    try:
        assert standardize._ccd_carbohydrates() == frozenset()  # pylint: disable=protected-access
    finally:
        standardize._ccd_carbohydrates.cache_clear()  # pylint: disable=protected-access
