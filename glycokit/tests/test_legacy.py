"""
glyco --mode v1 / --mode v2 versus GLYCO v1 / GLYCO-2, one test per structure
and gating check. Structures come from --glyco-structure / --glyco-indir.
"""

import pytest

from glycokit.tests.conftest import LEGACY_CHECKS

pytestmark = [pytest.mark.slow, pytest.mark.legacy, pytest.mark.glyco_kind("structure")]


def _format_details(details, limit=10):
    lines = []
    for key, value in details.items():
        if not key.startswith("n_") or not value:
            continue
        name = key[2:]
        lines.append(f"  {name}: {value}")
        for example in details.get(name, [])[:limit]:
            lines.append(f"      {example}")
    return "\n".join(lines) or "  (no details)"


def test_legacy_run_completes(case, compare_params, legacy_report):
    """compare.py (all four programs) ran to completion and wrote a summary."""
    r = legacy_report(case, compare_params)
    if r.returncode is None:
        pytest.fail(f"compare.py timed out\n{r.tail()}", pytrace=False)
    if r.report is None:
        pytest.fail(
            f"compare.py did not finish (exit code {r.returncode})\n{r.tail()}",
            pytrace=False,
        )


@pytest.mark.parametrize(
    "check", [pytest.param(label, id=short) for short, label in LEGACY_CHECKS]
)
def test_legacy_check(case, compare_params, check, legacy_report):
    r = legacy_report(case, compare_params)
    if r.report is None:
        pytest.fail(
            "no compare.py summary (see test_legacy_run_completes)", pytrace=False
        )
    details = r.report.get("checks", {}).get(check)
    if details is None:
        pytest.fail(
            f"check {check!r} missing from the summary; is compare.py up to date?",
            pytrace=False,
        )
    if not details["passed"]:
        where = r.report.get("output_dir")
        hint = (
            f"\noutputs kept in: {where}"
            if where
            else "\nrerun with --glyco-compare-keep to keep the outputs"
        )
        pytest.fail(f"{check}: FAIL\n{_format_details(details)}{hint}", pytrace=False)
