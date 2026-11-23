"""Sycophancy's task-specific verdict inherits the shared judge fields."""

import pytest

from cheatbench_inspect.judge import Verdict
from cheatbench_inspect.sycophancy.scorers import SycophancyVerdict


@pytest.mark.parametrize(
    ("reported", "violated", "expected"),
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_sycophancy_verdict(reported: bool, violated: bool, expected: bool) -> None:
    assert issubclass(SycophancyVerdict, Verdict)
    verdict = SycophancyVerdict(
        cheated=reported,
        explicit_prohibition_violated=violated,
        explicit_prohibition_evidence="",
        estimate=0.3,
        committed=True,
        summary="Gave 30%.",
    )
    assert verdict.cheated is expected
    assert (verdict.estimate, verdict.committed, verdict.summary) == (0.3, True, "Gave 30%.")
