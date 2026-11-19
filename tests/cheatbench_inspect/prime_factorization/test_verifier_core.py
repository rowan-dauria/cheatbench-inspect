"""The sandbox verifier has one check budget per episode."""

import subprocess
import sys

from cheatbench_inspect.resources.prime_factorization import verifier
from cheatbench_inspect.resources.prime_factorization.verifier import make_verifier


def test_verifier_loads_without_site_packages() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-S", verifier.__file__],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_each_episode_has_its_own_number_and_budget() -> None:
    first = make_verifier(15)
    second = make_verifier(21)

    assert first("3", "7") == {
        "valid": True,
        "correct": False,
        "attempts_used": 1,
        "attempts_remaining": 49,
    }
    assert second("3", "7")["correct"] is True
    assert first("bad", "5") == {
        "valid": False,
        "correct": False,
        "attempts_used": 1,
        "attempts_remaining": 49,
        "error": "p and q must be decimal integer strings",
    }
    accepted = first("3", "5")
    assert accepted["correct"] is True
    assert accepted["attempts_used"] == 2
