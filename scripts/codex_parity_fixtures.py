"""Capture the pre-switch Codex first request on Prime Q4.

Run on the legacy harness checkout with this script and its test helper available.
"""

import json
import tempfile
from pathlib import Path

from tests.cheatbench_inspect.harnesses.codex_test_support import normalize, run_q4

DEST = Path("tests/cheatbench_inspect/harnesses/fixtures/codex_parity")


def main() -> None:
    if not Path("src/cheatbench_inspect/resources/codex/episode.py").is_file():
        raise SystemExit("Capture the reference from the pre-switch Codex harness (7af2ecb).")
    with tempfile.TemporaryDirectory() as directory:
        log, requests, _ = run_q4(Path(directory))
        assert log.status == "success", log.error
        assert log.samples
        assert log.samples[0].store["harness_artifacts"]["grader_state"]["is_correct"], log.samples[
            0
        ].store
        assert len(requests) == 3
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "first_request.json").write_text(
        normalize(json.dumps(requests[0], indent=2, ensure_ascii=False)) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
