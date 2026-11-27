#!/bin/bash
set -eu
cd /app
export PYTHONPATH=/app/lib:/app
git reset --hard "$1"
git clean -fd
git checkout "$1"
printf 'PROBE_PARENT_COMMIT=%s\n' "$(git rev-parse HEAD)"
mkdir -p /logs/verifier
start=$(date +%s%N)
set +e
bash /tests/test.sh
verifier_status=$?
set -e
end=$(date +%s%N)
printf 'PROBE_VERIFIER_EXIT=%s\nPROBE_VERIFIER_MILLISECONDS=%s\n' "$verifier_status" "$(((end - start) / 1000000))"
test -f /logs/verifier/output.json
tail -26 /logs/verifier/run-script-stdout.txt
tail -20 /logs/verifier/run-script-stderr.txt
python3 - <<'PY'
import ast
import collections
import json

with open("/tests/config.json") as f:
    config = json.load(f)
with open("/logs/verifier/output.json") as f:
    tests = json.load(f)["tests"]

print("PROBE_ALL_TEST_STATUSES", json.dumps(collections.Counter(t["status"] for t in tests)))
statuses = {test["name"]: test["status"] for test in tests}
ok = True
for key, expected in [("fail_to_pass", "FAILED"), ("pass_to_pass", "PASSED")]:
    names = config.get(key) or []
    if isinstance(names, str):
        names = ast.literal_eval(names)
    observed = collections.Counter(statuses.get(name, "MISSING") for name in set(names))
    print(f"PROBE_{key.upper()}", json.dumps(observed))
    ok &= observed.get(expected, 0) == len(set(names))
    for name in names:
        if statuses.get(name) != expected:
            print("PROBE_UNEXPECTED_TEST", statuses.get(name, "MISSING"), name)
raise SystemExit(0 if ok else 1)
PY
