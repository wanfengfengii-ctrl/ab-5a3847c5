#!/usr/bin/env python3
"""One-shot verification service.

Waits for the API to become healthy, then runs, in order:

1. the unit-test suite (solver correctness + in-process API tests),
2. the application build (byte-compilation of every shipped module),
3. a live smoke test of the calibration API (feasible, infeasible and
   validation-error paths against the running container),

and exits with status 0 only if every stage passed.  The container running
this script is expected to stop on its own afterwards.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

APP_ROOT = os.environ.get("APP_ROOT", "/app")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
HEALTH_TIMEOUT_SECONDS = float(os.environ.get("VERIFY_HEALTH_TIMEOUT", "90"))


def wait_for_health() -> bool:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(API_BASE_URL + "/health", timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_unit_tests() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        cwd=APP_ROOT,
    )
    return result.returncode == 0


def run_build() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "app", "tests", "verify"],
        cwd=APP_ROOT,
    )
    return result.returncode == 0


def _post(payload):
    request = urllib.request.Request(
        API_BASE_URL + "/api/calibrations/resolve",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _check(condition, label, failures):
    print(f"[verify]   smoke: {label}: {'ok' if condition else 'FAILED'}", flush=True)
    if not condition:
        failures.append(label)


def run_smoke() -> bool:
    failures = []

    # --- health endpoint -------------------------------------------------
    try:
        with urllib.request.urlopen(API_BASE_URL + "/health", timeout=5) as response:
            healthy = response.status == 200 and json.loads(response.read())["status"] == "ok"
    except Exception:
        healthy = False
    _check(healthy, "GET /health returns ok", failures)

    # --- feasible case: exact known solution ------------------------------
    counts = [1000 + 25 * i for i in range(10)]
    times = [5000 + 300 * i for i in range(6)] + [6500 + 475 * (i - 5) for i in range(6, 10)]
    feasible_payload = {
        "records": [{"count": c, "window": [t - 7, t + 9]} for c, t in zip(counts, times)],
        "periodWindow": [8, 24],
    }
    code, body = _post(feasible_payload)
    _check(code == 200, "feasible case returns HTTP 200", failures)
    _check(body.get("status") == "feasible", "feasible case reports status=feasible", failures)
    solution = body.get("solution", {})
    _check(solution.get("switchBoundary") == 6, "switch boundary is 6", failures)
    _check(solution.get("periodBefore") == 12, "period before switch is 12", failures)
    _check(solution.get("periodAfter") == 19, "period after switch is 19", failures)
    _check(solution.get("startTime") == 5001, "start time is 5001", failures)
    _check(solution.get("sharedTimeAtBoundary") == 6501, "shared boundary moment is 6501", failures)
    objective = body.get("objective", {})
    _check(objective.get("maxAbsDeviation") == {"num": 0, "den": 2}, "max deviation is exactly 0", failures)
    _check(objective.get("totalAbsDeviation") == {"num": 0, "den": 2}, "total deviation is exactly 0", failures)

    records = body.get("records", [])
    _check(len(records) == 10, "response carries all 10 records", failures)
    if len(records) == 10:
        boundary = solution.get("switchBoundary", 6)
        consistent = True
        inside = True
        for idx, item in enumerate(records):
            lo, hi = item["window"]
            inside &= lo <= item["predicted"] <= hi
            expected_segment = "before" if idx < boundary else "after"
            if item["segment"] != expected_segment:
                consistent = False
            if item["deviation"]["num"] != 2 * item["predicted"] - (lo + hi):
                consistent = False
            if item["absDeviation"]["num"] != abs(item["deviation"]["num"]):
                consistent = False
        for idx in range(9):
            step = records[idx + 1]["predicted"] - records[idx]["predicted"]
            count_step = records[idx + 1]["count"] - records[idx]["count"]
            period = solution["periodBefore"] if idx + 1 < boundary else solution["periodAfter"]
            if step != count_step * period:
                consistent = False
        _check(inside, "every prediction lies inside its window", failures)
        _check(consistent, "segments, accumulation and deviation evidence are consistent", failures)

    # --- infeasible case: clear verdict, no partial fit -------------------
    infeasible_payload = {
        "records": [
            {"count": i, "window": [0, 0] if i % 2 == 0 else [100000, 100000]}
            for i in range(8)
        ],
        "periodWindow": [1, 10],
    }
    code, body = _post(infeasible_payload)
    _check(code == 200, "infeasible case returns HTTP 200", failures)
    _check(body.get("status") == "infeasible", "infeasible case reports status=infeasible", failures)
    _check("records" not in body and "solution" not in body, "infeasible case returns no partial fit", failures)

    # --- validation case ---------------------------------------------------
    too_few = {
        "records": [{"count": i, "window": [0, 1]} for i in range(7)],
        "periodWindow": [1, 10],
    }
    code, body = _post(too_few)
    _check(code == 400, "7 records are rejected with HTTP 400", failures)
    _check(body.get("status") == "error" and body.get("errors"), "validation errors are reported", failures)

    return not failures


STAGES = [
    ("api health gate", None),  # placeholder, handled first
    ("unit tests", run_unit_tests),
    ("application build", run_build),
    ("calibration api smoke", run_smoke),
]


def main() -> int:
    results = []
    healthy = wait_for_health()
    results.append(("api health gate", healthy))
    if not healthy:
        print(f"[verify] API at {API_BASE_URL} did not become healthy in time", flush=True)
    else:
        print(f"[verify] API at {API_BASE_URL} is healthy", flush=True)
        for name, stage in STAGES[1:]:
            print(f"[verify] running {name} ...", flush=True)
            try:
                ok = bool(stage())
            except Exception as exc:  # never let a crash hide the verdict
                print(f"[verify] {name} raised {exc!r}", flush=True)
                ok = False
            results.append((name, ok))

    print("[verify] ================= summary =================", flush=True)
    for name, ok in results:
        print(f"[verify] {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    overall = all(ok for _, ok in results)
    print(f"[verify] overall: {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
