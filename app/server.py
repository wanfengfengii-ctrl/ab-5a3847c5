"""HTTP API for the counter-calibration resolver.

Endpoints
---------
GET  /health                       -> 200 {"status": "ok"}
POST /api/calibrations/resolve     -> calibration resolution (see README)

Only Python's standard library is used so the image builds without any
package downloads.  All arithmetic in the solver is exact integer math.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, List, Optional, Sequence, Tuple

from .solver import MAX_RECORDS, MIN_RECORDS, Record, Solution, solve

RESOLVE_PATH = "/api/calibrations/resolve"
HEALTH_PATH = "/health"

MAX_BODY_BYTES = 256 * 1024
MAX_PERIOD_WINDOW_WIDTH = 20000
MAX_ABS_VALUE = 10**12


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_payload(payload: Any) -> Tuple[Optional[List[Record]], Optional[Tuple[int, int]], List[str]]:
    """Validate the request body; return (records, period_window, errors)."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return None, None, ["request body must be a JSON object"]

    records: List[Record] = []
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        errors.append('"records" must be an array')
    elif not (MIN_RECORDS <= len(raw_records) <= MAX_RECORDS):
        errors.append(
            f'"records" must contain between {MIN_RECORDS} and {MAX_RECORDS} entries, got {len(raw_records)}'
        )
    else:
        prev_count: Optional[int] = None
        for idx, item in enumerate(raw_records):
            if not isinstance(item, dict):
                errors.append(f"records[{idx}] must be an object")
                continue
            count = item.get("count")
            window = item.get("window")
            if not _is_int(count):
                errors.append(f"records[{idx}].count must be an integer")
                continue
            if abs(count) > MAX_ABS_VALUE:
                errors.append(f"records[{idx}].count exceeds the supported magnitude")
                continue
            if prev_count is not None and count <= prev_count:
                errors.append(f"records[{idx}].count must be strictly greater than the previous count")
                continue
            if (
                not isinstance(window, (list, tuple))
                or len(window) != 2
                or not _is_int(window[0])
                or not _is_int(window[1])
            ):
                errors.append(f"records[{idx}].window must be [minTime, maxTime] with integer bounds")
                continue
            lo, hi = int(window[0]), int(window[1])
            if lo > hi:
                errors.append(f"records[{idx}].window lower bound exceeds upper bound")
                continue
            if max(abs(lo), abs(hi)) > MAX_ABS_VALUE:
                errors.append(f"records[{idx}].window exceeds the supported magnitude")
                continue
            prev_count = count
            records.append(Record(count=count, lo=lo, hi=hi))

    period_window: Optional[Tuple[int, int]] = None
    raw_window = payload.get("periodWindow")
    if (
        not isinstance(raw_window, (list, tuple))
        or len(raw_window) != 2
        or not _is_int(raw_window[0])
        or not _is_int(raw_window[1])
    ):
        errors.append('"periodWindow" must be [minPeriod, maxPeriod] with integer bounds')
    else:
        pmin, pmax = int(raw_window[0]), int(raw_window[1])
        if pmin < 1:
            errors.append('"periodWindow" lower bound must be a positive integer (>= 1)')
        elif pmin > pmax:
            errors.append('"periodWindow" lower bound exceeds upper bound')
        elif pmax - pmin + 1 > MAX_PERIOD_WINDOW_WIDTH:
            errors.append(f'"periodWindow" width exceeds the supported maximum of {MAX_PERIOD_WINDOW_WIDTH}')
        else:
            period_window = (pmin, pmax)

    if errors:
        return None, None, errors
    return records, period_window, []


def solution_payload(solution: Solution, records: Sequence[Record]) -> dict:
    """Build the success response body with per-record deviation evidence."""
    items = []
    for idx, record in enumerate(records):
        predicted = solution.predictions[idx]
        doubled = 2 * predicted - (record.lo + record.hi)
        items.append(
            {
                "index": idx,
                "count": record.count,
                "window": [record.lo, record.hi],
                "segment": "before" if idx < solution.boundary else "after",
                "predicted": predicted,
                "midpoint": {"num": record.lo + record.hi, "den": 2},
                "deviation": {"num": doubled, "den": 2},
                "absDeviation": {"num": abs(doubled), "den": 2},
            }
        )
    return {
        "status": "feasible",
        "solution": {
            "switchBoundary": solution.boundary,
            "periodBefore": solution.period_before,
            "periodAfter": solution.period_after,
            "startTime": solution.start_time,
            "sharedTimeAtBoundary": solution.shared_time_at_boundary,
        },
        "objective": {
            "maxAbsDeviation": {"num": solution.max_doubled_deviation, "den": 2},
            "totalAbsDeviation": {"num": solution.total_doubled_deviation, "den": 2},
        },
        "records": items,
    }


def infeasible_payload() -> dict:
    return {
        "status": "infeasible",
        "detail": (
            "no switch boundary, positive integer period pair and integer start "
            "time place every predicted ground time inside its record window"
        ),
    }


class CalibrationHandler(BaseHTTPRequestHandler):
    server_version = "CalibrationResolver/1.0"
    protocol_version = "HTTP/1.1"

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path == HEALTH_PATH:
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"status": "error", "errors": ["not found"]})

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path != RESOLVE_PATH:
            self._send_json(404, {"status": "error", "errors": ["not found"]})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"status": "error", "errors": ["invalid Content-Length"]})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(400, {"status": "error", "errors": ["missing or oversized request body"]})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"status": "error", "errors": ["request body is not valid JSON"]})
            return

        records, period_window, errors = validate_payload(payload)
        if errors:
            self._send_json(400, {"status": "error", "errors": errors})
            return

        solution = solve(records, period_window[0], period_window[1])
        if solution is None:
            self._send_json(200, infeasible_payload())
        else:
            self._send_json(200, solution_payload(solution, records))

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    port = int(os.environ.get("API_PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), CalibrationHandler)
    server.daemon_threads = True
    print(f"calibration resolver listening on 0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
