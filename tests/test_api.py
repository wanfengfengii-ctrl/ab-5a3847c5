"""HTTP API tests: run the server in-process on an ephemeral port."""

import json
import threading
import unittest
import urllib.error
import urllib.request

from http.server import ThreadingHTTPServer

from app.server import CalibrationHandler, RESOLVE_PATH


def _feasible_payload():
    counts = [1000 + 25 * i for i in range(10)]
    times = [5000 + 300 * i for i in range(6)] + [6500 + 475 * (i - 5) for i in range(6, 10)]
    return {
        "records": [
            {"count": c, "window": [t - 7, t + 9]} for c, t in zip(counts, times)
        ],
        "periodWindow": [8, 24],
    }


def _infeasible_payload():
    return {
        "records": [
            {"count": i, "window": [0, 0] if i % 2 == 0 else [100000, 100000]}
            for i in range(8)
        ],
        "periodWindow": [1, 10],
    }


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), CalibrationHandler)
        cls.server.daemon_threads = True
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _post(self, payload, path=RESOLVE_PATH):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._url(path), data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health(self):
        with urllib.request.urlopen(self._url("/health"), timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read().decode("utf-8")), {"status": "ok"})

    def test_resolve_feasible(self):
        status, body = self._post(_feasible_payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "feasible")
        solution = body["solution"]
        self.assertEqual(solution["switchBoundary"], 6)
        self.assertEqual(solution["periodBefore"], 12)
        self.assertEqual(solution["periodAfter"], 19)
        self.assertEqual(solution["startTime"], 5001)
        self.assertEqual(solution["sharedTimeAtBoundary"], 6501)
        self.assertEqual(body["objective"]["maxAbsDeviation"], {"num": 0, "den": 2})
        self.assertEqual(body["objective"]["totalAbsDeviation"], {"num": 0, "den": 2})
        self.assertEqual(len(body["records"]), 10)
        for item in body["records"]:
            lo, hi = item["window"]
            self.assertTrue(lo <= item["predicted"] <= hi)
            self.assertEqual(item["midpoint"]["num"], lo + hi)
            self.assertEqual(item["deviation"]["num"], 2 * item["predicted"] - lo - hi)
            self.assertEqual(item["absDeviation"]["num"], abs(item["deviation"]["num"]))
            self.assertEqual(item["deviation"]["den"], 2)

    def test_resolve_infeasible(self):
        status, body = self._post(_infeasible_payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "infeasible")
        self.assertIn("detail", body)
        self.assertNotIn("records", body)  # no partial fit is ever returned

    def test_validation_errors(self):
        bad_payloads = [
            {},  # missing everything
            {"records": [], "periodWindow": [1, 5]},  # too few records
            {"records": [{"count": i, "window": [0, 1]} for i in range(25)], "periodWindow": [1, 5]},
            {"records": [{"count": i, "window": [0, 1]} for i in range(8)]},  # no periodWindow
            {"records": [{"count": i, "window": [0, 1]} for i in range(8)], "periodWindow": [0, 5]},
            {"records": [{"count": i, "window": [0, 1]} for i in range(8)], "periodWindow": [6, 5]},
            {"records": [{"count": i, "window": [1, 0]} for i in range(8)], "periodWindow": [1, 5]},
            {"records": [{"count": 5, "window": [0, 1]}] * 8, "periodWindow": [1, 5]},  # not increasing
            {"records": [{"count": i, "window": [0, 1.5]} for i in range(8)], "periodWindow": [1, 5]},
            {"records": [{"count": True, "window": [0, 1]}] * 8, "periodWindow": [1, 5]},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=str(payload)[:80]):
                status, body = self._post(payload)
                self.assertEqual(status, 400)
                self.assertEqual(body["status"], "error")
                self.assertTrue(body["errors"])

    def test_malformed_json(self):
        request = urllib.request.Request(
            self._url(RESOLVE_PATH), data=b"{not json", method="POST"
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(ctx.exception.code, 400)

    def test_unknown_routes(self):
        with urllib.request.urlopen(self._url("/health"), timeout=5):
            pass
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self._url("/nope"), timeout=5)
        self.assertEqual(ctx.exception.code, 404)
        status, _ = self._post({"records": [], "periodWindow": [1, 2]}, path="/elsewhere")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
