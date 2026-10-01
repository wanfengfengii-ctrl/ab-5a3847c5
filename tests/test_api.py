"""HTTP API 测试（基于 FastAPI TestClient）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _payload(lows, highs, counts=None, p_lo=1, p_hi=20):
    if counts is None:
        counts = list(range(len(lows)))
    return {
        "records": [
            {"count": c, "low": lo, "high": hi}
            for c, lo, hi in zip(counts, lows, highs)
        ],
        "period_range": {"low": p_lo, "high": p_hi},
    }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_resolve_success_structure():
    counts = [10, 11, 12, 13, 14, 16, 18, 21, 25, 30]
    k_true, p1, p2, t0 = 4, 5, 3, 100
    ck = counts[k_true]
    times = [
        t0 + p1 * (min(c, ck) - counts[0]) + p2 * max(0, c - ck)
        for c in counts
    ]
    r = client.post(
        "/api/calibrations/resolve",
        json=_payload(times, times, counts),
    )
    assert r.status_code == 200
    data = r.json()
    assert data["feasible"] is True
    assert data["switch_boundary"]["index"] == 4
    assert data["parameters"] == {
        "period_before": 5,
        "period_after": 3,
        "start_time": 100,
    }
    assert data["objective"]["max_abs_deviation_x2"] == 0
    assert len(data["predictions"]) == 10
    for row, t in zip(data["predictions"], times):
        assert row["predicted_time"] == t
        assert row["within_interval"] is True
        assert row["abs_deviation_x2"] == 0
    assert data["switch_boundary"]["between_counts"] == [13, 14]
    assert data["switch_boundary"]["shared_time"] == times[4]


def test_resolve_infeasible_message():
    counts = list(range(8))
    lows = [0, 10, 20, 30, 40, 41, 42, 43]
    highs = [1, 11, 21, 31, 41, 52, 63, 74]
    r = client.post(
        "/api/calibrations/resolve",
        json=_payload(lows, highs, counts, p_lo=1, p_hi=5),
    )
    assert r.status_code == 200
    data = r.json()
    assert data["feasible"] is False
    assert "不存在" in data["conclusion"]
    assert data["searched_boundaries"] == 7


def test_resolve_invalid_body():
    r = client.post("/api/calibrations/resolve", json={"records": []})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_input"


def test_resolve_non_integer_rejected():
    payload = _payload([0] * 8, [0] * 8)
    payload["records"][0]["low"] = 0.5
    r = client.post("/api/calibrations/resolve", json=payload)
    assert r.status_code == 400
