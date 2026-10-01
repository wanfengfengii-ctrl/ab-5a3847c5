#!/bin/sh
# 一次性校验服务：代码测试 + 应用构建 + 校时 API 冒烟。
# 由 Compose 的 depends_on(service_healthy) 保证 API 已健康后才启动。
set -eu

BASE_URL="${API_BASE_URL:-http://api:8080}"

echo "==> [1/3] 应用构建检查（字节码编译）"
python -m compileall -q app

echo "==> [2/3] 代码测试（pytest）"
python -m pytest -q tests

echo "==> [3/3] 校时 API 冒烟：${BASE_URL}"
python - "$BASE_URL" <<'PY'
import json
import sys
import urllib.request

base = sys.argv[1]


def call(method, path, payload=None):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


# 1) 健康检查
status, body = call("GET", "/health")
assert status == 200 and body["status"] == "ok", body
print("  [ok] GET /health")

# 2) 可行案例：p1=5, p2=3, k=4, t0=100
counts = [10, 11, 12, 13, 14, 16, 18, 21, 25, 30]
k, p1, p2, t0 = 4, 5, 3, 100
ck = counts[k]
times = [
    t0 + p1 * (min(c, ck) - counts[0]) + p2 * max(0, c - ck)
    for c in counts
]
payload = {
    "records": [
        {"count": c, "low": t, "high": t} for c, t in zip(counts, times)
    ],
    "period_range": {"low": 1, "high": 20},
}
status, body = call("POST", "/api/calibrations/resolve", payload)
assert status == 200, body
assert body["feasible"] is True, body
assert body["parameters"] == {
    "period_before": 5,
    "period_after": 3,
    "start_time": 100,
}, body["parameters"]
assert body["switch_boundary"]["index"] == 4
assert body["objective"]["max_abs_deviation_x2"] == 0
for row, t in zip(body["predictions"], times):
    assert row["predicted_time"] == t and row["within_interval"]
print("  [ok] POST /api/calibrations/resolve 可行案例，参数精确恢复")

# 3) 含接收误差的宽区间案例
lo = [t - 2 for t in times]
hi = [t + 2 for t in times]
payload = {
    "records": [
        {"count": c, "low": a, "high": b}
        for c, a, b in zip(counts, lo, hi)
    ],
    "period_range": {"low": 1, "high": 20},
}
status, body = call("POST", "/api/calibrations/resolve", payload)
assert status == 200 and body["feasible"] is True
assert body["parameters"]["period_before"] == 5
assert body["parameters"]["period_after"] == 3
for row in body["predictions"]:
    assert row["within_interval"]
print("  [ok] 宽区间案例全部预测落入闭区间，偏差证据完整")

# 4) 明确不可行案例：不得返回局部拟合
bad = {
    "records": [
        {"count": c, "low": a, "high": b}
        for c, a, b in zip(
            range(8),
            [0, 10, 20, 30, 40, 41, 42, 43],
            [1, 11, 21, 31, 41, 52, 63, 74],
        )
    ],
    "period_range": {"low": 1, "high": 5},
}
status, body = call("POST", "/api/calibrations/resolve", bad)
assert status == 200 and body["feasible"] is False
assert "不存在" in body["conclusion"]
assert "parameters" not in body
print("  [ok] 不可行案例返回明确结论，未给出局部拟合参数")

# 5) 非法输入
import urllib.error
try:
    call("POST", "/api/calibrations/resolve", {"records": []})
except urllib.error.HTTPError as e:
    err = json.loads(e.read().decode())
    assert e.code == 400 and err["error"] == "invalid_input", err
    print("  [ok] 非法输入返回 400 与明确错误")
else:
    raise AssertionError("非法输入应返回 400")
PY

echo ""
echo "全部校验通过：代码测试、应用构建与 API 冒烟均成功。"
