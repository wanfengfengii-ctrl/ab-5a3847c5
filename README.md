# 深空探测器计数器校时解析服务

探测器本地计数器在一次热切换后可能改变了走时周期。本服务从带接收误差的校时记录中恢复**连续**的地面时间：联合选择相邻记录间的切换边界、切换前后两个正整数周期与整数起始时刻，按计数差分段累计预测值，切换边界处两段共用同一时刻（时间线不在切换处断裂）。

全部比较均使用精确整数运算（中点偏差以二倍整数表示），不依赖任何浮点近似。

## 数学模型

给定 `n`（8–24）条记录：计数 `c_i`（严格递增整数）与地面时刻整数闭区间 `[lo_i, hi_i]`，以及周期共用的整数搜索闭区间 `[pmin, pmax]`。求解变量：

- 切换边界 `k ∈ {1, …, n-1}`：记录 `0..k-1` 属切换前段，记录 `k..n-1` 属切换后段；
- 正整数周期 `T1, T2 ∈ [pmin, pmax]`；
- 整数起始时刻 `t0`（记录 0 的预测地面时刻）。

预测值按计数差分段累计，边界记录 `k-1` 的时刻由两段共用：

```
pred_i = t0 + (c_i - c_0) · T1                                    i < k
pred_i = t0 + (c_{k-1} - c_0) · T1 + (c_i - c_{k-1}) · T2         i ≥ k
```

约束：对全部 `i`，`lo_i ≤ pred_i ≤ hi_i`。

目标（字典序依次最小化）：

1. 最大中点偏差 `max_i |pred_i − (lo_i + hi_i)/2|`；
2. 偏差总和 `Σ_i |pred_i − (lo_i + hi_i)/2|`；
3. 稳定裁决：切换边界 `k`、前周期 `T1`、后周期 `T2`、起始时刻 `t0`（均取较小者）。

中点可能为半整数，因此内部一律使用二倍偏差 `|2·pred_i − lo_i − hi_i|`（恒为整数）进行比较；响应中的有理数均以 `{"num", "den"}` 精确表示。

求解方法：对每个边界 `k`，先用记录区间的两两约束把 `T1`、`T2` 的可行区间精确裁剪（区间交非空 ⟺ 所有下界不超过所有上界），再枚举可行 `(T1, T2)`；对每组参数，`t0` 的可行域是整数区间交集，在其上先按 Chebyshev 中心最小化最大偏差，再向偏差绝对值和的中位数区间投影，精确取到最优 `t0`。测试中以全枚举暴力实现做差分对照（500 组随机用例结果完全一致）。

## 构建与运行

```bash
# 构建并启动（默认端口 8080）
docker compose up --build

# 可配置端口
API_PORT=9000 docker compose up --build
```

服务监听 `0.0.0.0:${API_PORT}`，容器内外使用同一端口；`api` 服务自带健康检查（`GET /health`）。

### 一次性验证服务 `verify`

`verify` 服务通过 `depends_on: service_healthy` 等待 API 健康后，依次运行：

1. **代码测试**：`python -m unittest discover`（求解器精确性、边界连续性、半整数中点、不可行判定、差分对照、API 行为）；
2. **应用构建**：`python -m compileall` 字节码构建全部交付模块；
3. **校时 API 冒烟**：对运行中的容器执行可行 / 不可行 / 校验失败三类请求，断言切换位置、周期、起始时刻、逐记录预测与偏差证据。

随后打印各阶段结果并以退出码汇报（全部通过为 `0`，否则为 `1`），容器自行退出：

```bash
API_PORT=8080 docker compose up --build --abort-on-container-exit --exit-code-from verify
echo $?   # verify 的退出码
```

## API

### `GET /health`

→ `200 {"status": "ok"}`

### `POST /api/calibrations/resolve`

请求体：

```json
{
  "records": [
    {"count": 1000, "window": [4993, 5009]},
    {"count": 1025, "window": [5293, 5309]}
  ],
  "periodWindow": [8, 24]
}
```

- `records`：8–24 条；`count` 为严格递增整数；`window` 为地面时刻整数闭区间 `[minTime, maxTime]`（`minTime ≤ maxTime`）；
- `periodWindow`：切换前后周期共用的整数搜索闭区间 `[minPeriod, maxPeriod]`，`1 ≤ minPeriod ≤ maxPeriod`。

#### 可行 → `200`

```json
{
  "status": "feasible",
  "solution": {
    "switchBoundary": 6,
    "periodBefore": 12,
    "periodAfter": 19,
    "startTime": 5001,
    "sharedTimeAtBoundary": 6501
  },
  "objective": {
    "maxAbsDeviation": {"num": 0, "den": 2},
    "totalAbsDeviation": {"num": 0, "den": 2}
  },
  "records": [
    {
      "index": 0,
      "count": 1000,
      "window": [4993, 5009],
      "segment": "before",
      "predicted": 5001,
      "midpoint": {"num": 10002, "den": 2},
      "deviation": {"num": 0, "den": 2},
      "absDeviation": {"num": 0, "den": 2}
    }
  ]
}
```

- `switchBoundary = k`：记录 `0..k-1` 属 `before` 段，记录 `k..n-1` 属 `after` 段；
- `sharedTimeAtBoundary`：两段共用的边界时刻，即记录 `k-1` 的预测时刻（连续性证据）；
- 所有 `{"num", "den"}` 为精确有理数 `num/den`；`den` 恒为 2（整数区间中点是 1/2 的整数倍），`deviation = predicted − midpoint`，`absDeviation` 为其绝对值；
- `objective` 给出字典序最优的最大中点偏差与偏差总和。

#### 不可行 → `200`

```json
{
  "status": "infeasible",
  "detail": "no switch boundary, positive integer period pair and integer start time place every predicted ground time inside its record window"
}
```

不存在共同连续解释时返回明确的不可行结论，**不会**返回任何局部拟合结果。

#### 请求非法 → `400`

```json
{"status": "error", "errors": ["\"records\" must contain between 8 and 24 entries, got 7"]}
```

记录数越界、计数非严格递增、区间下界大于上界、周期区间非正整数或倒序、非整数取值等均返回 `400` 并列出全部错误。

## 本地开发（无需 Docker）

```bash
python3 -m unittest discover -s tests -t . -v   # 代码测试
API_PORT=8080 python3 -m app.server             # 启动服务
APP_ROOT=$PWD API_BASE_URL=http://127.0.0.1:8080 python3 verify/verify.py  # 一次性验证
```

## 目录结构

```
app/
  solver.py     # 精确整数求解器（模型、裁剪、字典序最优选择）
  server.py     # 标准库 HTTP API（/health、/api/calibrations/resolve）
tests/
  test_solver.py  # 精确恢复、连续性、半整数中点、不可行、暴力差分对照
  test_api.py     # 进程内 API 行为与参数校验
verify/
  verify.py     # 一次性验证：等待健康 → 测试 → 构建 → 冒烟 → 退出码
Dockerfile        # runtime（API）与 verify（一次性验证）两个构建目标
docker-compose.yml
```
