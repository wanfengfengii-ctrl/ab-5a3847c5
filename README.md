# 深空测控校时恢复服务

探测器本地计数器在一次热切换后可能改变走时周期。本服务从带接收误差的
校时记录（每条记录给出计数与地面时刻整数闭区间）恢复**连续**地面时间，
联合求解切换边界、切换前后两个正整数周期与整数起始时刻，避免分段拟合
在切换处断裂。

## 数学模型（全程整数运算，无浮点）

记录 i：严格递增整数计数 c_i，地面时刻闭区间 [lo_i, hi_i]。

- 切换边界 k（1 ≤ k < n）：记录 0..k−1 用周期 p1，记录 k..n−1 用周期 p2；
- 起始时刻 t0：记录 0 的整数地面时刻。

分段累计预测（边界两侧共用同一时刻，不存在跳变）：

```
A_i = p1·(min(c_i, c_k) − c_0) + p2·max(0, c_i − c_k)
t_i = t0 + A_i
```

要求所有 t_i ∈ [lo_i, hi_i]。

为避免半整数中点引入浮点，所有偏差以 **2 倍值**进行整数比较：

```
midpoint_x2_i = lo_i + hi_i
residual_x2_i = 2·t_i − (lo_i + hi_i)
deviation_x2_i = |residual_x2_i|
```

裁决依次最小化（字典序）：

```
(max deviation_x2, sum deviation_x2, k, p1, p2, t0)
```

求解：消去 t0 后可行性是 (p1,p2) 平面上由差约束

```
lo_j − hi_i ≤ A_j − A_i ≤ hi_j − lo_i
```

界定的线性有理多胞形；小可行域整数网格精确枚举；大可行域枚举残差
等值线与可行边界直线排列的有理顶点及沿直线整数格点邻近点。整数 t0
的最优值在可行 t0 区间端点、残差零点与最大偏差中心附近解析取得。
**全程不出现浮点运算，比较不依赖浮点近似。**

## API

`POST /api/calibrations/resolve`

```json
{
  "records": [
    {"count": 10, "low": 100, "high": 102}
  ],
  "period_range": {"low": 1, "high": 1000}
}
```

- 8 至 24 条计数严格递增的记录；
- `period_range` 为切换前后周期共用的正整数搜索闭区间。

成功响应包含切换位置、参数（前/后周期、起始时刻）、逐记录预测与
偏差证据、边界共用时刻；无共同连续解释时返回

```json
{"feasible": false, "conclusion": "不存在满足全部时间闭区间的连续整数分段解释……"}
```

而非任何局部拟合结果。健康检查：`GET /health`。

## 运行

```bash
# 服务（端口由 API_PORT 配置，默认 8080）
API_PORT=9000 docker compose up --build api

# 一次性校验：等待 API 健康后自动运行
# 代码测试 + 应用构建 + 校时 API 冒烟，verify 完成后自行退出，
# 退出码透传（CI 中可直接据此判定成败）
docker compose up --build --abort-on-container-exit --exit-code-from verify
```

`verify` 服务以退出码汇报结果（全部通过退出码 0，否则非 0）。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
uvicorn app.main:app --port 8080
```
