"""校时恢复 HTTP 服务。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .solver import ValidationError, resolve, parse_inputs

app = FastAPI(
    title="深空测控校时恢复服务",
    version="1.0.0",
)


@app.exception_handler(ValidationError)
def _on_validation_error(_req: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"feasible": False, "error": "invalid_input", "detail": str(exc)},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _half(n: int) -> str:
    """把 2 倍放缩的整数偏差呈现为精确分数文本。"""
    if n % 2 == 0:
        return str(n // 2)
    return f"{n}/2"


@app.post("/api/calibrations/resolve")
async def resolve_calibration(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "feasible": False,
                "error": "invalid_input",
                "detail": "请求体必须为合法 JSON 对象",
            },
        )
    recs, p_lo, p_hi = parse_inputs(body)
    # 求解为纯 CPU 密集型，放入线程池以免阻塞事件循环与健康检查。
    sol = await run_in_threadpool(resolve, recs, p_lo, p_hi)

    if sol is None:
        return JSONResponse(
            status_code=200,
            content={
                "feasible": False,
                "conclusion": (
                    "不存在满足全部时间闭区间的连续整数分段解释；"
                    "任何切换边界与正整数周期组合均无法同时落入给定区间。"
                ),
                "searched_boundaries": len(recs) - 1,
                "period_range": {"low": p_lo, "high": p_hi},
            },
        )

    predictions = []
    for i, rec in enumerate(recs):
        pred = sol.predictions[i]
        res2 = sol.residuals2[i]
        predictions.append(
            {
                "index": i,
                "count": rec.count,
                "predicted_time": pred,
                "interval": {"low": rec.low, "high": rec.high},
                "midpoint_x2": rec.low + rec.high,
                "residual_x2": res2,
                "abs_deviation_x2": abs(res2),
                "abs_deviation": _half(abs(res2)),
                "within_interval": rec.low <= pred <= rec.high,
                "segment": (
                    "before" if i < sol.switch_k else "after"
                ),
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "feasible": True,
            "switch_boundary": {
                "index": sol.switch_k,
                "position": sol.switch_k + 1,
                "between_counts": [
                    recs[sol.switch_k - 1].count,
                    recs[sol.switch_k].count,
                ],
                "shared_time": sol.predictions[sol.switch_k],
            },
            "parameters": {
                "period_before": sol.period_before,
                "period_after": sol.period_after,
                "start_time": sol.t0,
            },
            "objective": {
                "max_abs_deviation_x2": sol.max_dev2,
                "max_abs_deviation": _half(sol.max_dev2),
                "sum_abs_deviation_x2": sol.sum_dev2,
                "sum_abs_deviation": _half(sol.sum_dev2),
            },
            "predictions": predictions,
            "tie_break_order": [
                "max_abs_deviation_x2",
                "sum_abs_deviation_x2",
                "switch_boundary.index",
                "period_before",
                "period_after",
                "start_time",
            ],
            "arithmetic": "integer-only (deviations stored as 2x integers)",
        },
    )
