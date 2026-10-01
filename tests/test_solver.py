"""求解器测试：与暴力穷举完全对照。"""

from __future__ import annotations

import random

import pytest

from app.solver import (
    Record,
    Solution,
    ValidationError,
    parse_inputs,
    resolve,
    solve_for_boundary,
)


def brute_force(recs, p_lo, p_hi):
    """逐 (k,p1,p2,t0) 穷举的参考实现（仅用于小规模对照）。"""
    n = len(recs)
    counts = [r.count for r in recs]
    best = None
    for k in range(1, n):
        ck = counts[k]
        c0 = counts[0]
        for p1 in range(p_lo, p_hi + 1):
            for p2 in range(p_lo, p_hi + 1):
                accum = [
                    p1 * (min(c, ck) - c0) + p2 * max(0, c - ck)
                    for c in counts
                ]
                t_lo = max(r.low - a for r, a in zip(recs, accum))
                t_hi = min(r.high - a for r, a in zip(recs, accum))
                for t0 in range(t_lo, t_hi + 1):
                    d2 = []
                    for r, a in zip(recs, accum):
                        pred = t0 + a
                        d2.append(abs(2 * pred - (r.low + r.high)))
                    sol = Solution(
                        switch_k=k,
                        period_before=p1,
                        period_after=p2,
                        t0=t0,
                        max_dev2=max(d2),
                        sum_dev2=sum(d2),
                        predictions=tuple(t0 + a for a in accum),
                        residuals2=tuple(
                            2 * (t0 + a) - (r.low + r.high)
                            for r, a in zip(recs, accum)
                        ),
                    )
                    if best is None or sol.key() < best.key():
                        best = sol
    return best


def make_recs(counts, lows, highs):
    return tuple(Record(c, lo, hi) for c, lo, hi in zip(counts, lows, highs))


def assert_solution_valid(sol, recs):
    assert sol is not None
    for i, r in enumerate(recs):
        p = sol.predictions[i]
        assert r.low <= p <= r.high, f"记录 {i} 预测 {p} 越出 [{r.low},{r.high}]"
        assert sol.residuals2[i] == 2 * p - (r.low + r.high)


def test_exact_recovery_with_switch():
    # 真实参数：p1=5, p2=3, k=4, t0=100
    counts = [10, 11, 12, 13, 14, 16, 18, 21, 25, 30]
    k_true, p1_true, p2_true, t0 = 4, 5, 3, 100
    ck = counts[k_true]
    times = [
        t0 + p1_true * (min(c, ck) - counts[0])
        + p2_true * max(0, c - ck)
        for c in counts
    ]
    recs = make_recs(counts, times, times)  # 区间宽度 0
    sol = resolve(recs, 1, 20)
    assert_solution_valid(sol, recs)
    assert sol.max_dev2 == 0
    assert sol.switch_k == k_true
    assert sol.period_before == p1_true
    assert sol.period_after == p2_true
    assert sol.t0 == t0


def test_boundary_shared_time_continuity():
    counts = [0, 1, 2, 3, 4, 5, 6, 7]
    # k=3：p1=2 走到 count=3 时 t=1000+6=1006；p2 段必须从同一时刻起算
    k_true, p1_true, p2_true, t0 = 3, 2, 7, 1000
    ck = counts[k_true]
    times = [
        t0 + p1_true * (min(c, ck) - 0) + p2_true * max(0, c - ck)
        for c in counts
    ]
    recs = make_recs(counts, times, times)
    sol = resolve(recs, 1, 15)
    assert sol is not None
    # 边界记录(k-1 与 k)的预测必须连续、共用同一模型
    assert sol.predictions[k_true - 1] == times[k_true - 1]
    assert sol.predictions[k_true] == times[k_true]
    # 手工核验边界共用：A_k = p1*(c_k-c0)，不含任何额外跳变
    assert sol.predictions[k_true] - sol.predictions[k_true - 1] == (
        p1_true * (counts[k_true] - counts[k_true - 1])
    )


def test_infeasible_returns_none():
    # 两段斜率要求相互矛盾的区间，无共同连续解释
    counts = [0, 1, 2, 3, 4, 5, 6, 7]
    lows = [0, 10, 20, 30, 40, 41, 42, 43]
    highs = [1, 11, 21, 31, 41, 52, 63, 74]
    recs = make_recs(counts, lows, highs)
    # 前段每计数差至少 9，后段区间允许 10..11，但周期搜索域 [1,5] 无法解释
    assert resolve(recs, 1, 5) is None


def test_wide_intervals_half_integer_midpoint():
    # 区间宽度为奇数，中点为半整数，验证 2 倍偏差表示与裁决
    counts = [0, 2, 4, 6, 8, 10, 12, 14]
    # 真实 p1=3, p2=5, k=4, t0=10
    k_true, p1_true, p2_true, t0 = 4, 3, 5, 10
    ck = counts[k_true]
    center = [
        t0 + p1_true * (min(c, ck)) + p2_true * max(0, c - ck)
        for c in counts
    ]
    recs = make_recs(counts, [t - 1 for t in center], [t + 2 for t in center])
    sol = resolve(recs, 1, 10)
    assert_solution_valid(sol, recs)
    assert sol.period_before == 3 and sol.period_after == 5


def test_fuzz_matches_bruteforce_small():
    rng = random.Random(20261001)
    for trial in range(60):
        n = rng.randint(8, 10)
        counts = sorted(rng.sample(range(0, 60), n))
        k_true = rng.randint(2, n - 2)
        p1_true = rng.randint(1, 5)
        p2_true = rng.randint(1, 5)
        t0_true = rng.randint(-50, 50)
        ck = counts[k_true]
        c0 = counts[0]
        true_times = [
            t0_true + p1_true * (min(c, ck) - c0)
            + p2_true * max(0, c - ck)
            for c in counts
        ]
        w = rng.randint(0, 4)
        recs = tuple(
            Record(c, t - w, t + w) for c, t in zip(counts, true_times)
        )
        p_lo, p_hi = 1, 7
        expected = brute_force(recs, p_lo, p_hi)
        got = resolve(recs, p_lo, p_hi)
        if expected is None:
            assert got is None, f"trial {trial}: 期望不可行"
        else:
            assert got is not None, f"trial {trial}: 期望可行"
            assert got.key() == expected.key(), (
                f"trial {trial}: {got.key()} != {expected.key()}"
            )
            assert got.predictions == expected.predictions
            assert_solution_valid(got, recs)


def test_fuzz_infeasible_matches_bruteforce():
    rng = random.Random(42)
    for trial in range(40):
        n = rng.randint(8, 10)
        counts = sorted(rng.sample(range(0, 40), n))
        recs = tuple(
            Record(c, rng.randint(-30, 30), rng.randint(-30, 30) + rng.randint(0, 3))
            for c in counts
        )
        # 规整 low<=high
        recs = tuple(
            Record(r.count, min(r.low, r.high), max(r.low, r.high))
            for r in recs
        )
        expected = brute_force(recs, 1, 6)
        got = resolve(recs, 1, 6)
        if expected is None:
            assert got is None
        else:
            assert got is not None
            assert got.key() == expected.key()


def test_large_period_range_matches_indirect():
    # 大周期域（走路径二）与缩小到含最优的小域（走路径一）结论一致
    rng = random.Random(7)
    n = 10
    counts = sorted(rng.sample(range(0, 200), n))
    k_true = 4
    p1_true, p2_true, t0 = 37, 91, 1000
    ck = counts[k_true]
    c0 = counts[0]
    times = [
        t0 + p1_true * (min(c, ck) - c0) + p2_true * max(0, c - ck)
        for c in counts
    ]
    recs = tuple(Record(c, t - 2, t + 2) for c, t in zip(counts, times))
    big = resolve(recs, 1, 100_000)
    small = resolve(recs, 30, 100)
    assert big is not None and small is not None
    assert big.key() == small.key()


def test_tie_break_on_boundary_then_periods():
    # 构造完全对称数据：p1=p2 时任意切换边界都可行且偏差相同，
    # 应裁决最小边界、最小周期、最小 t0
    counts = [0, 1, 2, 3, 4, 5, 6, 7]
    p, t0 = 4, 0
    times = [t0 + p * c for c in counts]
    recs = make_recs(counts, times, times)
    sol = resolve(recs, 1, 10)
    assert sol.max_dev2 == 0 and sol.sum_dev2 == 0
    assert sol.switch_k == 1
    assert sol.period_before == p and sol.period_after == p
    assert sol.t0 == t0


def test_per_boundary_grid_path_correctness():
    # 直接对照固定边界下的小网格枚举与大域顶点法
    rng = random.Random(99)
    for trial in range(25):
        n = 10
        counts = sorted(rng.sample(range(0, 120), n))
        k = rng.randint(1, n - 1)
        recs = tuple(
            Record(c, rng.randint(-200, 200), rng.randint(-200, 200) + rng.randint(0, 6))
            for c in counts
        )
        recs = tuple(
            Record(r.count, min(r.low, r.high), max(r.low, r.high))
            for r in recs
        )
        small = solve_for_boundary(recs, 1, 12, k)
        # 大域包含 [1,12]，最优值不应更差
        big = solve_for_boundary(recs, 1, 500_000, k)
        if small is None:
            return
        assert big is not None
        assert (big.max_dev2, big.sum_dev2) <= (
            small.max_dev2,
            small.sum_dev2,
        )


def test_parse_validation():
    with pytest.raises(ValidationError):
        parse_inputs({})
    good = {
        "records": [
            {"count": i, "low": i, "high": i} for i in range(8)
        ],
        "period_range": {"low": 1, "high": 5},
    }
    recs, lo, hi = parse_inputs(good)
    assert len(recs) == 8 and lo == 1 and hi == 5

    bad_count = {
        "records": [
            {"count": 0 if i == 0 else 0, "low": 0, "high": 1}
            for i in range(8)
        ],
        "period_range": {"low": 1, "high": 5},
    }
    with pytest.raises(ValidationError):
        parse_inputs(bad_count)

    bad_range = {
        "records": [
            {"count": i, "low": 2, "high": 1} for i in range(8)
        ],
        "period_range": {"low": 1, "high": 5},
    }
    with pytest.raises(ValidationError):
        parse_inputs(bad_range)


def test_record_count_bounds():
    payload = {
        "records": [{"count": i, "low": 0, "high": 0} for i in range(7)],
        "period_range": {"low": 1, "high": 5},
    }
    with pytest.raises(ValidationError):
        parse_inputs(payload)
    payload = {
        "records": [{"count": i, "low": 0, "high": 0} for i in range(25)],
        "period_range": {"low": 1, "high": 5},
    }
    with pytest.raises(ValidationError):
        parse_inputs(payload)
