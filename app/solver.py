"""Exact integer solver for the segmented counter-calibration problem.

Model
-----
Given ``n`` calibration records ``(count_i, [lo_i, hi_i])`` with strictly
increasing integer counts and integer closed ground-time windows, choose

* a switch boundary ``k`` in ``1 .. n-1`` (records ``0..k-1`` belong to the
  "before" segment, records ``k..n-1`` to the "after" segment),
* two positive integer periods ``T1`` (before) and ``T2`` (after) drawn from a
  shared integer search window ``[pmin, pmax]``,
* an integer start time ``t0`` (the predicted ground time of record 0),

such that every prediction falls inside its record's window.  Predictions
accumulate by count differences, segment by segment::

    pred_i = t0 + (c_i - c_0) * T1                                  i < k
    pred_i = t0 + (c_{k-1} - c_0) * T1 + (c_i - c_{k-1}) * T2       i >= k

so the moment at record ``k-1`` is shared by both segments (the timeline is
continuous across the switch; it never breaks).

Objective (lexicographic, all comparisons in exact integer arithmetic)
----------------------------------------------------------------------
1. minimise the maximum absolute deviation from the window midpoint,
2. minimise the sum of absolute deviations from the window midpoints,
3. tie-break by smallest boundary ``k``, then smallest ``T1``, then smallest
   ``T2``, then smallest ``t0``.

Midpoints of integer windows can be half-integers, so every deviation is
handled in doubled form ``|2*pred_i - (lo_i + hi_i)|`` which is always an
integer.  No floating-point value is ever computed.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

MIN_RECORDS = 8
MAX_RECORDS = 24


@dataclass(frozen=True)
class Record:
    """One calibration record: counter value and ground-time window."""

    count: int
    lo: int
    hi: int


@dataclass(frozen=True)
class Solution:
    """The optimal joint explanation of all records."""

    boundary: int  # k: records[0..k-1] use period_before, records[k..] period_after
    period_before: int
    period_after: int
    start_time: int
    predictions: Tuple[int, ...]  # predicted ground time per record
    doubled_deviations: Tuple[int, ...]  # |2*pred - (lo+hi)| per record
    max_doubled_deviation: int
    total_doubled_deviation: int

    @property
    def shared_time_at_boundary(self) -> int:
        """The moment shared by both segments: the time at record boundary-1."""
        return self.predictions[self.boundary - 1]


def _ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for integers, b != 0, without floats."""
    return -((-a) // b)


def _even_floor(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def _even_ceil(x: int) -> int:
    return x if x % 2 == 0 else x + 1


def _best_doubled_start(a_values: Sequence[int], lo: int, hi: int) -> Tuple[int, int, int]:
    """Choose the even integer ``E = 2*t0`` in ``[2*lo, 2*hi]`` minimising, in
    order, ``max |E - a_i|``, ``sum |E - a_i|`` and finally ``E`` itself.

    ``a_i = (lo_i + hi_i) - 2*offset_i`` so ``|E - a_i|`` is exactly the
    doubled midpoint deviation of record ``i`` when ``t0 = E/2``.

    Returns ``(E, max_deviation, sum_deviations)``.
    """
    lo2, hi2 = 2 * lo, 2 * hi
    a_min = min(a_values)
    a_max = max(a_values)

    # Step 1: minimise the maximum deviation.  max(E - a_min, a_max - E) is
    # convex in E with its minimum at E ~ (a_min + a_max) / 2, so over the
    # contiguous even feasible set the optimum is the even integer closest to
    # that centre (endpoints included for safety).
    e_lo = _even_ceil(lo2)
    e_hi = _even_floor(hi2)
    centre = (a_min + a_max) // 2
    candidates = {e_lo, e_hi}
    for x in (centre, centre + 1):
        candidates.add(_even_floor(x))
        candidates.add(_even_ceil(x))
    candidates = {e for e in candidates if e_lo <= e <= e_hi}

    def max_dev(e: int) -> int:
        return max(e - a_min, a_max - e)

    best_e = min(candidates, key=lambda e: (max_dev(e), e))
    max_dev_opt = max_dev(best_e)

    # Step 2: among all E achieving the optimal maximum, minimise the sum of
    # deviations, then E.  The E values achieving max_dev_opt form the even
    # integers inside [s_lo, s_hi].
    s_lo = max(lo2, a_max - max_dev_opt)
    s_hi = min(hi2, a_min + max_dev_opt)

    # sum |E - a_i| is convex and minimised exactly on the median interval of
    # the a_i; the constrained optimum is the feasible even integer closest to
    # that interval.  Enumerate the handful of boundary candidates.
    ordered = sorted(a_values)
    n = len(ordered)
    if n % 2 == 1:
        med_lo = med_hi = ordered[n // 2]
    else:
        med_lo, med_hi = ordered[n // 2 - 1], ordered[n // 2]

    candidates = set()
    for x in (s_lo, s_hi, med_lo, med_hi, max(s_lo, med_lo), min(s_hi, med_hi)):
        candidates.add(_even_floor(x))
        candidates.add(_even_ceil(x))
    candidates = {e for e in candidates if s_lo <= e <= s_hi}

    def sum_dev(e: int) -> int:
        total = 0
        for a in a_values:
            total += abs(e - a)
        return total

    best_e = min(candidates, key=lambda e: (sum_dev(e), e))
    return best_e, max_dev_opt, sum_dev(best_e)


def solve(records: Sequence[Record], pmin: int, pmax: int) -> Optional[Solution]:
    """Find the optimal joint calibration, or ``None`` if infeasible.

    ``records`` must be validated beforehand (strictly increasing counts,
    ``lo <= hi``, 1 <= pmin <= pmax).
    """
    n = len(records)
    counts = [r.count for r in records]
    lows = [r.lo for r in records]
    highs = [r.hi for r in records]
    sums = [lows[i] + highs[i] for i in range(n)]
    deltas = [counts[i] - counts[0] for i in range(n)]

    best_key: Optional[Tuple[int, int, int, int, int, int]] = None
    best_state: Optional[Tuple[int, int, int, int, List[int]]] = None

    for k in range(1, n):
        # Feasible T1 interval from the "before" segment alone: every lower
        # bound on t0 must not exceed every upper bound, i.e. for all i, j < k
        #   lo_i - d_i*T1 <= hi_j - d_j*T1   <=>   (d_j - d_i)*T1 <= hi_j - lo_i
        t1_lo, t1_hi = pmin, pmax
        for i in range(k):
            for j in range(k):
                dd = deltas[j] - deltas[i]
                rhs = highs[j] - lows[i]
                if dd > 0:
                    t1_hi = min(t1_hi, rhs // dd)
                elif dd < 0:
                    t1_lo = max(t1_lo, _ceil_div(rhs, dd))
        if t1_lo > t1_hi:
            continue

        d_boundary = deltas[k - 1]

        for t1 in range(t1_lo, t1_hi + 1):
            # t0 bounds contributed by the "before" segment.
            l1 = max(lows[i] - deltas[i] * t1 for i in range(k))
            r1 = min(highs[i] - deltas[i] * t1 for i in range(k))
            if l1 > r1:
                continue

            base = d_boundary * t1  # shared moment offset at record k-1

            # Feasible T2 interval given t1: exists t0 satisfying every bound.
            # "after" record i gives t0 bounds  lo_i - base - e_i*T2 <= t0
            # <= hi_i - base - e_i*T2  with e_i = c_i - c_{k-1} > 0.
            t2_lo, t2_hi = pmin, pmax
            for i in range(k, n):
                e_i = counts[i] - counts[k - 1]
                # (a) pairs inside the "after" segment:
                #     (e_j - e_i)*T2 <= hi_j - lo_i
                for j in range(k, n):
                    dd = (counts[j] - counts[k - 1]) - e_i
                    rhs = highs[j] - lows[i]
                    if dd > 0:
                        t2_hi = min(t2_hi, rhs // dd)
                    elif dd < 0:
                        t2_lo = max(t2_lo, _ceil_div(rhs, dd))
                for j in range(k):
                    # (b) after-lower <= before-upper:
                    #     e_i*T2 >= lo_i - base - hi_j + d_j*T1
                    t2_lo = max(
                        t2_lo, _ceil_div(lows[i] - base - highs[j] + deltas[j] * t1, e_i)
                    )
                    # (c) before-lower <= after-upper:
                    #     e_i*T2 <= hi_i - base - lo_j + d_j*T1
                    t2_hi = min(
                        t2_hi, (highs[i] - base - lows[j] + deltas[j] * t1) // e_i
                    )
            if t2_lo > t2_hi:
                continue

            offsets_pre = [deltas[i] * t1 for i in range(k)]
            a_pre = [sums[i] - 2 * offsets_pre[i] for i in range(k)]
            e_post = [counts[i] - counts[k - 1] for i in range(k, n)]

            for t2 in range(t2_lo, t2_hi + 1):
                offsets = offsets_pre + [base + e * t2 for e in e_post]
                lo_t0 = l1
                hi_t0 = r1
                a_values = list(a_pre)
                for idx in range(k, n):
                    off = offsets[idx]
                    lo_t0 = max(lo_t0, lows[idx] - off)
                    hi_t0 = min(hi_t0, highs[idx] - off)
                    a_values.append(sums[idx] - 2 * off)
                if lo_t0 > hi_t0:
                    continue

                e_best, max_dev, sum_dev = _best_doubled_start(a_values, lo_t0, hi_t0)
                t0 = e_best // 2
                key = (max_dev, sum_dev, k, t1, t2, t0)
                if best_key is None or key < best_key:
                    best_key = key
                    best_state = (k, t1, t2, t0, offsets)

    if best_state is None:
        return None

    k, t1, t2, t0, offsets = best_state
    predictions = tuple(t0 + off for off in offsets)
    doubled = tuple(abs(2 * predictions[i] - sums[i]) for i in range(n))
    return Solution(
        boundary=k,
        period_before=t1,
        period_after=t2,
        start_time=t0,
        predictions=predictions,
        doubled_deviations=doubled,
        max_doubled_deviation=max(doubled),
        total_doubled_deviation=sum(doubled),
    )
