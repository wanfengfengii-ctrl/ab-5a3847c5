"""Unit tests for the exact integer solver."""

import random
import unittest

from app.solver import Record, Solution, solve


def brute_force(records, pmin, pmax):
    """Reference implementation: plain enumeration of every (k, T1, T2, t0).

    Returns the best key tuple and the associated solution tuple, or None.
    """
    n = len(records)
    counts = [r.count for r in records]
    lows = [r.lo for r in records]
    highs = [r.hi for r in records]
    sums = [lows[i] + highs[i] for i in range(n)]
    deltas = [counts[i] - counts[0] for i in range(n)]

    best_key = None
    best_solution = None
    for k in range(1, n):
        for t1 in range(pmin, pmax + 1):
            offsets_pre = [deltas[i] * t1 for i in range(k)]
            base = deltas[k - 1] * t1
            for t2 in range(pmin, pmax + 1):
                offsets = offsets_pre + [
                    base + (counts[i] - counts[k - 1]) * t2 for i in range(k, n)
                ]
                lo_t0 = max(lows[i] - offsets[i] for i in range(n))
                hi_t0 = min(highs[i] - offsets[i] for i in range(n))
                if lo_t0 > hi_t0:
                    continue
                for t0 in range(lo_t0, hi_t0 + 1):
                    deviations = [
                        abs(2 * (t0 + offsets[i]) - sums[i]) for i in range(n)
                    ]
                    key = (max(deviations), sum(deviations), k, t1, t2, t0)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_solution = (k, t1, t2, t0, tuple(t0 + o for o in offsets))
    if best_solution is None:
        return None
    return best_key, best_solution


def make_records(counts, windows):
    return [Record(count=c, lo=w[0], hi=w[1]) for c, w in zip(counts, windows)]


def ground_truth_times(counts, k, t1, t2, t0):
    """Continuous segmented ground truth used to build synthetic cases."""
    times = []
    for i, c in enumerate(counts):
        if i < k:
            times.append(t0 + (c - counts[0]) * t1)
        else:
            times.append(t0 + (counts[k - 1] - counts[0]) * t1 + (c - counts[k - 1]) * t2)
    return times


class ExactRecoveryTests(unittest.TestCase):
    def test_exact_zero_width_windows_recovers_unique_model(self):
        counts = list(range(0, 10))
        k, t1, t2, t0 = 5, 7, 11, 100
        times = ground_truth_times(counts, k, t1, t2, t0)
        records = make_records(counts, [(t, t) for t in times])
        sol = solve(records, 1, 20)
        self.assertIsNotNone(sol)
        self.assertEqual(sol.boundary, k)
        self.assertEqual(sol.period_before, t1)
        self.assertEqual(sol.period_after, t2)
        self.assertEqual(sol.start_time, t0)
        self.assertEqual(sol.max_doubled_deviation, 0)
        self.assertEqual(sol.total_doubled_deviation, 0)
        self.assertEqual(list(sol.predictions), times)

    def test_continuity_at_switch_boundary(self):
        counts = [100 + 5 * i for i in range(12)]
        k, t1, t2, t0 = 7, 9, 13, 4000
        times = ground_truth_times(counts, k, t1, t2, t0)
        records = make_records(counts, [(t - 2, t + 3) for t in times])
        sol = solve(records, 5, 20)
        self.assertIsNotNone(sol)
        # Segment 1 accumulates with period_before up to record k-1 ...
        for i in range(sol.boundary - 1):
            step = sol.predictions[i + 1] - sol.predictions[i]
            self.assertEqual(step, (counts[i + 1] - counts[i]) * sol.period_before)
        # ... the moment at record k-1 is shared, then period_after takes over.
        for i in range(sol.boundary - 1, len(counts) - 1):
            step = sol.predictions[i + 1] - sol.predictions[i]
            self.assertEqual(step, (counts[i + 1] - counts[i]) * sol.period_after)
        shared = sol.predictions[sol.boundary - 1]
        self.assertEqual(
            shared,
            sol.start_time + (counts[sol.boundary - 1] - counts[0]) * sol.period_before,
        )
        for i, record in enumerate(records):
            self.assertTrue(record.lo <= sol.predictions[i] <= record.hi)

    def test_half_integer_midpoints_use_exact_arithmetic(self):
        # Windows with odd lo+hi have half-integer midpoints; deviations of
        # exactly 0.5 must be representable and comparable without floats.
        counts = list(range(0, 8))
        times = ground_truth_times(counts, 4, 3, 5, 10)
        windows = [(t, t + 1) for t in times]  # midpoint = t + 0.5
        records = make_records(counts, windows)
        sol = solve(records, 1, 10)
        self.assertIsNotNone(sol)
        self.assertEqual(sol.max_doubled_deviation, 1)  # exactly 0.5
        self.assertEqual(sol.total_doubled_deviation, len(counts))

    def test_infeasible_when_windows_contradict(self):
        counts = list(range(0, 8))
        windows = [(0, 0) if i % 2 == 0 else (100000, 100000) for i in range(8)]
        records = make_records(counts, windows)
        self.assertIsNone(solve(records, 1, 10))

    def test_lexicographic_max_before_sum(self):
        # The solver must prefer a smaller maximum deviation even at the cost
        # of a larger total; the differential test below covers this too, but
        # pin the behaviour with a deterministic regression case.
        counts = list(range(0, 8))
        times = ground_truth_times(counts, 4, 4, 6, 50)
        windows = [(t - 1, t + 1) for t in times]
        windows[3] = (times[3] - 5, times[3] + 5)  # one wide window
        records = make_records(counts, windows)
        sol = solve(records, 1, 12)
        self.assertIsNotNone(sol)
        key, expected = brute_force(records, 1, 12)
        self.assertEqual(
            (sol.max_doubled_deviation, sol.total_doubled_deviation,
             sol.boundary, sol.period_before, sol.period_after, sol.start_time),
            key,
        )
        self.assertEqual(tuple(sol.predictions), expected[4])


class DifferentialTests(unittest.TestCase):
    """The pruned enumerator must agree with full brute force everywhere."""

    def _random_case(self, rng):
        n = rng.randint(8, 12)
        counts = [rng.randint(-30, 30)]
        for _ in range(n - 1):
            counts.append(counts[-1] + rng.randint(1, 5))
        k = rng.randint(1, n - 1)
        t1 = rng.randint(1, 6)
        t2 = rng.randint(1, 6)
        t0 = rng.randint(-60, 60)
        times = ground_truth_times(counts, k, t1, t2, t0)
        windows = []
        for t in times:
            slack_lo = rng.randint(0, 3)
            slack_hi = rng.randint(0, 3)
            windows.append((t - slack_lo, t + slack_hi))
        pmin = rng.randint(1, 3)
        pmax = rng.randint(6, 9)
        return make_records(counts, windows), pmin, pmax

    def _random_noisy_case(self, rng):
        n = rng.randint(8, 12)
        counts = [rng.randint(-30, 30)]
        for _ in range(n - 1):
            counts.append(counts[-1] + rng.randint(1, 5))
        windows = []
        cursor = rng.randint(-100, 100)
        for _ in range(n):
            cursor += rng.randint(-20, 40)
            windows.append((cursor - rng.randint(0, 6), cursor + rng.randint(0, 6)))
        pmin = rng.randint(1, 4)
        pmax = rng.randint(4, 8)
        return make_records(counts, windows), pmin, pmax

    def _assert_agree(self, records, pmin, pmax):
        expected = brute_force(records, pmin, pmax)
        sol = solve(records, pmin, pmax)
        if expected is None:
            self.assertIsNone(sol)
            return
        key, (k, t1, t2, t0, predictions) = expected
        self.assertIsNotNone(sol)
        self.assertEqual(
            (sol.max_doubled_deviation, sol.total_doubled_deviation,
             sol.boundary, sol.period_before, sol.period_after, sol.start_time),
            key,
        )
        self.assertEqual(tuple(sol.predictions), predictions)
        self.assertEqual(sol.boundary, k)
        self.assertEqual(sol.period_before, t1)
        self.assertEqual(sol.period_after, t2)
        self.assertEqual(sol.start_time, t0)

    def test_differential_on_truth_perturbed_cases(self):
        rng = random.Random(20261001)
        for _ in range(250):
            records, pmin, pmax = self._random_case(rng)
            self._assert_agree(records, pmin, pmax)

    def test_differential_on_noisy_cases(self):
        rng = random.Random(987654321)
        for _ in range(250):
            records, pmin, pmax = self._random_noisy_case(rng)
            self._assert_agree(records, pmin, pmax)

    def test_maximum_record_count(self):
        rng = random.Random(42)
        n = 24
        counts = [1000]
        for _ in range(n - 1):
            counts.append(counts[-1] + rng.randint(2, 7))
        k, t1, t2, t0 = 13, 8, 14, 9000
        times = ground_truth_times(counts, k, t1, t2, t0)
        records = make_records(counts, [(t - 3, t + 4) for t in times])
        sol = solve(records, 4, 30)
        self.assertIsNotNone(sol)
        for i, record in enumerate(records):
            self.assertTrue(record.lo <= sol.predictions[i] <= record.hi)


if __name__ == "__main__":
    unittest.main()
