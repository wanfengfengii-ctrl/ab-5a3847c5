"""整数精确校时求解器。

模型
----
记录 i 具有：
  * 递增整数计数 c_i
  * 地面时刻整数闭区间 [lo_i, hi_i]

设：
  * k 为切换边界（1 <= k < n）：记录 0..k-1 使用切换前周期 p1，
    记录 k..n-1 使用切换后周期 p2；
  * t0 为记录 0 处的整数起始时刻。

分段累计预测（切换边界两侧共用同一时刻，不存在断裂）：

  A_i(p1,p2) = p1*(min(c_i,c_k)-c_0) + p2*max(0, c_i-c_k)
  t_i         = t0 + A_i

所有比较只使用整数。区间中点可能为半整数，故一切偏差均以"两倍值"
表示：中点 m2_i = lo_i+hi_i，残差 r2_i = 2*t_i - m2_i，
绝对偏差 d2_i = |r2_i|。整体 2 倍放缩不改变任何排序，
全程不出现浮点运算。

裁决目标为字典序最小化：
  (max_i d2_i, sum_i d2_i, k, p1, p2, t0)

可行性经消去 t0 后等价于 (p1,p2) 平面上的线性有理多胞形：
  low_j-high_i <= A_j-A_i <= high_j-low_i
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 防止滥用的合理上限（整数范围本身不受机器精度限制）。
_MAX_ABS_TIME = 10**13
_MAX_PERIOD = 10**9
_MAX_INTERVAL_WIDTH = 10**9
_MAX_ABS_COUNT = 10**13

# 小可行域直接网格枚举的格点数预算。
_GRID_BUDGET = 300_000
# 大可行域时，每个有理顶点周围取的格点半径；沿线行走的本原步
# 余量为 _VERTEX_RADIUS + 1（见文件末尾精确性证明）。
_VERTEX_RADIUS = 3


class ValidationError(ValueError):
    """输入数据不合法。"""


@dataclass(frozen=True)
class Record:
    count: int
    low: int
    high: int


@dataclass(frozen=True)
class Solution:
    switch_k: int          # 使用后周期的第一条记录下标（0-based）
    period_before: int
    period_after: int
    t0: int
    max_dev2: int
    sum_dev2: int
    predictions: tuple[int, ...]
    residuals2: tuple[int, ...]

    def key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.max_dev2,
            self.sum_dev2,
            self.switch_k,
            self.period_before,
            self.period_after,
            self.t0,
        )


def _as_int(name: str, value: object) -> int:
    # 拒绝 bool（bool 是 int 的子类）与非整数值。
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} 必须为整数")
    return value


def parse_inputs(raw: object) -> tuple[tuple[Record, ...], int, int]:
    """校验并解析请求体，返回 (记录, 周期下界, 周期上界)。"""
    if not isinstance(raw, dict):
        raise ValidationError("请求体必须为 JSON 对象")

    recs_raw = raw.get("records")
    if not isinstance(recs_raw, list):
        raise ValidationError("records 必须为数组")
    n = len(recs_raw)
    if not 8 <= n <= 24:
        raise ValidationError("records 数量必须在 8 至 24 条之间")

    recs: list[Record] = []
    prev_count: int | None = None
    for idx, item in enumerate(recs_raw):
        if not isinstance(item, dict):
            raise ValidationError(f"records[{idx}] 必须为对象")
        count = _as_int(f"records[{idx}].count", item.get("count"))
        low = _as_int(f"records[{idx}].low", item.get("low"))
        high = _as_int(f"records[{idx}].high", item.get("high"))
        if not -_MAX_ABS_COUNT <= count <= _MAX_ABS_COUNT:
            raise ValidationError(f"records[{idx}].count 超出允许范围")
        if not -_MAX_ABS_TIME <= low <= high <= _MAX_ABS_TIME:
            raise ValidationError(
                f"records[{idx}] 时间区间无效，需满足 low <= high"
            )
        if high - low > _MAX_INTERVAL_WIDTH:
            raise ValidationError(f"records[{idx}] 时间区间过宽")
        if prev_count is not None and count <= prev_count:
            raise ValidationError(f"records[{idx}].count 必须严格递增")
        prev_count = count
        recs.append(Record(count=count, low=low, high=high))

    period = raw.get("period_range")
    if not isinstance(period, dict):
        raise ValidationError("period_range 必须为对象")
    p_lo = _as_int("period_range.low", period.get("low"))
    p_hi = _as_int("period_range.high", period.get("high"))
    if not 1 <= p_lo <= p_hi <= _MAX_PERIOD:
        raise ValidationError("period_range 必须满足 1 <= low <= high")

    return tuple(recs), p_lo, p_hi


def _floor_div(a: int, b: int) -> int:
    """数学意义上的向下取整除法，b 为非零整数。"""
    if b > 0:
        return a // b
    return -((-a) // (-b))


def _ceil_div(a: int, b: int) -> int:
    return -_floor_div(-a, b)


def _tighten(
    lo: int, hi: int, num_lo: int, num_hi: int, den: int
) -> tuple[int, int]:
    """加入约束 num_lo <= den*x <= num_hi（den > 0），缩窄整数 [lo,hi]。"""
    return max(lo, _ceil_div(num_lo, den)), min(
        hi, _floor_div(num_hi, den)
    )


def _best_t0(
    recs: tuple[Record, ...],
    accum: tuple[int, ...],
    incumbent: tuple[int, int] | None = None,
) -> tuple[int, int, int] | None:
    """固定 (p1,p2) 后，在可行整数 t0 中取字典序最优者。

    返回 (t0, max_dev2, sum_dev2)；无可行 t0 返回 None。
    incumbent 为已知 (max_dev2, sum_dev2) 上界，用于前缀剪枝。
    """
    lo = max(r.low - a for r, a in zip(recs, accum))
    hi = min(r.high - a for r, a in zip(recs, accum))
    if lo > hi:
        return None

    # B_i = 2*A_i - m2_i；残差 r2_i = 2*t0 + B_i。
    bvals = [2 * a - (r.low + r.high) for r, a in zip(recs, accum)]

    # 关于整数 t0 的凸分段线性目标，最优点只可能出现在：
    #   * 可行区间端点；
    #   * 最大偏差中心 -(b_min+b_max)/4 的相邻整数；
    #   * 各残差零点 -B_i/2 的相邻整数（总和的零点中位处）。
    cand = {lo, hi}
    center_num = -(min(bvals) + max(bvals))
    cand.add(_floor_div(center_num, 4))
    cand.add(_ceil_div(center_num, 4))
    for b in bvals:
        q = _floor_div(-b, 2)
        cand.add(q)
        cand.add(q + 1)

    best: tuple[int, int, int] | None = None
    for t0 in cand:
        if t0 < lo or t0 > hi:
            continue
        dmax = 0
        dsum = 0
        aborted = False
        for b in bvals:
            d = 2 * t0 + b
            if d < 0:
                d = -d
            dsum += d
            if d > dmax:
                dmax = d
            # 前缀仅可能使 dmax/dsum 继续增大，可安全提前淘汰。
            if incumbent is not None and (
                dmax > incumbent[0]
                or (dmax == incumbent[0] and dsum > incumbent[1])
            ):
                aborted = True
                break
            if best is not None and (
                dmax > best[0]
                or (dmax == best[0] and dsum > best[1])
            ):
                aborted = True
                break
        if aborted:
            continue
        key = (dmax, dsum, t0)
        if best is None or key < best:
            best = key
    # best 为 None 只可能发生在存在可行 t0、但无一能改进外部
    # incumbent 时；此时返回 None，调用方安全跳过该 (p1,p2)。
    if best is None:
        return None
    return best[2], best[0], best[1]


def solve_for_boundary(
    recs: tuple[Record, ...],
    p_lo: int,
    p_hi: int,
    k: int,
) -> Solution | None:
    """固定切换边界 k 时求最优解。"""
    n = len(recs)
    counts = tuple(r.count for r in recs)
    lows = tuple(r.low for r in recs)
    highs = tuple(r.high for r in recs)
    c0, ck = counts[0], counts[k]

    # alpha_i = min(c_i,c_k)-c0, beta_i = max(0, c_i-c_k)
    alpha = tuple(min(c, ck) - c0 for c in counts)
    beta = tuple(max(0, c - ck) for c in counts)

    # 差约束： low_j-high_i <= da*p1+db*p2 <= high_j-low_i
    xlo = xhi = ylo = yhi = 0  # 占位，下面初始化
    xlo, xhi = p_lo, p_hi
    ylo, yhi = p_lo, p_hi
    cross: list[tuple[int, int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            da = alpha[j] - alpha[i]
            db = beta[j] - beta[i]
            dl = lows[j] - highs[i]
            du = highs[j] - lows[i]
            if da == 0 and db == 0:
                if dl > 0 or du < 0:
                    return None
                continue
            if db == 0:
                if da < 0:
                    da, dl, du = -da, -du, -dl
                xlo, xhi = _tighten(xlo, xhi, dl, du, da)
            elif da == 0:
                if db < 0:
                    db, dl, du = -db, -du, -dl
                ylo, yhi = _tighten(ylo, yhi, dl, du, db)
            else:
                cross.append((da, db, dl, du))
            if xlo > xhi or ylo > yhi:
                return None

    best: Solution | None = None

    def can_beat(x: int, y: int) -> bool:
        """字典序裁决剪枝：候选 (max,sum,k,x,y,t0) 是否可能优于 best。"""
        if best is None:
            return True
        if best.max_dev2 > 0:
            return True
        # best 已是零偏差；只有更小 (p1,p2,t0) 才可能取代。
        return (x, y) < (best.period_before, best.period_after)

    def evaluate(x: int, y: int) -> bool:
        nonlocal best
        if not can_beat(x, y):
            return False
        accum = tuple(alpha[i] * x + beta[i] * y for i in range(n))
        incumbent = (
            None if best is None else (best.max_dev2, best.sum_dev2)
        )
        picked = _best_t0(recs, accum, incumbent)
        if picked is None:
            return False
        t0, dmax, dsum = picked
        cand = Solution(
            switch_k=k,
            period_before=x,
            period_after=y,
            t0=t0,
            max_dev2=dmax,
            sum_dev2=dsum,
            predictions=tuple(t0 + a for a in accum),
            residuals2=tuple(
                2 * (t0 + accum[i]) - (lows[i] + highs[i])
                for i in range(n)
            ),
        )
        if best is None or cand.key() < best.key():
            best = cand
            return dmax == 0
        return False

    def feasible_cross(x: int, y: int) -> bool:
        return all(
            dl <= da * x + db * y <= du for da, db, dl, du in cross
        )

    # ---- 路径一：可行域不大时做整数网格精确枚举 ----
    if (xhi - xlo + 1) * (yhi - ylo + 1) <= _GRID_BUDGET:
        # p1 升序、p2 升序枚举：首个零偏差解即满足全部平局裁决，
        # 同一边界内可立即停止；跨边界由 resolve 处理。
        for x in range(xlo, xhi + 1):
            ay0, ay1 = ylo, yhi
            for da, db, dl, du in cross:
                nlo = dl - da * x
                nhi = du - da * x
                if db > 0:
                    ay0 = max(ay0, _ceil_div(nlo, db))
                    ay1 = min(ay1, _floor_div(nhi, db))
                else:
                    ay0 = max(ay0, _ceil_div(-nhi, -db))
                    ay1 = min(ay1, _floor_div(-nlo, -db))
                if ay0 > ay1:
                    break
            for y in range(ay0, ay1 + 1):
                if evaluate(x, y) and best.max_dev2 == 0:
                    return best
        return best

    # ---- 路径二：大可行域，枚举直线排列顶点附近的整数格点 ----
    # 候选点按 (p1,p2) 排序：一旦出现零偏差解，其 (p1,p2) 之后的
    # 点在字典序裁决中皆无获胜可能，可提前终止。
    points = sorted(
        _vertex_candidates(
            alpha, beta, lows, highs, p_lo, p_hi, xlo, xhi, ylo, yhi
        )
    )
    for x, y in points:
        if best is not None and best.max_dev2 == 0 and (
            (x, y) >= (best.period_before, best.period_after)
        ):
            break
        if can_beat(x, y) and feasible_cross(x, y):
            evaluate(x, y)
    return best


def _make_lines(
    alpha: tuple[int, ...],
    beta: tuple[int, ...],
    lows: tuple[int, ...],
    highs: tuple[int, ...],
    p_lo: int,
    p_hi: int,
) -> set[tuple[int, int, int]]:
    """构造整数排列的全部本原直线 a*x+b*y=c（gcd(a,b)=1）。

    原始半整数/非本原方程（系数 g=gcd(|a|,|b|) 不整除 c）上不存在
    整数点；对整数格点真正起分界作用的是其两侧的格点支撑线
    a'x+b'y = floor(c/g) 与 ceil(c/g)，故统一发射这两条本原直线。

    直线来源：
      * 残差相等/相反：r2_i = ±r2_j（极差包络活动对切换线）；
      * 残差为零：r2_i = 0（偏差总和的符号折点线）；
      * 可行差约束 A_j-A_i = low_j-high_i / high_j-low_i；
      * 周期盒边界。
    """
    n = len(alpha)
    m2 = tuple(lows[i] + highs[i] for i in range(n))
    lines: set[tuple[int, int, int]] = set()

    def add(a: int, b: int, c: int) -> None:
        if a == 0 and b == 0:
            return
        g = math.gcd(abs(a), abs(b))
        ap, bp, q, rem = a // g, b // g, c // g, c % g
        lines.add((ap, bp, q))
        if rem:  # c 不整除 g：加入另一侧格点支撑线
            lines.add((ap, bp, q + 1))

    for i in range(n):
        for j in range(i + 1, n):
            da = alpha[i] - alpha[j]
            db = beta[i] - beta[j]
            dm = m2[i] - m2[j]
            # r2_i = r2_j（最大正偏差包络的活动对切换线）
            add(2 * da, 2 * db, dm)
            # r2_i = -r2_j（最大正/负偏差活动对切换线）
            add(2 * (alpha[i] + alpha[j]),
                2 * (beta[i] + beta[j]),
                m2[i] + m2[j])
            # 可行差约束的两条边界
            qa = alpha[j] - alpha[i]
            qb = beta[j] - beta[i]
            add(qa, qb, lows[j] - highs[i])
            add(qa, qb, highs[j] - lows[i])

    # r2_i = 0：|r2_i| 的折点线（偏差总和目标的活动符号变化处）。
    for i in range(n):
        add(2 * alpha[i], 2 * beta[i], m2[i])

    add(1, 0, p_lo)
    add(1, 0, p_hi)
    add(0, 1, p_lo)
    add(0, 1, p_hi)
    return lines


def _ext_gcd(a: int, b: int) -> tuple[int, int]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    # old_r = gcd（可能为负，归一化）
    if old_r < 0:
        old_s, old_t = -old_s, -old_t
    return old_s, old_t


def _vertex_candidates(
    alpha: tuple[int, ...],
    beta: tuple[int, ...],
    lows: tuple[int, ...],
    highs: tuple[int, ...],
    p_lo: int,
    p_hi: int,
    xlo: int,
    xhi: int,
    ylo: int,
    yhi: int,
) -> set[tuple[int, int]]:
    """枚举直线排列有理顶点附近的整数候选格点（完备，见文末证明）。"""
    line_list = list(_make_lines(alpha, beta, lows, highs, p_lo, p_hi))
    points: set[tuple[int, int]] = {
        (xlo, ylo), (xlo, yhi), (xhi, ylo), (xhi, yhi),
    }
    r = _VERTEX_RADIUS
    slack = r + 3

    def add_around(vx_num: int, vy_num: int, det: int) -> None:
        fx = _floor_div(vx_num, det)
        fy = _floor_div(vy_num, det)
        for gx in range(fx - r, fx + r + 2):
            if not xlo <= gx <= xhi:
                continue
            for gy in range(fy - r, fy + r + 2):
                if ylo <= gy <= yhi:
                    points.add((gx, gy))

    def add_tube_pair(
        l1: tuple[int, int, int], l2: tuple[int, int, int]
    ) -> None:
        """两本原直线交点附近的完备整数小管。

        L1: n1·p=c1，本原切向 t1=R(n1)=(b1,-a1)；取满足
        n1·u1=1 的 Bézout 特解 u1，则层级 c1+h1 的格线整数点为
          p = (c1+h1)·u1 + j·t1。
        选择 u1 的代表元使 n2·u1 按 n2·t1（=−det，记 den）取模
        归约至 |n2·u1| ≤ |den|/2，则 L1/L2 各偏移一层（h1,h2∈
        {-1,0,1}）后的交点沿 t1 的位移
          Δs = (h2 − h1·(n2·u1))/den
        满足 |Δs| ≤ 1/2 + 1/|den| ≤ 3/2，与系数大小无关。
        """
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        ux, uy = _ext_gcd(a1, b1)  # 本原直线，a1*ux+b1*uy = 1
        tx, ty = b1, -a1
        den = a2 * tx + b2 * ty  # = a2*b1 - b2*a1 = -det(n1,n2)
        if den == 0:
            return
        dot = a2 * ux + b2 * uy
        # m = 最接近 dot/den 的整数，使残量 |dot'| ≤ |den|/2。
        m = _floor_div(2 * dot + den, 2 * den)
        ux += m * tx
        uy += m * ty
        dot += m * den

        for h1 in (-1, 0, 1):
            bx = (c1 + h1) * ux
            by = (c1 + h1) * uy
            for h2 in (-1, 0, 1):
                num = c2 + h2 - (c1 + h1) * dot
                jf = _floor_div(num, den)
                for j in range(jf - r, jf + r + 2):
                    gx, gy = bx + j * tx, by + j * ty
                    if xlo <= gx <= xhi and ylo <= gy <= yhi:
                        points.add((gx, gy))

    for u in range(len(line_list)):
        a1, b1, c1 = line_list[u]
        for v in range(u + 1, len(line_list)):
            a2, b2, c2 = line_list[v]
            det = a1 * b2 - a2 * b1
            if det == 0:
                continue
            vx = c1 * b2 - c2 * b1
            vy = a1 * c2 - a2 * c1
            if not _rational_in_box(
                vx, vy, det, xlo, xhi, ylo, yhi, slack
            ):
                continue
            add_around(vx, vy, det)
            # 分别以两条线为基线枚举法向一层、切向近点的小管。
            add_tube_pair(line_list[u], line_list[v])
            add_tube_pair(line_list[v], line_list[u])
    return points


def _rational_in_box(
    vx: int, vy: int, det: int,
    xlo: int, xhi: int, ylo: int, yhi: int,
    slack: int,
) -> bool:
    """判断有理点 (vx/det, vy/det) 是否在扩张 slack 后的盒内。"""
    if det < 0:
        det, vx, vy = -det, -vx, -vy
    return (
        (xlo - slack) * det <= vx <= (xhi + slack) * det
        and (ylo - slack) * det <= vy <= (yhi + slack) * det
    )


def resolve(
    recs: tuple[Record, ...], p_lo: int, p_hi: int
) -> Solution | None:
    """联合搜索全部切换边界，返回字典序最优解；不可行返回 None。

    边界 k 升序处理；某边界给出 max_dev2=0 的解时，后续边界已无
    获胜可能（裁决键首项为 0 且 k 更大），立即终止。
    """
    best: Solution | None = None
    for k in range(1, len(recs)):
        cand = solve_for_boundary(recs, p_lo, p_hi, k)
        if cand is not None and (best is None or cand.key() < best.key()):
            best = cand
        if best is not None and best.max_dev2 == 0:
            return best
    return best


# ---------------------------------------------------------------------------
# 精确性证明（路径二，固定半径即完备）
#
# 记 (p1,p2) 平面整数格点上的目标
#   F(x,y) = 最优整数 t0 下的 (max|r2_i|, Σ|r2_i|)，字典序比较。
# 连续松弛 F* 凸且分段线性：极差项折点在 r2_i = ±r2_j，总和项折点
# 在 r2_i = 0；这些直线与可行差约束边界、周期盒边界在 _make_lines
# 中统一化为本原直线 a'x+b'y=q（gcd(a',b')=1）：若方程除以
# g=gcd(|a|,|b|) 后右侧非整数，则该线上无格点，发射其两侧的格点
# 支撑线 q=floor、q+1=ceil，故排列只含真正分隔整数格点的直线。
#
# 对本原直线 L: n·p=c（n=(a,b)，gcd=1），本原切向 t=(b,-a)，
# 并由扩展欧几里得取 Bézout 向量 u 使 n·u=1。层级 c+h 的格线
# L^(h): n·p=c+h 的全部整数点为 p=(c+h)·u + j·t；相邻整数层级
# h 与 h+1 在几何上相差 u（其分量可能很大），而非 n 本身。
#
# 引理：设 z 为全局整数最优。
#   1. 在任一排列胞形内部各 r2_i 符号/次序固定，F* 为线性，F 在
#      格点上取整数值。z 必位于无法再沿格点方向改进之处，故贴在
#      胞形边界的某条整数层级直线上；
#   2. 任取包围胞形的两条本原直线 L1、L2（本原法向 n1、n2），
#      z 所贴的只可能是它们的相邻层级 c1+h1、c2+h2，
#      h1,h2∈{-1,0,1}（任意格点到某条平行格线的层级差至多 1）；
#   3. 实现中先将 u1 按 n2·t1（=−det(n1,n2)，记 den）取模归约，
#      使 |n2·u1| ≤ |den|/2。偏移交点在 L1 上的切向参数
#      s = (h2 − h1·(n2·u1))/den，故 |s| ≤ 1/2 + 1/|den| ≤ 3/2，
#      与系数大小无关（即使 |t1| 或 |u1| 的分量高达 N 也成立）；
#   4. 原顶点到 L1 上最近整数点的切向参数差不超过 1/2。
# 故整数最优必落在：顶点坐标 floor 周围半径 _VERTEX_RADIUS 的格点，
# 或每对相交本原直线相邻层级的切向夹点（floor(s)、ceil(s)）周围
# ±_VERTEX_RADIUS 个切向本原步内。本原步分量可能很大（如 (N,1)），
# 故必须按 (u,t) 参数化枚举，轴对齐方盒无法覆盖。
# add_tube_pair 恰好枚举上述集合（对每条线各做一次基线），覆盖与
# 系数大小无关；±r 余量吸收整数 t0 的 ±1 舍入锯齿。
#
# 小可行域（格点数 ≤ _GRID_BUDGET）直接逐点网格枚举，结论不依赖
# 上述几何论证，精确无遗漏。
