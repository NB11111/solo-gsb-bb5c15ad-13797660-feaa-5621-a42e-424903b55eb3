"""响度增益规划核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

给定 1～300 个按时间排列且互不重叠的测量帧（每帧含 start_ms / end_ms /
loudness_lufs / true_peak_dbfs），为每帧选择 -12～12 的整数 dB 增益 g_i：

* 真峰约束：true_peak_dbfs_i + g_i <= 0（增益后不得削波）；
* 爬坡约束：|g_i - g_{i-1}| <= max_step_db（相邻帧增益变化不超步长）。

最小化总成本

    Σ_i |loudness_lufs_i + g_i - target_lufs|          （响度偏差）
  + 2 × Σ_{i>=1} |g_i - g_{i-1}|                       （增益变化代价），

同分取整条增益序列**字典序最小**者。

算法（后向 DP + 贪心重建），O(n·25²) 时间、O(n·25) 空间（n ≤ 300）：

* 每帧合法增益为连续区间 [-12, -true_peak_dbfs_i]，至多 25 个取值；
  max_step_db=0 时转移退化为只允许相等；
* bwd[i][g] = 第 i 帧取 g 时，后缀 [i, n) 能取得的最小成本（含帧 i 的
  响度偏差与其后全部相邻变化代价）；末帧仅含自身响度偏差；
* 重建时维护“当前前缀下的剩余最优成本”，在帧 i 的合法范围内由小到大
  枚举增益，取第一个满足爬坡且“到前帧变化代价 + bwd[i][g]”恰等于剩余
  成本者——每步取可行最小增益，归纳可得整条增益序列字典序最小；
  选定后剩余成本更新为 bwd[i][g] - 本帧偏差；
* 全 -12 始终满足真峰（true_peak_dbfs ≥ -80）与爬坡约束，故必有解。
"""

from .errors import DomainError

MIN_GAIN_DB = -12          # 每帧增益下限
MAX_GAIN_DB = 12           # 每帧增益上限
MAX_FRAMES = 300           # 测量帧数上限
CHANGE_COST_FACTOR = 2     # 相邻增益变化的代价系数
INF = 10 ** 18             # 不可达状态（某增益无法扩展到后续帧）


def _fail(code, message, **extra):
    raise DomainError(code, message, **extra)


def validate_frames(frames):
    """校验帧序列：非空、帧内时间严格为正、相邻帧不倒序、不重叠（端点相等允许）。

    字段取值范围（响度/峰值 -80～0、帧数上限等）由 Pydantic 模型负责。
    """
    if not frames:
        _fail("empty_frames", "frames 不能为空")

    prev = None
    for idx, f in enumerate(frames):
        if f.start_ms < 0:
            _fail(
                "invalid_time_range",
                f"第 {idx} 帧的 start_ms({f.start_ms}) 为负，时间必须非负",
                frame_index=idx,
            )
        if f.end_ms <= f.start_ms:
            _fail(
                "invalid_time_range",
                f"第 {idx} 帧的 end_ms({f.end_ms}) 不晚于 start_ms({f.start_ms})",
                frame_index=idx,
            )
        if prev is not None:
            if f.start_ms < prev.start_ms:
                _fail(
                    "time_out_of_order",
                    f"第 {idx} 帧的 start_ms({f.start_ms}) 早于前一帧的 "
                    f"start_ms({prev.start_ms})，输入必须按时间排列",
                    frame_index=idx,
                )
            if f.start_ms < prev.end_ms:
                _fail(
                    "overlapping_time_range",
                    f"第 {idx} 帧的 start_ms({f.start_ms}) 早于前一帧的 "
                    f"end_ms({prev.end_ms})，相邻帧时间不允许重叠",
                    frame_index=idx,
                )
        prev = f


def plan_gains(frames, target_lufs, max_step_db):
    """为每帧选择增益，返回 (gains, total_cost)。

    参数 frames 中每个对象需具备属性：start_ms / end_ms /
    loudness_lufs / true_peak_dbfs；target_lufs 与 max_step_db 为已通过
    Pydantic 范围校验的整数。

    gains 为逐帧整数 dB 增益列表（与 frames 同序），total_cost 为最小
    总成本：Σ 响度绝对偏差 + 2 × Σ 相邻增益绝对变化量。
    """
    validate_frames(frames)
    n = len(frames)

    # 每帧合法增益上界：true_peak + gain <= 0 => gain <= -true_peak
    upper = [min(MAX_GAIN_DB, -f.true_peak_dbfs) for f in frames]
    deviation = [
        {g: abs(f.loudness_lufs + g - target_lufs) for g in range(MIN_GAIN_DB, upper[i] + 1)}
        for i, f in enumerate(frames)
    ]

    # ---- 后向 DP：bwd[i][g] = 帧 i 取 g 时后缀 [i, n) 的最小成本 ----
    # 某些 g 因爬坡后下一帧真峰上界内无可达增益而不可行（如 g=12、
    # max_step_db=0 而下一帧仅允许 -12），记 INF。
    bwd = [None] * n
    bwd[n - 1] = dict(deviation[n - 1])
    for i in range(n - 2, -1, -1):
        row = {}
        suffix = bwd[i + 1]
        for g, dev in deviation[i].items():
            lo = max(MIN_GAIN_DB, g - max_step_db)
            hi = min(upper[i + 1], g + max_step_db)
            best = INF
            for h in range(lo, hi + 1):
                if suffix[h] >= INF:
                    continue
                candidate = CHANGE_COST_FACTOR * abs(g - h) + suffix[h]
                if candidate < best:
                    best = candidate
            row[g] = INF if best >= INF else dev + best
        bwd[i] = row

    # ---- 贪心重建：每步在仍能达成最优总成本的增益中取最小者 ----
    # 不变量：进入帧 i 时 remaining 等于“边 (i-1,i) 的变化代价（帧 0 为 0）
    # + 后缀 [i, n) 成本”在当前已选前缀下的目标值。帧 0 即
    # min_g bwd[0][g]；候选 g 需满足爬坡且 change(g) + bwd[i][g] == remaining。
    # 选定 g 后，下一轮（帧 i+1）的目标为
    # bwd[i][g] - dev_i(g) = min_h (2|g-h| + bwd[i+1][h]) 中沿 g 的最优边值。
    gains = []
    remaining = min(cost for cost in bwd[0].values() if cost < INF)
    total_cost = remaining
    prev_gain = None
    for i in range(n):
        for g in range(MIN_GAIN_DB, upper[i] + 1):
            if prev_gain is not None and abs(g - prev_gain) > max_step_db:
                continue
            change = 0 if prev_gain is None else CHANGE_COST_FACTOR * abs(g - prev_gain)
            if bwd[i][g] < INF and change + bwd[i][g] == remaining:
                chosen = g
                break
        gains.append(chosen)
        remaining = bwd[i][chosen] - deviation[i][chosen]
        prev_gain = chosen

    return gains, total_cost
