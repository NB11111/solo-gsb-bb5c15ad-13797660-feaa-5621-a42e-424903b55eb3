"""响度增益规划核心逻辑单元测试 + 独立前向 DP 对拍（纯标准库）。

独立对拍：前向 DP 直接为每个末态增益保留 (最小成本, 字典序最小序列元组)，
与被测实现（后向 DP + 贪心重建）的递推方向、数据结构完全不同，互验最优
成本与字典序裁决。
"""

import random
import unittest
from types import SimpleNamespace

from app.errors import DomainError
from app.gain_plan import (
    CHANGE_COST_FACTOR,
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    plan_gains,
)


def F(start=0, end=1000, loudness=-20, peak=-2, start_ms=None, end_ms=None):
    if start_ms is not None:
        start = start_ms
    if end_ms is not None:
        end = end_ms
    return SimpleNamespace(
        start_ms=start,
        end_ms=end,
        loudness_lufs=loudness,
        true_peak_dbfs=peak,
    )


def independent_solve(frames, target_lufs, max_step_db):
    """前向 DP：每帧每个增益保留 (成本, 序列元组)，按 (成本, 序列) 取最优。"""
    upper = [min(MAX_GAIN_DB, -f.true_peak_dbfs) for f in frames]
    states = {}
    for g in range(MIN_GAIN_DB, upper[0] + 1):
        states[g] = (abs(frames[0].loudness_lufs + g - target_lufs), (g,))
    for i in range(1, len(frames)):
        nxt = {}
        for prev_gain, (prev_cost, seq) in states.items():
            lo = max(MIN_GAIN_DB, prev_gain - max_step_db)
            hi = min(upper[i], prev_gain + max_step_db)
            for g in range(lo, hi + 1):
                cost = (
                    prev_cost
                    + CHANGE_COST_FACTOR * abs(g - prev_gain)
                    + abs(frames[i].loudness_lufs + g - target_lufs)
                )
                candidate = (cost, seq + (g,))
                old = nxt.get(g)
                if old is None or candidate < old:
                    nxt[g] = candidate
        states = nxt
    cost, seq = min(states.values(), key=lambda item: (item[0], item[1]))
    return list(seq), cost


class BasicGainPlanTests(unittest.TestCase):
    def test_single_frame_ideal_gain(self):
        # 响度 -20、目标 -16：增益 4 恰好达标，真峰 -6 允许（-6+4=-2<=0）
        gains, cost = plan_gains([F(loudness=-20, peak=-6)], -16, 3)
        self.assertEqual(gains, [4])
        self.assertEqual(cost, 0)

    def test_single_frame_true_peak_caps_gain(self):
        # 理想增益 4，但真峰 -2 => 增益至多 2；偏差 |-20+2+16|=2
        gains, cost = plan_gains([F(loudness=-20, peak=-2)], -16, 3)
        self.assertEqual(gains, [2])
        self.assertEqual(cost, 2)

    def test_single_frame_gain_clamped_to_range(self):
        # 理想增益 40（-40 -> 0），增益上限 12；偏差 |-40+12-0|=28
        gains, cost = plan_gains([F(loudness=-40, peak=-80)], 0, 12)
        self.assertEqual(gains, [12])
        self.assertEqual(cost, 28)
        # 理想增益 -15（-5 -> -20），增益下限 -12；偏差 |-5-12+20|=3
        gains, cost = plan_gains([F(loudness=-5, peak=0)], -20, 12)
        self.assertEqual(gains, [-12])
        self.assertEqual(cost, 3)

    def test_change_cost_prefers_constant_gain(self):
        # 理想增益分别为 4、1、6（真峰 -80 允许全部增益），max_step=12；
        # 跟随理想序列的变化代价 2×(3+5)=16，而常量 4 偏差仅 3+1+3=7
        frames = [F(i * 100, i * 100 + 90, loudness=lo, peak=-80)
                  for i, lo in enumerate((-20, -15, -22))]
        gains, cost = plan_gains(frames, -16, 12)
        self.assertEqual(gains, [4, 4, 4])
        self.assertEqual(cost, 7)

    def test_step_zero_forces_equal_gains(self):
        frames = [F(0, 100, -30, -5), F(100, 200, -28, -10)]
        gains, cost = plan_gains(frames, -20, 0)
        self.assertEqual(len(set(gains)), 1)
        self.assertEqual(gains, [5, 5])  # 偏差 5+3=8；6 需 6+2=8 但字典序取 5
        self.assertEqual(cost, 8)

    def test_ramp_limit_between_frames(self):
        # 帧 0 真峰 0（增益<=0）、帧 1 需要 12 才能贴近目标，max_step=3
        frames = [F(0, 100, loudness=-40, peak=0),
                  F(100, 200, loudness=-20, peak=0)]
        gains, cost = plan_gains(frames, -8, 3)
        self.assertLessEqual(abs(gains[1] - gains[0]), 3)
        # 与独立实现一致
        bg, bc = independent_solve(frames, -8, 3)
        self.assertEqual((gains, cost), (bg, bc))

    def test_unreachable_high_gain_states_handled(self):
        # 末帧真峰 0 => 末帧只能取非正增益；步长 3 时前两帧的高增益状态无法
        # 延伸到末帧（DP 以 INF 处理不可达状态，重建时跳过）：
        frames = [
            F(0, 100, -10, -80),
            F(100, 200, -10, -80),
            F(200, 300, -40, 0),
        ]
        gains, cost = plan_gains(frames, -40, 3)
        # 末帧最优 0（偏差 0）要求帧 1 >= -3；帧 0/1 响度 -10 取 -3 偏差 27，
        # 而三帧同为 -12：偏差 2+2+28 与爬坡零代价，共 32（实测最优）
        self.assertLessEqual(gains[2], 0)
        self.assertLessEqual(abs(gains[2] - gains[1]), 3)
        self.assertLessEqual(abs(gains[1] - gains[0]), 3)
        bg, bc = independent_solve(frames, -40, 3)
        self.assertEqual((gains, cost), (bg, bc))

    def test_step_zero_two_frame_equal(self):
        frames = [F(0, 100, -10, -80), F(100, 200, -40, 0)]
        gains, _ = plan_gains(frames, -40, 0)  # step=0 两帧相等
        self.assertEqual(gains[0], gains[1])
        bg, _ = independent_solve(frames, -40, 0)
        self.assertEqual(gains, bg)

    def test_tie_picks_lexicographically_smallest(self):
        # 单帧响度 -20、目标 -20：增益 0 偏差 0，严格最优
        gains, cost = plan_gains([F(loudness=-20, peak=-80)], -20, 0)
        self.assertEqual(gains, [0])
        self.assertEqual(cost, 0)
        # 峰值把正增益全部排除时，-1 与 +1 的“偏差相等”裁决不再出现，
        # 此时取可达的最小偏差增益：响度 -20、目标 -19、真峰 0 =>
        # 增益 0 偏差 1（+1 被真峰禁止）
        gains, cost = plan_gains([F(loudness=-20, peak=0)], -19, 0)
        self.assertEqual(gains, [0])
        self.assertEqual(cost, 1)
        # 两帧折中：L=-22/-18、目标 -20、step=4，常量 -2 偏差 4 优于常量 0（同 4）
        # ——二者同为 4 时取字典序较小的 [-2,-2]
        frames = [F(0, 10, -22, -80), F(10, 20, -18, -80)]
        gains, cost = plan_gains(frames, -20, 4)
        self.assertEqual(gains, [-2, -2])
        self.assertEqual(cost, 4)

    def test_cost_formula(self):
        frames = [F(0, 100, -20, -2), F(100, 200, -25, -6),
                  F(200, 300, -18, -1)]
        gains, cost = plan_gains(frames, -16, 3)
        expected = sum(abs(frames[i].loudness_lufs + gains[i] + 16)
                       for i in range(3))
        expected += 2 * sum(abs(gains[i] - gains[i - 1]) for i in range(1, 3))
        self.assertEqual(cost, expected)
        for i, g in enumerate(gains):
            self.assertTrue(MIN_GAIN_DB <= g <= MAX_GAIN_DB)
            self.assertLessEqual(frames[i].true_peak_dbfs + g, 0)

    def test_returns_integer_gains(self):
        frames = [F(i * 10, i * 10 + 9, -17, -3) for i in range(10)]
        gains, _ = plan_gains(frames, -16, 2)
        self.assertTrue(all(isinstance(g, int) for g in gains))


class InputValidationTests(unittest.TestCase):
    def test_empty_frames_rejected(self):
        with self.assertRaises(DomainError) as ctx:
            plan_gains([], -16, 3)
        self.assertEqual(ctx.exception.code, "empty_frames")

    def test_negative_start_rejected(self):
        with self.assertRaises(DomainError) as ctx:
            plan_gains([F(start_ms=-1, end_ms=100)], -16, 3)
        self.assertEqual(ctx.exception.code, "invalid_time_range")
        self.assertEqual(ctx.exception.extra["frame_index"], 0)

    def test_end_not_after_start_rejected(self):
        for end in (100, 50):
            with self.assertRaises(DomainError) as ctx:
                plan_gains([F(start_ms=100, end_ms=end)], -16, 3)
            self.assertEqual(ctx.exception.code, "invalid_time_range")
            self.assertEqual(ctx.exception.extra["frame_index"], 0)

    def test_out_of_order_rejected(self):
        with self.assertRaises(DomainError) as ctx:
            plan_gains([F(100, 200, -20, -2), F(50, 90, -20, -2)], -16, 3)
        self.assertEqual(ctx.exception.code, "time_out_of_order")
        self.assertEqual(ctx.exception.extra["frame_index"], 1)

    def test_overlapping_rejected(self):
        with self.assertRaises(DomainError) as ctx:
            plan_gains([F(0, 100, -20, -2), F(99, 200, -20, -2)], -16, 3)
        self.assertEqual(ctx.exception.code, "overlapping_time_range")
        self.assertEqual(ctx.exception.extra["frame_index"], 1)

    def test_touching_endpoints_allowed(self):
        frames = [F(0, 100, -20, -2), F(100, 200, -20, -2)]
        gains, _ = plan_gains(frames, -16, 3)
        self.assertEqual(len(gains), 2)


class DifferentialTests(unittest.TestCase):
    def test_matches_independent_forward_dp(self):
        random.seed(20260926)
        for _ in range(800):
            n = random.randint(1, 6)
            frames = [
                F(i * 10, i * 10 + 9,
                  random.randint(-80, 0), random.randint(-80, 0))
                for i in range(n)
            ]
            target = random.randint(-40, 0)
            step = random.randint(0, 12)
            gains, cost = plan_gains(frames, target, step)
            expected_gains, expected_cost = independent_solve(frames, target, step)
            self.assertEqual(cost, expected_cost)
            self.assertEqual(gains, expected_gains)

    def test_matches_independent_on_edge_cases(self):
        random.seed(7)
        for _ in range(200):
            n = random.randint(1, 40)
            frames = [
                F(i * 10, i * 10 + 9, random.randint(-80, 0),
                  random.choice([-80, -79, -13, -12, -11, -1, 0]))
                for i in range(n)
            ]
            target = random.choice([-40, -23, -1, 0])
            step = random.choice([0, 1, 11, 12])
            gains, cost = plan_gains(frames, target, step)
            expected_gains, expected_cost = independent_solve(frames, target, step)
            self.assertEqual((cost, gains), (expected_cost, expected_gains))

    def test_max_300_frames(self):
        frames = [
            F(i * 10, i * 10 + 9,
              -17 if i % 2 else -25, -1 if i % 3 else -80)
            for i in range(300)
        ]
        gains, cost = plan_gains(frames, -16, 6)
        self.assertEqual(len(gains), 300)
        for i, g in enumerate(gains):
            self.assertTrue(MIN_GAIN_DB <= g <= MAX_GAIN_DB)
            self.assertLessEqual(frames[i].true_peak_dbfs + g, 0)
            if i:
                self.assertLessEqual(abs(g - gains[i - 1]), 6)
        expected_gains, expected_cost = independent_solve(frames, -16, 6)
        self.assertEqual(cost, expected_cost)
        self.assertEqual(gains, expected_gains)


if __name__ == "__main__":
    unittest.main()
