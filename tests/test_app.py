# -*- coding: utf-8 -*-
"""智麦鲜酵大屏核心逻辑测试套件

运行：E:\\Conda\\envs\\food_defect\\python.exe -m unittest discover -s tests -v
"""
import os
import re
import unittest

import numpy as np

import app
from lgaknet_predict import SPEC_LEN, R2_MOISTURE, R2_PROTEIN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PARAMS = dict(moisture=12.8, protein=12.4, temp=33.0, humidity=75.0, yeast=1.5)


class TestGlutenGrade(unittest.TestCase):
    def test_high_gluten_boundary(self):
        self.assertEqual(app.gluten_grade(12.5)[0], "高筋")

    def test_mid_gluten_boundary(self):
        self.assertEqual(app.gluten_grade(12.49)[0], "中筋")
        self.assertEqual(app.gluten_grade(10.5)[0], "中筋")

    def test_low_gluten(self):
        self.assertEqual(app.gluten_grade(9.5)[0], "低筋")


class TestFermentKinetics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t, cls.expansion, cls.pH, cls.maturity = app.compute_curves(**DEFAULT_PARAMS)

    def test_curve_lengths(self):
        self.assertEqual(len(self.t), 361)
        for arr in (self.expansion, self.pH, self.maturity):
            self.assertEqual(len(arr), 361)

    def test_expansion_is_sigmoid_and_bounded(self):
        self.assertTrue(np.all(np.diff(self.expansion) >= -1e-9))  # S 型单调递增
        self.assertGreater(self.expansion[-1], 120.0)
        self.assertLess(self.expansion[-1], 280.0)

    def test_ph_declines(self):
        self.assertTrue(np.all(np.diff(self.pH) <= 1e-9))
        self.assertLess(self.pH[-1], self.pH[0])

    def test_maturity_range(self):
        self.assertGreaterEqual(self.maturity.min(), 0.0)
        self.assertLessEqual(self.maturity.max(), 100.0)

    def test_protein_boosts_expansion(self):
        _, exp_low, _, _ = app.compute_curves(12.8, 9.5, 33.0, 75.0, 1.5)
        _, exp_high, _, _ = app.compute_curves(12.8, 14.5, 33.0, 75.0, 1.5)
        self.assertGreater(exp_high[-1], exp_low[-1])  # 蛋白质 -> 面筋强度 -> 膨胀上限


class TestMaturityStatus(unittest.TestCase):
    def test_three_states(self):
        self.assertIn("不足", app.maturity_status(30)[0])
        self.assertIn("适度", app.maturity_status(70)[0])
        self.assertIn("过度", app.maturity_status(95)[0])

    def test_colors_follow_theme(self):
        self.assertEqual(app.maturity_status(70)[2], "#15803d")   # 绿=适度
        self.assertEqual(app.maturity_status(30)[2], "#b45309")   # 黄=不足
        self.assertEqual(app.maturity_status(95)[2], "#dc2626")   # 红=过发


class TestBakeWindow(unittest.TestCase):
    def test_default_params_reach_window(self):
        t, _, _, maturity = app.compute_curves(**DEFAULT_PARAMS)
        window = app.bake_window(t, maturity)
        self.assertIsNotNone(window)
        self.assertLess(window[0], window[1])
        self.assertGreaterEqual(window[0], 0.0)
        self.assertLessEqual(window[1], 180.0)

    def test_extreme_cold_window_consistent(self):
        t, _, _, maturity = app.compute_curves(12.8, 12.4, 24.0, 55.0, 0.5)
        window = app.bake_window(t, maturity)
        if window is not None:
            self.assertLess(window[0], window[1])


class TestTab1Flow(unittest.TestCase):
    def test_load_sample_cycles_and_returns_spectrum(self):
        _, status, _, state, _ = app.load_sample({})
        self.assertIn("S001", status)
        self.assertEqual(state["spectrum"].size, SPEC_LEN)
        _, status2, _, state2, _ = app.load_sample(state)
        self.assertIn("S002", status2)

    def test_detect_without_sample_warns(self):
        _, status, cards, _, _ = app.run_detect({})
        self.assertIn("请先", status)
        self.assertIn("请先加载样品光谱", cards)

    def test_detect_fills_state_and_cards(self):
        _, _, _, state, _ = app.load_sample({})
        _, status, cards, state, chip = app.run_detect(state)
        self.assertIsNotNone(state.get("moisture"))
        self.assertIsNotNone(state.get("protein"))
        self.assertIn("检测完成", status)
        self.assertIn(f"{R2_MOISTURE:.4f}", cards)
        self.assertIn(f"{R2_PROTEIN:.4f}", cards)
        self.assertIn("S001", chip)


class TestTab2Flow(unittest.TestCase):
    def test_ferment_outputs_default_flour(self):
        _, metrics, alert, chip = app.run_ferment(90, 33.0, 75.0, 1.5, {})
        self.assertIn("当前膨胀倍率", metrics)
        self.assertIn("最佳烘烤成熟度判定", alert)
        self.assertIn("默认示例值", chip)

    def test_ferment_reads_tab1_values(self):
        state = {"moisture": 13.0, "protein": 14.0,
                 "sample_id": "S001", "sample_name": "德州优质麦粉"}
        _, _, _, chip = app.run_ferment(90, 33.0, 75.0, 1.5, state)
        self.assertIn("S001", chip)
        self.assertIn("13.00", chip)


class TestImagesNoBlackBorder(unittest.TestCase):
    """回归：展示图必须为白底，不允许出现黑边/黑底"""

    def _assert_white_corners(self, img):
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.ndim, 3)
        for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            self.assertGreater(int(img[r, c].sum()), 700)  # 接近纯白

    def test_spectrum_preview_white_bg(self):
        img = app.plot_spectrum(np.zeros(SPEC_LEN), "TEST")
        self._assert_white_corners(img)

    def test_spectrum_result_white_bg(self):
        img = app.plot_spectrum(np.zeros(SPEC_LEN), "TEST", 12.5, 13.8)
        self._assert_white_corners(img)

    def test_ferment_plot_white_bg(self):
        t, expansion, pH, maturity = app.compute_curves(**DEFAULT_PARAMS)
        img = app.plot_ferment(t, expansion, pH, maturity, 90,
                               app.bake_window(t, maturity))
        self._assert_white_corners(img)


class TestUIStatics(unittest.TestCase):
    def test_header_title_exact(self):
        self.assertIn("'智麦鲜酵'——基于 Vis-NIR 与 LGAKNet 的麦粉品质与发酵智能监测系统",
                      app.HEADER_HTML)

    def test_auto_open_browser_enabled(self):
        """回归：启动服务必须自动打开浏览器（launch inbrowser=True）"""
        with open(os.path.join(PROJECT_ROOT, "app.py"), encoding="utf-8") as f:
            self.assertIn("inbrowser=True", f.read())

    def test_no_emoji_in_source(self):
        with open(os.path.join(PROJECT_ROOT, "app.py"), encoding="utf-8") as f:
            src = f.read()
        emoji = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
        self.assertIsNone(emoji.search(src))

    def test_no_dark_styles(self):
        """回归：禁止深色主题/深色背景，图像区必须显式白底"""
        self.assertNotIn("color_schemes", app.CSS)
        self.assertNotIn("#0b1020", app.CSS)
        self.assertIn(".image-frame", app.CSS)

    def test_title_animated_gradient(self):
        """回归：标题栏必须为动态渐变（渐变动画 + 流光扫过）"""
        self.assertIn("@keyframes titleGradient", app.CSS)
        self.assertIn("animation: titleGradient", app.CSS)
        self.assertIn("background-size: 300% 300%", app.CSS)
        self.assertIn("titleShine", app.CSS)

    def test_buttons_prominent(self):
        """回归：操作按钮必须使用 action-btn 3D 立体样式"""
        with open(os.path.join(PROJECT_ROOT, "app.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertGreaterEqual(src.count('elem_classes=["action-btn"]'), 3)
        self.assertIn(".action-btn", app.CSS)
        self.assertIn("rotateX", app.CSS)            # 悬停 3D 翻转
        self.assertIn("@keyframes btnPulse", app.CSS)  # 主按钮呼吸光环
        self.assertIn("0 0 24px", app.CSS)           # 悬停绿色光晕
        self.assertIn("0 0 48px", app.CSS)           # 悬停淡白外圈光晕

    def test_bat_points_to_conda_env(self):
        with open(os.path.join(PROJECT_ROOT, "启动大屏.bat"), encoding="utf-8") as f:
            bat = f.read()
        self.assertIn("food_defect", bat)
        self.assertTrue(os.path.exists(r"E:\Conda\envs\food_defect\python.exe"))


if __name__ == "__main__":
    unittest.main()
