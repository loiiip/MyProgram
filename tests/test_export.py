# -*- coding: utf-8 -*-
"""样品光谱 CSV 导出回归测试"""
import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import app
from export_sample_spectra import EXPORT_NAMES, OUT_DIR, export_samples
from lgaknet_predict import SPEC_LEN, WL_MAX, WL_MIN, simulate_nir_spectrum


class TestSampleExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = export_samples()

    def test_three_csvs_exist(self):
        self.assertEqual(len(self.paths), 3)
        for s in app.SAMPLES:
            path = os.path.join(OUT_DIR, EXPORT_NAMES[s["id"]])
            self.assertTrue(os.path.exists(path), f"缺少导出文件: {path}")

    def test_csv_shape_and_axis(self):
        for path in self.paths:
            data = np.genfromtxt(path, delimiter=",", skip_header=1,
                                 encoding="utf-8-sig")
            self.assertEqual(data.shape, (SPEC_LEN, 2), f"{path} 应为 2048 行 2 列")
            self.assertAlmostEqual(data[0, 0], WL_MIN, places=2)
            self.assertAlmostEqual(data[-1, 0], WL_MAX, places=2)

    def test_csv_matches_builtin_spectrum(self):
        """导出数据必须与系统内置光谱逐点一致（同种子复现）"""
        for idx, s in enumerate(app.SAMPLES):
            rng = np.random.default_rng(app.SAMPLE_SEED_BASE + idx)
            spec, _, _ = simulate_nir_spectrum(moisture=s["moisture"],
                                               protein=s["protein"], rng=rng)
            path = os.path.join(OUT_DIR, EXPORT_NAMES[s["id"]])
            data = np.genfromtxt(path, delimiter=",", skip_header=1,
                                 encoding="utf-8-sig")
            self.assertTrue(
                np.allclose(data[:, 1], spec, atol=1e-6),
                f"{EXPORT_NAMES[s['id']]} 与内置光谱不一致")


if __name__ == "__main__":
    unittest.main()
