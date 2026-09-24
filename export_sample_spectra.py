# -*- coding: utf-8 -*-
"""将 app.py 内置的 3 组样品光谱导出为 CSV 文件（测试样品/ 目录）。

光谱生成方式与 Tab1「加载样品光谱」完全一致（同一种子、同一模拟函数），
保证导出数据与系统内置数据逐点一致。
"""
import os

import numpy as np

from app import SAMPLES, SAMPLE_SEED_BASE
from lgaknet_predict import WL_MAX, WL_MIN, SPEC_LEN, simulate_nir_spectrum

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "测试样品")

EXPORT_NAMES = {
    "S001": "S001_德州高筋粉.csv",
    "S002": "S002_苏北中筋粉.csv",
    "S003": "S003_济南低筋粉.csv",
}


def export_samples(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    wl = np.linspace(WL_MIN, WL_MAX, SPEC_LEN)
    paths = []
    for idx, s in enumerate(SAMPLES):
        rng = np.random.default_rng(SAMPLE_SEED_BASE + idx)
        spec, _, _ = simulate_nir_spectrum(moisture=s["moisture"],
                                           protein=s["protein"], rng=rng)
        path = os.path.join(out_dir, EXPORT_NAMES[s["id"]])
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("wavelength_nm,absorbance\n")
            for w, a in zip(wl, spec):
                f.write(f"{w:.2f},{a:.6f}\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in export_samples():
        print("已导出:", p)
