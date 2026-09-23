# -*- coding: utf-8 -*-
"""
'智麦鲜酵'——基于 Vis-NIR 与 LGAKNet 的麦粉品质与发酵智能监测系统

Tab 1【鲜粉品质 Vis-NIR 近红外快检】
  加载样品光谱 -> LGAKNet 智能检测 -> 水分 / 蛋白质 / 面筋等级
Tab 2【基于粉质的面团发酵动态监测】
  自动读取 Tab 1 粉质指标 -> S 型膨胀曲线 + pH 下降曲线 -> 最佳烘烤成熟度判定
"""

import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import math
from io import BytesIO

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lgaknet_predict import (LGAKNetPredictor, simulate_nir_spectrum,
                             R2_MOISTURE, R2_PROTEIN,
                             WATER_PEAKS, PROTEIN_PEAKS,
                             SPEC_LEN, WL_MIN, WL_MAX)

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 样品库与 LGAKNet 预测器（懒加载，权重由 lgaknet_predict.py 缓存）
# ---------------------------------------------------------------------------
SAMPLES = [
    {"id": "S001", "name": "德州优质麦粉", "moisture": 12.5, "protein": 14.2},
    {"id": "S002", "name": "济南标准麦粉", "moisture": 12.9, "protein": 12.0},
    {"id": "S003", "name": "苏北低筋麦粉", "moisture": 13.4, "protein": 9.5},
]
SAMPLE_SEED_BASE = 20260923

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = LGAKNetPredictor()
    return _predictor


def gluten_grade(protein):
    """按蛋白质含量划分面筋等级"""
    if protein >= 12.5:
        return "高筋", "#15803d", "#f0fdf4", "#bbf7d0"
    if protein >= 10.5:
        return "中筋", "#b45309", "#fffbeb", "#fde68a"
    return "低筋", "#6d28d9", "#f5f3ff", "#ddd6fe"


# ---------------------------------------------------------------------------
# 光谱绘图
# ---------------------------------------------------------------------------
def fig_to_np(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return img


def plot_spectrum(spec, sample_label, moisture=None, protein=None):
    """moisture/protein 为 None 时绘制待检测预览，否则绘制峰位标注结果图"""
    wl = np.linspace(WL_MIN, WL_MAX, np.asarray(spec).size)
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=120)
    ax.plot(wl, spec, color="#16a34a", lw=1.2)
    ax.set_xlabel("波长 (nm)")
    ax.set_ylabel("吸光度 (Abs)")

    if moisture is None:
        ax.set_title(f"{sample_label} · Vis-NIR 光谱预览（待检测）",
                     fontsize=12, fontweight="bold")
        ax.text(0.99, 0.03, f"光谱长度 {SPEC_LEN}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#64748b")
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.10)
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        for i, c in enumerate(WATER_PEAKS):
            ax.axvspan(c - 45, c + 45, color="#3b82f6", alpha=0.12)
            ax.axvline(c, color="#3b82f6", ls="--", lw=1)
            ax.text(c, ymax - span * (0.03 + 0.09 * i), f"水分 {c}nm",
                    ha="center", va="top", fontsize=8.5, color="#1d4ed8")
        for i, c in enumerate(PROTEIN_PEAKS):
            ax.axvspan(c - 40, c + 40, color="#f59e0b", alpha=0.12)
            ax.axvline(c, color="#f59e0b", ls="--", lw=1)
            ax.text(c, ymin + span * (0.03 + 0.09 * (i % 2)), f"蛋白质 {c}nm",
                    ha="center", va="bottom", fontsize=8.5, color="#b45309")
        ax.set_title(f"{sample_label} · LGAKNet 检测完成："
                     f"水分 {moisture:.2f}% | 蛋白质 {protein:.2f}%",
                     fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig_to_np(fig)


# ---------------------------------------------------------------------------
# 发酵动力学（Monod / Logistic，与训练数据同源）
# ---------------------------------------------------------------------------
def compute_curves(moisture, protein, temp, humidity, yeast):
    t = np.linspace(0, 180, 361)
    fT = math.exp(-((temp - 36.5) / 8.0) ** 2)      # 温度钟形响应，最适 36.5℃
    fH = humidity / (humidity + 55.0)               # 湿度 Monod
    fY = yeast / (yeast + 1.2)                      # 酵母 Monod
    fW = moisture / (moisture + 11.0)               # 粉质水分可用性

    E_max = 60 + 9.0 * protein + 130 * fT * fH * fY  # 蛋白质 -> 面筋网络强度
    k_e = 0.055 * max(fT, 0.15) * max(fY, 0.2) * (0.5 + 0.5 * fW) * (0.8 + 0.6 * fH)
    t50 = 62 - 22 * fT * fY
    expansion = E_max / (1 + np.exp(-k_e * (t - t50)))

    pH0 = 6.2 - 0.15 * fY
    dH = 1.9 * (0.5 + 0.5 * fT) * max(fY, 0.2) * (0.7 + 0.3 * fH)
    Kh = max(48 - 20 * fT, 20)
    pH = pH0 - dH * t / (t + Kh)

    m_e = expansion / max(E_max, 1e-6)
    acidity = np.clip((pH0 - pH) / 1.3, 0, 1)
    maturity = 100 * np.clip(0.58 * m_e + 0.42 * acidity, 0, 1)
    return t, expansion, pH, maturity


def maturity_status(m):
    """成熟度判定：绿=适度、黄=不足、红=过发"""
    if m < 55:
        return ("发酵不足", "面筋网络尚未充分舒展，建议延长醒发时间后再评估。",
                "#b45309", "#fffbeb", "#fde68a", "amber")
    if m <= 88:
        return ("适度发酵 · 最佳烘烤窗口", "面团膨胀与酸度处于理想区间，可立即整形入炉。",
                "#15803d", "#f0fdf4", "#bbf7d0", "green")
    return ("发酵过度", "面筋结构开始松弛、酸度偏高，建议缩短醒发或降低温度。",
            "#dc2626", "#fef2f2", "#fecaca", "red")


def bake_window(t, maturity):
    """最佳烘烤时间窗：成熟度 70%~88% 对应的时段"""
    i70 = np.argmax(maturity >= 70) if (maturity >= 70).any() else None
    i88 = np.argmax(maturity > 88) if (maturity > 88).any() else len(t) - 1
    if i70 is None:
        return None
    return float(t[i70]), float(t[i88])


def plot_ferment(t, expansion, pH, maturity, t_now, window):
    fig, ax1 = plt.subplots(figsize=(9.2, 4.6), dpi=120)
    ax2 = ax1.twinx()

    if window:
        ax1.axvspan(window[0], window[1], color="#16a34a", alpha=0.08)
        ax1.text(np.mean(window), ax1.get_ylim()[1], "最佳烘烤窗口",
                 ha="center", va="bottom", fontsize=9, color="#15803d")

    ax1.plot(t, expansion, color="#16a34a", lw=2, label="面团膨胀倍率 (S 型)")
    ax2.plot(t, pH, color="#8b5cf6", lw=2, ls="--", label="pH 酸度下降")
    ax1.axvline(t_now, color="#dc2626", ls=":", lw=1.5)
    ax1.text(t_now, ax1.get_ylim()[0], f" 当前 {t_now:.0f} min",
             fontsize=9, color="#dc2626", va="bottom")

    ax1.set_xlabel("发酵时长 (min)")
    ax1.set_ylabel("膨胀倍率 (%)", color="#15803d")
    ax2.set_ylabel("pH", color="#7c3aed")
    ax1.set_xlim(0, 180)
    ax2.set_ylim(3.8, 6.4)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
    ax1.set_title("面团发酵动态推演：S 型膨胀曲线 / pH 下降曲线",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig_to_np(fig)


# ---------------------------------------------------------------------------
# HTML 片段
# ---------------------------------------------------------------------------
CSS = """
body, .gradio-container {
  background: radial-gradient(1200px 500px at 70% -10%, #dcfce7 0%, transparent 60%),
              radial-gradient(900px 400px at 10% 110%, #fef9c3 0%, transparent 55%),
              #f8fafc !important;
}
.qc-card {
  background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px;
  padding: 18px 20px; box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
}
.qc-page-title {
  background: linear-gradient(120deg, #16a34a, #0d9488, #22c55e, #0f766e, #16a34a);
  background-size: 300% 300%;
  animation: titleGradient 12s ease infinite;
  border-radius: 20px; padding: 26px 30px; color: #fff;
  box-shadow: 0 12px 32px rgba(22, 163, 74, 0.25);
  position: relative; overflow: hidden;
}
@keyframes titleGradient {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.qc-page-title::after {
  content: ""; position: absolute; top: 0; left: -60%;
  width: 40%; height: 100%;
  background: linear-gradient(105deg, transparent, rgba(255, 255, 255, 0.25), transparent);
  animation: titleShine 4.5s ease-in-out infinite;
}
@keyframes titleShine {
  0% { left: -60%; }
  60%, 100% { left: 130%; }
}
.qc-page-title h1, .qc-page-title p { position: relative; z-index: 1; }
.qc-page-title h1 { margin: 0 0 6px; font-size: 26px; font-weight: 800; letter-spacing: 1px; }
.qc-page-title p { margin: 0; opacity: 0.92; font-size: 14px; }
.qc-section-tag {
  display: inline-block; background: #dcfce7; color: #15803d; font-weight: 700;
  border-radius: 999px; padding: 4px 14px; font-size: 13px; margin-bottom: 10px;
}
.fm-big-row { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 6px; }
.fm-big-card {
  flex: 1; min-width: 150px; border-radius: 16px; padding: 18px;
  border: 1.5px solid #e2e8f0; background: #f8fafc; text-align: center;
}
.fm-big-card .lab { font-size: 13px; color: #64748b; font-weight: 600; }
.fm-big-card .val { font-size: 34px; font-weight: 800; margin: 6px 0 2px; }
.fm-big-card .unit { font-size: 13px; color: #94a3b8; }
.fm-r2-chip {
  margin-top: 14px; background: #eff6ff; border: 1px solid #bfdbfe;
  color: #1d4ed8; border-radius: 12px; padding: 10px 14px; font-size: 13px;
}
.fm-flour-chip {
  background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
  padding: 10px 14px; font-size: 13.5px; color: #14532d; margin-top: 12px;
}
.fm-flour-chip b { color: #15803d; }
.fm-alert {
  border-radius: 16px; padding: 16px 18px; border: 1.5px solid; margin-top: 12px;
}
.fm-alert .t { font-size: 17px; font-weight: 800; }
.fm-alert .d { font-size: 13.5px; margin-top: 4px; line-height: 1.6; }
.fm-mini-row { display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
.fm-mini {
  flex: 1; min-width: 120px; border-radius: 12px; padding: 10px 12px;
  background: #f8fafc; border: 1px solid #e2e8f0; text-align: center;
}
.fm-mini .k { font-size: 12px; color: #64748b; }
.fm-mini .v { font-size: 20px; font-weight: 800; margin-top: 2px; }

/* 图像展示区：显式白底白边，覆盖深色残留，去除黑边 */
.image-frame, .image-container, .image-frame .image-container {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 14px !important;
}
.image-frame img { border-radius: 12px !important; background: #ffffff !important; }

/* 操作按钮：3D 立体按键效果 */
.action-btn {
  border-radius: 999px !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  transform-style: preserve-3d;
  box-shadow: 0 6px 0 #15803d,
              0 14px 22px rgba(22, 163, 74, 0.30),
              inset 0 2px 2px rgba(255, 255, 255, 0.45) !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease !important;
}
.action-btn:hover {
  transform: perspective(500px) rotateX(12deg) translateY(-3px) !important;
  box-shadow: 0 9px 0 #15803d,
              0 20px 30px rgba(22, 163, 74, 0.38),
              0 0 24px rgba(74, 222, 128, 0.75),
              0 0 48px rgba(220, 252, 231, 0.9),
              inset 0 2px 2px rgba(255, 255, 255, 0.45) !important;
  filter: brightness(1.06) !important;
}
.action-btn:active {
  transform: translateY(5px) !important;
  box-shadow: 0 1px 0 #15803d,
              0 5px 10px rgba(22, 163, 74, 0.25) !important;
}
.action-btn.primary, button.primary.action-btn {
  animation: btnPulse 2.2s ease-out infinite;
}
@keyframes btnPulse {
  0%, 100% { box-shadow: 0 6px 0 #15803d, 0 14px 22px rgba(22, 163, 74, 0.30),
             0 0 0 0 rgba(34, 197, 94, 0.45), inset 0 2px 2px rgba(255, 255, 255, 0.45); }
  50% { box-shadow: 0 6px 0 #15803d, 0 14px 22px rgba(22, 163, 74, 0.30),
        0 0 0 12px rgba(34, 197, 94, 0), inset 0 2px 2px rgba(255, 255, 255, 0.45); }
}
button.action-btn.secondary, .action-btn.secondary {
  background: #ffffff !important;
  color: #15803d !important;
  border: 2px solid #16a34a !important;
  box-shadow: 0 6px 0 #bbf7d0,
              0 14px 22px rgba(22, 163, 74, 0.18),
              inset 0 2px 2px rgba(255, 255, 255, 0.9) !important;
}
button.action-btn.secondary:hover, .action-btn.secondary:hover {
  box-shadow: 0 9px 0 #bbf7d0,
              0 20px 30px rgba(22, 163, 74, 0.22),
              0 0 24px rgba(134, 239, 172, 0.65),
              0 0 48px rgba(240, 253, 244, 0.95),
              inset 0 2px 2px rgba(255, 255, 255, 0.9) !important;
}
button.action-btn.secondary:active, .action-btn.secondary:active {
  box-shadow: 0 1px 0 #bbf7d0,
              0 5px 10px rgba(22, 163, 74, 0.15) !important;
}
"""

HEADER_HTML = """
<div class="qc-page-title">
  <h1>'智麦鲜酵'——基于 Vis-NIR 与 LGAKNet 的麦粉品质与发酵智能监测系统</h1>
  <p>近红外光谱快检 · LGAKNet 品质预测 · 面团发酵动力学动态监测</p>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center;color:#94a3b8;font-size:12px;padding:14px 0 4px;">
  智麦鲜酵 · Vis-NIR Spectroscopy + LGAKNet + Fermentation Kinetics
</div>
"""


def cards_html(moisture, protein, sample_label):
    grade, fc, bg, bd = gluten_grade(protein)
    return f"""
<div class="fm-big-row">
  <div class="fm-big-card" style="background:#eff6ff;border-color:#bfdbfe;">
    <div class="lab">水分含量</div>
    <div class="val" style="color:#1d4ed8;">{moisture:.2f}</div>
    <div class="unit">%</div>
  </div>
  <div class="fm-big-card" style="background:#f0fdf4;border-color:#bbf7d0;">
    <div class="lab">蛋白质含量</div>
    <div class="val" style="color:#15803d;">{protein:.2f}</div>
    <div class="unit">%</div>
  </div>
  <div class="fm-big-card" style="background:{bg};border-color:{bd};">
    <div class="lab">面筋等级评估</div>
    <div class="val" style="color:{fc};">{grade}</div>
    <div class="unit">蛋白质 {"≥12.5%" if grade == "高筋" else "10.5~12.5%" if grade == "中筋" else "<10.5%"}</div>
  </div>
</div>
<div class="fm-r2-chip">
  LGAKNet 拟合指标：水分 R² = {R2_MOISTURE:.4f} · 蛋白质 R² = {R2_PROTEIN:.4f}
  ｜ 样品：{sample_label} ｜ 粉质指标已自动传递至 Tab 2
</div>
"""


def flour_chip_html(state, prefix=""):
    m, p = state.get("moisture"), state.get("protein")
    if m is None:
        return ('<div class="fm-flour-chip">当前粉质数据：<b>默认示例值 '
                '水分 12.80% · 蛋白质 12.40%</b><br>'
                '数据来源：未做光谱快检时的演示默认值，请先在 Tab 1 完成 LGAKNet 检测</div>')
    grade, _, _, _ = gluten_grade(p)
    return (f'<div class="fm-flour-chip">当前粉质数据：<b>水分 {m:.2f}% · '
            f'蛋白质 {p:.2f}% · {grade}粉</b><br>'
            f'数据来源：{prefix}{state["sample_id"]} {state["sample_name"]} '
            f'LGAKNet 实测光谱数据</div>')


CARDS_PLACEHOLDER = '<div class="fm-r2-chip">请先加载样品光谱，再点击「LGAKNet 智能检测」。</div>'


# ---------------------------------------------------------------------------
# Tab 1 事件
# ---------------------------------------------------------------------------
def load_sample(state):
    idx = state.get("next_idx", 0) % len(SAMPLES)
    s = SAMPLES[idx]
    rng = np.random.default_rng(SAMPLE_SEED_BASE + idx)
    spec, _, _ = simulate_nir_spectrum(moisture=s["moisture"], protein=s["protein"], rng=rng)

    state = {"next_idx": idx + 1, "sample_id": s["id"], "sample_name": s["name"],
             "spectrum": spec}
    label = f"{s['id']} {s['name']}"
    img = plot_spectrum(spec, label)
    status = f"已加载样品光谱：{label}（{WL_MIN:.0f}~{WL_MAX:.0f}nm，长度 {SPEC_LEN}），请点击「LGAKNet 智能检测」。"
    return img, status, CARDS_PLACEHOLDER, state, flour_chip_html({})


def run_detect(state):
    spec = state.get("spectrum")
    if spec is None:
        return (gr.update(), "请先点击「加载样品光谱」按钮。",
                CARDS_PLACEHOLDER, state, gr.update())
    moisture, protein = get_predictor().predict(spec)
    state["moisture"], state["protein"] = moisture, protein
    label = f"{state['sample_id']} {state['sample_name']}"
    img = plot_spectrum(spec, label, moisture, protein)
    grade, _, _, _ = gluten_grade(protein)
    status = (f"检测完成：{label} → 水分 {moisture:.2f}%，蛋白质 {protein:.2f}%，"
              f"判定为{grade}粉。")
    return img, status, cards_html(moisture, protein, label), state, \
        flour_chip_html(state, prefix="Tab 1 ")


# ---------------------------------------------------------------------------
# Tab 2 事件
# ---------------------------------------------------------------------------
def run_ferment(t_now, temp, humidity, yeast, state):
    m = state.get("moisture")
    p = state.get("protein")
    if m is None:
        m, p = 12.8, 12.4
    t, expansion, pH, maturity = compute_curves(m, p, temp, humidity, yeast)
    window = bake_window(t, maturity)

    exp_now = float(np.interp(t_now, t, expansion))
    ph_now = float(np.interp(t_now, t, pH))
    mat_now = float(np.interp(t_now, t, maturity))
    title, desc, fc, bg, bd, _ = maturity_status(mat_now)

    img = plot_ferment(t, expansion, pH, maturity, t_now, window)

    metrics = f"""
<div class="fm-mini-row">
  <div class="fm-mini"><div class="k">当前膨胀倍率</div>
    <div class="v" style="color:#15803d;">{exp_now:.1f}%</div></div>
  <div class="fm-mini"><div class="k">当前 pH 酸度</div>
    <div class="v" style="color:#7c3aed;">{ph_now:.2f}</div></div>
  <div class="fm-mini"><div class="k">醒发成熟度</div>
    <div class="v" style="color:{fc};">{mat_now:.1f}%</div></div>
</div>
"""
    win_txt = (f"建议入炉时段：第 {window[0]:.0f} ~ {window[1]:.0f} min。"
               if window else "当前工艺参数下 180 min 内未达到最佳烘烤窗口，建议提高温度或酵母比例。")
    alert = f"""
<div class="fm-alert" style="background:{bg};border-color:{bd};color:{fc};">
  <div class="t">最佳烘烤成熟度判定：{title}</div>
  <div class="d">{desc}<br>{win_txt}</div>
</div>
"""
    return img, metrics, alert, flour_chip_html(state)


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------
with gr.Blocks(title="智麦鲜酵 · Vis-NIR 与 LGAKNet 智能监测系统") as demo:
    state = gr.State({})
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        # ===================== Tab 1 =====================
        with gr.Tab("鲜粉品质 Vis-NIR 近红外快检"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=420):
                    with gr.Group(elem_classes=["qc-card"]):
                        gr.HTML('<div class="qc-section-tag">STEP 1 · 样品光谱加载</div>')
                        load_btn = gr.Button("加载样品光谱（如 S001 德州优质麦粉）",
                                             variant="secondary", size="lg",
                                             elem_classes=["action-btn"])
                        detect_btn = gr.Button("LGAKNet 智能检测",
                                               variant="primary", size="lg",
                                               elem_classes=["action-btn"])
                        scan_status = gr.Markdown("尚未加载样品光谱。")
                with gr.Column(scale=7, min_width=560):
                    with gr.Group(elem_classes=["qc-card"]):
                        gr.HTML('<div class="qc-section-tag">STEP 2 · 光谱曲线与检测结果</div>')
                        spec_img = gr.Image(label="Vis-NIR 光谱曲线",
                                            interactive=False, height=360)
                        result_cards = gr.HTML(CARDS_PLACEHOLDER)

        # ===================== Tab 2 =====================
        with gr.Tab("基于粉质的面团发酵动态监测"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=360):
                    with gr.Group(elem_classes=["qc-card"]):
                        gr.HTML('<div class="qc-section-tag">发酵工艺参数</div>')
                        time_slider = gr.Slider(30, 180, value=90, step=5,
                                                label="发酵时长 (min)")
                        temp_slider = gr.Slider(24, 42, value=33, step=0.5,
                                                label="醒发温度 (℃)")
                        hum_slider = gr.Slider(55, 95, value=75, step=1,
                                               label="环境湿度 (%)")
                        yeast_slider = gr.Slider(0.5, 3.0, value=1.5, step=0.1,
                                                 label="酵母比例 (%)")
                        ferment_btn = gr.Button("启动发酵动态推演",
                                                variant="primary", size="lg",
                                                elem_classes=["action-btn"])
                        flour_chip = gr.HTML(flour_chip_html({}))
                with gr.Column(scale=8, min_width=560):
                    with gr.Group(elem_classes=["qc-card"]):
                        gr.HTML('<div class="qc-section-tag">发酵动态推演结果</div>')
                        curve_img = gr.Image(label="S 型膨胀曲线 / pH 下降曲线",
                                             interactive=False, height=360)
                        metrics_html = gr.HTML("")
                        alert_html = gr.HTML("")

    gr.HTML(FOOTER_HTML)

    # ------------------------- 事件绑定 -------------------------
    load_btn.click(
        fn=load_sample,
        inputs=[state],
        outputs=[spec_img, scan_status, result_cards, state, flour_chip]
    )
    detect_btn.click(
        fn=run_detect,
        inputs=[state],
        outputs=[spec_img, scan_status, result_cards, state, flour_chip]
    )
    ferment_btn.click(
        fn=run_ferment,
        inputs=[time_slider, temp_slider, hum_slider, yeast_slider, state],
        outputs=[curve_img, metrics_html, alert_html, flour_chip]
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Default(),
                server_name="127.0.0.1", server_port=7860,
                show_error=True, inbrowser=True)
