"""Generate the three paper figures as 300-dpi PNGs.

  图1 STL分解图          — real Solar-Site1 window decomposed into trend/seasonal/residual
  图2 FFT-DCT双域周期图   — real spectra, top-5 peaks marked, showing the two period grids differ
  图3 并行结构示意图       — PA-STL-DualTimesNet block diagram

Run: python make_figures.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data_provider.data_factory import data_provider
from models.TimesNetFull import STLDecomposition, dct_ii

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

OUT = "figures"
STEPS_PER_DAY = 96          # Solar-Site1, 15-min sampling


def load_site1(seq_len):
    args = argparse.Namespace(
        task_name="long_term_forecast", data="solar_site1", root_path="./data_provider/",
        data_path="site1.xlsx", features="M", target="Active_Power", freq="t",
        embed="timeF", scale=True, seasonal_patterns="Monthly",
        seq_len=seq_len, label_len=48, pred_len=96, batch_size=8,
        num_workers=0, enc_in=8, dec_in=8, c_out=8, d_model=64, d_ff=64,
        e_layers=2, d_layers=1, top_k=5, num_kernels=6, dropout=0.1,
        use_stl=True, stl_period=96, stl_impl="torch", use_autocorr=True, use_dct=True,
    )
    ds, loader = data_provider(args, "train")
    x, _, _, _ = next(iter(loader))
    return x, args


# ----------------------------------------------------------------------
# 图1  STL decomposition
# ----------------------------------------------------------------------
def fig_stl():
    T = 3 * STEPS_PER_DAY
    x, args = load_site1(T)
    series = x[0, :, 0].numpy()                       # one variable
    xt = torch.tensor(series, dtype=torch.float32).reshape(1, T, 1)
    stl = STLDecomposition(period=STEPS_PER_DAY, impl="torch")
    tr, se, re_ = stl(xt)
    tr, se, re_ = tr[0, :, 0].numpy(), se[0, :, 0].numpy(), re_[0, :, 0].numpy()

    t = np.arange(T) * 0.25                           # 15 min -> hours
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.4), sharex=True)
    for ax, y, title, color in zip(
            axes,
            [series, tr, se, re_],
            ["原始功率序列 $x_t$", "趋势分量 Trend", "季节分量 Seasonal", "残差分量 Residual"],
            ["black", "tab:red", "tab:blue", "tab:gray"]):
        ax.plot(t, y, color=color, lw=1.1)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=8)
    axes[0].set_title("STL分解（周期 p = 96 步 = 1 天）", fontsize=10)
    axes[-1].set_xlabel("时间 (小时)", fontsize=9)
    # mark one cycle
    for ax in axes:
        ax.axvspan(24, 48, color="tab:green", alpha=0.08, lw=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_stl.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# 图2  FFT vs DCT spectra and the two period grids
# ----------------------------------------------------------------------
def fig_domain():
    T = 4 * STEPS_PER_DAY                            # 4 days
    x, args = load_site1(T)
    series = x[0, :, 0].numpy()
    xt = torch.tensor(series, dtype=torch.float32).reshape(1, T, 1)

    fa = torch.fft.rfft(xt, dim=1).abs().mean(0).mean(-1).numpy()
    da = dct_ii(xt).abs().mean(0).mean(-1).numpy()
    fa[0] = 0.0
    da[0] = da[1] = 0.0

    kf = np.argsort(fa)[::-1][:5]
    kd = np.argsort(da)[::-1][:5]
    f_periods = [T / k for k in kf]
    d_periods = [2 * T / k for k in kd]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6))

    t = np.arange(T) * 0.25
    axes[0].plot(t, series, color="black", lw=1.0)
    axes[0].set_title("输入序列（Solar-Site1，4 天）", fontsize=10)
    axes[0].set_ylabel("功率", fontsize=9)
    axes[0].grid(alpha=0.3, lw=0.4)

    # FFT spectrum, plotted against period
    fp = T / np.arange(1, len(fa))
    axes[1].plot(fp, fa[1:], color="tab:blue", lw=1.0)
    axes[1].scatter(f_periods, fa[kf], color="tab:red", zorder=5, s=28)
    for p, v in zip(f_periods, fa[kf]):
        axes[1].annotate("%.0f步\n%.1fh" % (p, p * 0.25),
                         (p, v), textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=7, color="tab:red")
    axes[1].set_xlim(0, T)
    axes[1].set_ylabel("FFT 幅值", fontsize=9)
    axes[1].set_title("FFT 频谱：栅格 $T/k$，检出周期与日周期谐波对齐", fontsize=10)
    axes[1].grid(alpha=0.3, lw=0.4)

    dp = 2 * T / np.arange(2, len(da))
    fp2 = fa[1:]
    axes[2].plot(dp, da[2:], color="tab:green", lw=1.0)
    axes[2].scatter(d_periods, da[kd], color="tab:red", zorder=5, s=28)
    for p, v in zip(d_periods, da[kd]):
        axes[2].annotate("%.0f步\n%.1fh" % (p, p * 0.25),
                         (p, v), textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=7, color="tab:red")
    axes[2].set_xlim(0, T)
    axes[2].set_ylabel("DCT 幅值", fontsize=9)
    axes[2].set_xlabel("周期 (步)", fontsize=9)
    axes[2].set_title("DCT 频谱：栅格 $2T/k$（间距为FFT的一半），候选周期不同", fontsize=10)
    axes[2].grid(alpha=0.3, lw=0.4)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.axvline(STEPS_PER_DAY, color="gray", ls="--", lw=0.8)
    axes[1].text(STEPS_PER_DAY + 2, axes[1].get_ylim()[1] * 0.75,
                 "日周期 96步", fontsize=7.5, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_dualdmain.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# 图3  parallel architecture
# ----------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec="black", fs=8.5, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.35)


def _arrow(ax, p0, p1, color="black", lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 lw=lw, color=color,
                                 shrinkA=0, shrinkB=0))


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # left column: pipeline
    _box(ax, 0.30, 0.915, 0.40, 0.065, "输入序列 $x_{1:T}$", "#eef3fb")
    _box(ax, 0.30, 0.805, 0.40, 0.065, "STL分解（周期 p）", "#eef3fb")
    _box(ax, 0.24, 0.700, 0.52, 0.062,
         "趋势 / 季节 / 残差  →  三分量独立嵌入  →  相加", "#eef3fb")

    # the block
    ax.add_patch(FancyBboxPatch((0.055, 0.075), 0.89, 0.575,
                                boxstyle="round,pad=0.012",
                                fc="#fbfbfd", ec="black", lw=1.2, ls="--"))
    ax.text(0.075, 0.635, "DualDomainTimesBlock × N", fontsize=9, va="center")

    # branch A (conv, two sub-paths)
    _box(ax, 0.09, 0.455, 0.20, 0.075, "FFT 周期检测", "#dce9f7")
    _box(ax, 0.09, 0.360, 0.20, 0.075, "DCT 周期检测", "#dce9f7")
    _box(ax, 0.355, 0.455, 0.21, 0.075, "2D折叠+Inception", "#dce9f7", fs=7.6)
    _box(ax, 0.355, 0.360, 0.21, 0.075, "2D折叠+Inception", "#dce9f7", fs=7.6)
    _box(ax, 0.615, 0.408, 0.13, 0.075, "门控融合", "#dce9f7", fs=7.6)

    # branch B
    _box(ax, 0.09, 0.175, 0.20, 0.075, "自相关谱（FFT加速）", "#e4f2e4", fs=7.4)
    _box(ax, 0.355, 0.175, 0.21, 0.075, "Top-K延迟聚合", "#e4f2e4", fs=7.6)

    # branch labels
    ax.text(0.20, 0.565, "分支A：双域周期卷积", fontsize=8, ha="center", color="tab:blue")
    ax.text(0.20, 0.100, "分支B：自相关注意力", fontsize=8, ha="center", color="tab:green")

    _box(ax, 0.615, 0.275, 0.115, 0.075, "门控融合", "#f2e6f2", fs=7.6)

    # merge
    _box(ax, 0.30, 0.150, 0.16, 0.062, "+ 残差 / LayerNorm", "#f5f0e1", fs=7.6)

    # right: head
    _box(ax, 0.30, 0.030, 0.40, 0.062, "预测头 → $\\hat{y}_{T+1:T+H}$", "#eef3fb")

    # arrows: pipeline
    _arrow(ax, (0.50, 0.915), (0.50, 0.870))
    _arrow(ax, (0.50, 0.805), (0.50, 0.762))
    _arrow(ax, (0.50, 0.700), (0.50, 0.655))

    # split into branches
    _arrow(ax, (0.42, 0.650), (0.19, 0.530))
    _arrow(ax, (0.58, 0.650), (0.19, 0.250))

    # conv branch internal
    _arrow(ax, (0.19, 0.455), (0.19, 0.435))
    _arrow(ax, (0.19, 0.530), (0.355, 0.493))
    _arrow(ax, (0.19, 0.397), (0.355, 0.397))
    _arrow(ax, (0.565, 0.493), (0.615, 0.455))
    _arrow(ax, (0.565, 0.397), (0.615, 0.435))

    # attn branch internal
    _arrow(ax, (0.19, 0.250), (0.19, 0.212))
    _arrow(ax, (0.29, 0.212), (0.355, 0.212))
    _arrow(ax, (0.565, 0.212), (0.672, 0.275))

    # gate outputs into residual
    _arrow(ax, (0.672, 0.408), (0.46, 0.181))
    _arrow(ax, (0.672, 0.275), (0.46, 0.181))
    _arrow(ax, (0.38, 0.150), (0.38, 0.092))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_parallel.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_stl()
    print("图1 完成")
    fig_domain()
    print("图2 完成")
    fig_arch()
    print("图3 完成")
    print("输出目录:", os.path.abspath(OUT))
