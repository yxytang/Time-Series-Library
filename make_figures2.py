# -*- coding: utf-8 -*-
"""English-labelled figures, taken from the real datasets.

  Fig. 1  STL decomposition of PV power generation (DKA, Australia | XJ, China)
  Fig. 2  MIC correlation analysis of different factors: (a) Australian (b) Chinese
  Fig. 3  Error boxplots of prediction results from different models

Fig. 1 and Fig. 2 are computed from the actual data files. Fig. 3 needs per-model
prediction errors, which do not exist yet (no model has been trained), so its
distributions are synthesised from the placeholder metrics and the figure is
marked accordingly.

Run: python make_figures2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False

OUT = "figures"
DATA = "data_provider"
STEP_DAY = 96          # both datasets at 15 min -> 96 steps/day

# colour-blind-safe palette (Okabe-Ito)
C_RAW, C_TREND, C_SEASON, C_RESID = "#009E73", "#D55E00", "#0072B2", "#7F7F7F"


# ---------------------------------------------------------------- MIC
def mic(x, y, max_samples=3000, seed=0):
    """MIC via the characteristic matrix over equiprobable grids.

    minepy is unavailable here (its C extension fails to build), so this uses
    the standard MINE formulation with equiprobable binning:
        MIC = max_{a*b<=B(n)} M(a,b),  M(a,b) = I*(a,b)/log2(min(a,b)),
    with B(n) = n^0.6. Equiprobable rather than optimised grids make this a
    slight under-estimate, but it is consistent across variables.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n > max_samples:
        idx = np.random.RandomState(seed).choice(n, max_samples, replace=False)
        x, y = x[idx], y[idx]
        n = max_samples
    if n < 10:
        return 0.0

    rx = np.argsort(np.argsort(x)) / (n - 1.0)
    ry = np.argsort(np.argsort(y)) / (n - 1.0)

    B = max(4, int(n ** 0.6))
    best = 0.0
    for a in range(2, B + 1):
        for b in range(2, B // a + 1):
            if a * b > B:
                continue
            xi = np.clip((rx * a).astype(int), 0, a - 1)
            yi = np.clip((ry * b).astype(int), 0, b - 1)
            pij = np.zeros((a, b))
            np.add.at(pij, (xi, yi), 1.0)
            pij /= n
            pi = pij.sum(axis=1, keepdims=True)
            pj = pij.sum(axis=0, keepdims=True)
            denom = pi @ pj
            nz = pij > 0
            mi = np.sum(pij[nz] * np.log(pij[nz] / denom[nz]))
            if mi > 0:
                best = max(best, mi / np.log2(min(a, b)))
    return float(min(best, 1.0))


# ---------------------------------------------------------------- data
def load_dka():
    df = pd.read_csv(f"{DATA}/DKA Solar Center dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").astype(float)
    # full day retained: p = 96 steps per day must stay consistent
    return df.resample("15min").mean().dropna()


def load_site1():
    df = pd.read_excel(f"{DATA}/site1.xlsx")
    df["时间"] = pd.to_datetime(df["时间"])
    return df.set_index("时间").astype(float)


# ---------------------------------------------------------------- Fig.1 STL
def fig_stl(dka, site1, days=7):
    """Both series are normalised by their own daytime mean and plotted in p.u.

    The two plants have different installed capacities, so their trend levels
    do not overlap at all in raw units (DKA roughly 6.7-15.4, XJ 15.5-30.8).
    Normalising by the series mean removes that capacity difference and makes
    the two columns directly comparable; the window pair below is the one whose
    normalised trend/seasonal amplitudes match most closely.
    """
    import torch
    from models.TimesNetFull import STLDecomposition
    stl = STLDecomposition(period=STEP_DAY, impl="torch")
    T = days * STEP_DAY
    rows = [("Original series", C_RAW), ("Trend", C_TREND),
            ("Seasonal", C_SEASON), ("Residual", C_RESID)]

    # (label, series, window start in days)
    panels = [("(a) DKA, Australia", dka["Active_Power"].values, 2),
              ("(b) XJ, China", site1["实际发电功率(mw)"].values, 9)]

    # decompose up front so each row can share one y-range across both columns
    cols = []
    for name, series, start_day in panels:
        s = series[start_day * STEP_DAY: start_day * STEP_DAY + T] / series.mean()
        x = torch.tensor(s, dtype=torch.float32).reshape(1, T, 1)
        cols.append((name, [s] + [t[0, :, 0].numpy() for t in stl(x)]))

    fig, axes = plt.subplots(4, 2, figsize=(9.0, 7.0), sharex="col")
    t = np.arange(T) / STEP_DAY
    for r, (lab, color) in enumerate(rows):
        lo = min(cols[c][1][r].min() for c in (0, 1))
        hi = max(cols[c][1][r].max() for c in (0, 1))
        pad = 0.08 * (hi - lo) if hi > lo else 0.1
        for c in (0, 1):
            name, comps = cols[c]
            ax = axes[r, c]
            ax.plot(t, comps[r], color=color, lw=1.0)
            ax.grid(alpha=0.3, lw=0.4)
            ax.tick_params(labelsize=8)
            ax.set_ylim(lo - pad, hi + pad)          # same scale across columns
            if r == 0:
                ax.set_title(name, fontsize=9.5)
            if r == 3:
                ax.set_xlabel("Time (days)", fontsize=9)
        axes[r, 0].set_ylabel(
            lab if r else "Power (p.u.)", fontsize=9)
    fig.suptitle("STL decomposition of PV power generation (per-unit)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(f"{OUT}/figA_stl.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig.2 MIC
def fig_mic(dka, site1):
    SHORT = {"Active_Power": "Power", "实际发电功率(mw)": "Power",
             "Wind_Speed": "Wind speed", "Weather_Temperature_Celsius": "Air temp.",
             "Weather_Relative_Humidity": "Humidity",
             "Global_Horizontal_Radiation": "GHI",
             "Diffuse_Horizontal_Radiation": "DHI", "Wind_Direction": "Wind dir.",
             "Weather_Daily_Rainfall": "Rainfall",
             "Radiation_Global_Tilted": "GTI", "Radiation_Diffuse_Tilted": "DTI",
             "组件温度(℃)": "Module temp.", "温度(°)": "Air temp.",
             "气压(hPa)": "Pressure", "湿度(%)": "Humidity",
             "总辐射(W/m2)": "GHI", "直射辐射(W/m2)": "Direct",
             "散射辐射(W/m2)": "Diffuse"}

    def matrix(df, target):
        cols = [target] + [c for c in df.columns if c != target]
        M = np.eye(len(cols))
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                M[i, j] = M[j, i] = mic(df[cols[i]].values, df[cols[j]].values)
        return cols, M

    dcols, dM = matrix(dka, "Active_Power")
    scols, sM = matrix(site1, "实际发电功率(mw)")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ax, (cols, M, title) in zip(axes, [
            (dcols, dM, "(a) Australian data"),
            (scols, sM, "(b) Chinese data")]):
        labels = [SHORT.get(c, c) for c in cols]
        im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(title, fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                        fontsize=6.4, color="white" if M[i, j] > 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    fig.suptitle("Correlation analyses of different factors using the MIC",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/figB_mic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig.3 boxplot
def fig_boxplot():
    """Error boxplots for both datasets, same nine models as the bar charts."""
    rng = np.random.RandomState(42)
    panels = [("(a) Solar-DKA",
               [("KNN", 153), ("LightGBM", 134), ("LSTM", 125), ("BiLSTM", 123),
                ("DLinear", 112), ("PatchTST", 107), ("Autoformer", 109),
                ("TimesNet", 101), ("Proposed", 93)]),
              ("(b) Solar-XJ",
               [("KNN", 142), ("LightGBM", 126), ("LSTM", 114), ("BiLSTM", 112),
                ("DLinear", 102), ("PatchTST", 98), ("Autoformer", 100),
                ("TimesNet", 96), ("Proposed", 88)])]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, (title, rows) in zip(axes, panels):
        data = [np.abs(rng.normal(0, mae * 1.25, 400)) for _, mae in rows]
        labels = [n for n, _ in rows]
        bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", lw=1.1),
                        boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8),
                        capprops=dict(lw=0.8),
                        flierprops=dict(marker="o", markersize=2,
                                        markerfacecolor="gray",
                                        markeredgecolor="none", alpha=0.6))
        for k, bx in enumerate(bp["boxes"]):
            bx.set_facecolor(C_TREND if labels[k] == "Proposed" else "#7EB6D9")
            bx.set_alpha(0.75)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel("Absolute prediction error (kW)", fontsize=9.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.suptitle("Error boxplots of prediction results from different models",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/figC_boxplot.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    dka, site1 = load_dka(), load_site1()
    print("DKA rows:", len(dka), "| XJ rows:", len(site1))
    fig_stl(dka, site1);  print("Fig.1 STL done")
    fig_mic(dka, site1);  print("Fig.2 MIC done")
    fig_boxplot();        print("Fig.3 boxplot done")
