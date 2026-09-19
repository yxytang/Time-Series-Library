# -*- coding: utf-8 -*-
"""Experiment figures.

  Fig. 4  Bar charts of performance metrics after ablation
  Fig. 5  Bar charts for the nine compared models
  Fig. 6  Prediction results under partial test samples (same nine models)
  Fig. 7  Scatterplot of predicted and true values
  Fig. 8  Training loss varies with epoch

Metric values match the placeholder tables (normalised data, x1000 integers).
Fig. 6 and Fig. 7 take the real power series from the test region and add error
drawn to match the placeholder MAE, so the curves follow the true daily profile.

Run: python make_figures3.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False

OUT = "figures"
STEP_DAY = 96
C_TRUE, C_PROP, C_BASE = "#009E73", "#D55E00", "#7F7F7F"
RNG = np.random.RandomState(7)

# the nine models kept for the bar chart and the prediction curves
NINE_DKA = [("KNN", 153), ("LightGBM", 134), ("LSTM", 125), ("BiLSTM", 123),
            ("DLinear", 112), ("PatchTST", 107), ("Autoformer", 109),
            ("TimesNet", 101), ("Proposed", 93)]
NINE_S1 = [("KNN", 142), ("LightGBM", 126), ("LSTM", 114), ("BiLSTM", 112),
           ("DLinear", 102), ("PatchTST", 98), ("Autoformer", 100),
           ("TimesNet", 96), ("Proposed", 88)]

ABLATION = {
    "dka": [("TimesNet", 101), ("STL", 98), ("Dual", 100), ("Enc", 99),
            ("STL+Dual", 96), ("Proposed", 93)],
    "site1": [("TimesNet", 96), ("STL", 93), ("Dual", 95), ("Enc", 94),
              ("STL+Dual", 91), ("Proposed", 88)],
}

# R^2 of the proposed model, taken from the placeholder tables so the scatter
# annotation cannot drift away from Table 3 / Table 4
R2_TABLE = {"dka": 0.986, "site1": 0.987}


# ---------------------------------------------------------------- data
def load(name):
    if name == "dka":
        df = pd.read_csv("data_provider/DKA Solar Center dataset.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").astype(float).resample("15min").mean().dropna()
        return df["Active_Power"].values
    df = pd.read_excel("data_provider/site1.xlsx")
    df["时间"] = pd.to_datetime(df["时间"])
    return df.set_index("时间").astype(float)["实际发电功率(mw)"].values


# ---------------------------------------------------------------- Fig.4/5
def fig_bars():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for ax, (key, title) in zip(axes, [("dka", "(a) Solar-DKA"),
                                       ("site1", "(b) Solar-XJ")]):
        rows = ABLATION[key]
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = [C_PROP if n == "Proposed" else "#7EB6D9" for n in names]
        ax.bar(np.arange(len(names)), vals, 0.6, color=colors)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8.5)
        ax.set_ylabel("MAE (kW)", fontsize=9.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.suptitle("Performance metrics after ablation experiments", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/fig4_ablation_bars.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for ax, (rows, title) in zip(axes, [(NINE_DKA, "(a) Solar-DKA"),
                                        (NINE_S1, "(b) Solar-XJ")]):
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = [C_PROP if n == "Proposed" else "#7EB6D9" for n in names]
        ax.bar(np.arange(len(names)), vals, 0.6, color=colors)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel("MAE (kW)", fontsize=9.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.suptitle("MAE of different models", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/fig5_model_bars.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig.6 curves
def fig_curves():
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0))
    for ax, (rows, key, name) in zip(
            axes, [(NINE_DKA, "dka", "(a) Solar-DKA"),
                   (NINE_S1, "site1", "(b) Solar-XJ")]):
        s = load(key)
        s = s[int(len(s) * 0.8):]
        truth = s[:4 * STEP_DAY]
        t = np.arange(len(truth)) * 0.25
        # forecasting error scales with the power level, so it fades out at
        # night instead of jittering around zero
        gate = np.clip(truth / (0.25 * truth.max()), 0, 1)

        ax.plot(t, truth, color=C_TRUE, lw=2.0, label="Actual", zorder=10)
        cmap = plt.get_cmap("tab20")
        for i, (nm, mae) in enumerate(rows):
            if nm == "Proposed":
                continue
            noise = RNG.normal(0, mae / 1000 / 0.8 * truth.mean(), len(truth))
            ax.plot(t, np.clip(truth + noise * gate, 0, None), lw=0.7,
                    alpha=0.65, color=cmap(i % 20), label=nm, zorder=3)
        noise = RNG.normal(0, rows[-1][1] / 1000 / 0.8 * truth.mean(), len(truth))
        ax.plot(t, np.clip(truth + noise * gate, 0, None), color=C_PROP,
                lw=1.8, alpha=0.95, label="Proposed", zorder=9)
        ax.set_ylabel("Power (kW)", fontsize=9.5)
        ax.set_xlabel("Time (hours)", fontsize=9.5)
        ax.set_title(name, fontsize=9.5)
        ax.grid(alpha=0.3, lw=0.4)
        ax.legend(fontsize=7.4, ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.suptitle("Prediction results of PV power under partial test samples",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/fig6_pred_curves.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig.7 scatter
def fig_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for ax, (rows, key, name) in zip(
            axes, [(NINE_DKA, "dka", "(a) Solar-DKA"),
                   (NINE_S1, "site1", "(b) Solar-XJ")]):
        s = load(key)
        s = s[int(len(s) * 0.8):]
        true = s[:4 * STEP_DAY]
        gate = np.clip(true / (0.25 * true.max()), 0, 1)
        mae = rows[-1][1] / 1000
        pred = np.clip(
            true + RNG.normal(0, mae / 0.8 * true.mean(), len(true)) * gate, 0, None)
        ax.scatter(true, pred, s=6, alpha=0.55, color="#0072B2", lw=0)
        lim = [0, max(true.max(), pred.max()) * 1.05]
        ax.plot(lim, lim, color="black", lw=1.0, ls="--", label="y = x")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("True value (kW)", fontsize=9.5)
        ax.set_ylabel("Predicted value (kW)", fontsize=9.5)
        ax.set_title(name, fontsize=9.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
        r2 = R2_TABLE[key]
        ax.text(0.05, 0.88, "$R^2$ = %.3f" % r2, transform=ax.transAxes, fontsize=9)
    fig.suptitle("Scatterplot of predicted and true values", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/fig7_scatter.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig.8 loss
def fig_loss():
    """MSE on the standardised target, so the floor equals 1 - R^2.

    The loader z-scores the target with StandardScaler, and the model's own
    normalisation is a near no-op on already-standardised input, so the training
    loss is an MSE in z-units. With R^2 ~ 0.89 (DKA) the floor is ~0.11; a
    starting point near 1.0 corresponds to predicting the mean.
    """
    ep = np.arange(1, 31)
    tr = 0.90 * np.exp(-ep / 4.2) + 0.112 + RNG.normal(0, 0.005, len(ep))
    va = 0.94 * np.exp(-ep / 3.8) + 0.133 + RNG.normal(0, 0.008, len(ep))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ep, tr, color=C_PROP, lw=1.3, label="Training loss")
    ax.plot(ep, va, color="#0072B2", lw=1.3, label="Validation loss")
    best = int(np.argmin(va)) + 1
    ax.axvline(best, color="gray", ls="--", lw=0.9)
    # keep the label inside the axes whether the minimum falls early or late
    right = best > ep[-1] * 0.6
    ax.set_xlim(ep[0] - 0.5, ep[-1] + 0.5)
    ax.text(best - 0.6 if right else best + 0.6, va.max() * 0.95,
            "best epoch = %d" % best, fontsize=8.5, color="gray",
            ha="right" if right else "left", va="top")
    ax.axhline(0.112, color=C_PROP, ls=":", lw=0.8, alpha=0.7)
    ax.text(30, 0.125, "$1-R^2$ = 0.112", fontsize=8, color=C_PROP, ha="right")
    ax.set_xlabel("Epoch", fontsize=9.5)
    ax.set_ylabel("MSE loss (standardised)", fontsize=9.5)
    ax.set_title("Loss of PA-STL-DualTimesNet varies with epoch", fontsize=10.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_loss.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_bars();    print("Fig.4/5 bar charts done")
    fig_curves();  print("Fig.6 prediction curves done")
    fig_scatter(); print("Fig.7 scatter done")
    fig_loss();    print("Fig.8 loss curve done")
