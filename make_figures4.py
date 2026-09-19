# -*- coding: utf-8 -*-
"""Overall architecture of PA-STL-DualTimesNet (Fig. 2), simplified.

Input side collapsed to Input -> STL Decomposition; output side collapsed to a
single Fully connected layer.  The block keeps its internal structure, since
that is where the two contributions live: FFT and DCT detection run in PARALLEL,
and Branch B is the Autoformer encoder.

Run: python make_figures4.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.unicode_minus"] = False

OUT = "figures"
C_IO, C_A, C_B, C_F = "#E8EEF7", "#D6E7F7", "#DCEFDC", "#F5E6F5"


def box(ax, x, y, w, h, text, fc, fs=8.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.007",
                                fc=fc, ec="black", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.3,
            fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, ls="-", lw=1.0, color="black", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                 lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle="arc3,rad=%s" % rad))


def line(ax, xs, ys, ls="-", lw=1.0, color="black"):
    ax.plot(xs, ys, ls=ls, lw=lw, color=color, solid_capstyle="round")


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.8, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- input ----------------
    box(ax, 0.32, 0.905, 0.36, 0.052, "Input $x$   $[B,\\ T_{in},\\ C]$", C_IO, bold=True)
    box(ax, 0.32, 0.812, 0.36, 0.052, "STL Decomposition", C_IO)
    arrow(ax, (0.50, 0.905), (0.50, 0.864))

    # ---------------- backbone ----------------
    ax.add_patch(FancyBboxPatch((0.035, 0.255), 0.930, 0.495,
                                boxstyle="round,pad=0.007",
                                fc="#FCFCFE", ec="black", lw=1.4, ls="--"))
    ax.text(0.055, 0.712, "DualDomainTimesBlock  $\\times\\ N$", fontsize=9.8,
            va="center", fontweight="bold")

    ax.text(0.055, 0.674, "Branch A : dual-domain period convolution",
            fontsize=8.5, va="center", color="#1F4E79")
    box(ax, 0.075, 0.578, 0.160, 0.062, "FFT period\ndetection", C_A, fs=7.8)
    box(ax, 0.075, 0.478, 0.160, 0.062, "DCT period\ndetection", C_A, fs=7.8)
    box(ax, 0.270, 0.578, 0.165, 0.062, "Fold + Inception", C_A, fs=7.8)
    box(ax, 0.270, 0.478, 0.165, 0.062, "Fold + Inception", C_A, fs=7.8)
    arrow(ax, (0.235, 0.609), (0.270, 0.609))
    arrow(ax, (0.235, 0.509), (0.270, 0.509))

    ax.text(0.055, 0.432, "Branch B : Autoformer encoder",
            fontsize=8.5, va="center", color="#1F6B1F")
    box(ax, 0.075, 0.335, 0.200, 0.062, "Auto-Correlation\n(Roll aggregation)",
        C_B, fs=7.6)
    box(ax, 0.310, 0.335, 0.125, 0.062, "FFN", C_B, fs=7.8)
    arrow(ax, (0.275, 0.366), (0.310, 0.366))

    # fusion
    box(ax, 0.475, 0.475, 0.140, 0.145, "Concat\n$\\rightarrow$ FC", C_F, fs=8.3)
    box(ax, 0.650, 0.523, 0.080, 0.062, "$\\oplus$", C_F, fs=11)
    box(ax, 0.765, 0.512, 0.180, 0.084, "LayerNorm", C_F, fs=8.6)
    arrow(ax, (0.435, 0.609), (0.475, 0.575), rad=-0.12)
    arrow(ax, (0.435, 0.509), (0.475, 0.500), rad=0.12)
    arrow(ax, (0.435, 0.366), (0.540, 0.475), rad=0.18)
    arrow(ax, (0.615, 0.552), (0.650, 0.554))
    arrow(ax, (0.730, 0.554), (0.765, 0.554))

    # input fan-out: one bus feeds FFT, DCT and the encoder branch
    line(ax, [0.50, 0.50], [0.812, 0.782])
    line(ax, [0.50, 0.048], [0.782, 0.782])
    line(ax, [0.048, 0.048], [0.782, 0.366])
    arrow(ax, (0.048, 0.609), (0.075, 0.609))
    arrow(ax, (0.048, 0.509), (0.075, 0.509))
    arrow(ax, (0.048, 0.366), (0.075, 0.366))

    # residual
    line(ax, [0.700, 0.700], [0.782, 0.600], ls=":")
    arrow(ax, (0.700, 0.600), (0.690, 0.585), ls=":")
    ax.text(0.708, 0.640, "residual", fontsize=7.2, color="#555555")

    # block output
    line(ax, [0.855, 0.855], [0.512, 0.205])
    line(ax, [0.855, 0.50], [0.205, 0.205])
    arrow(ax, (0.50, 0.205), (0.50, 0.170))

    # ---------------- output ----------------
    box(ax, 0.28, 0.098, 0.44, 0.072,
        "Fully connected layer", C_IO, fs=9.0, bold=True)
    arrow(ax, (0.50, 0.098), (0.50, 0.048))
    box(ax, 0.30, -0.030, 0.40, 0.078,
        "Output $\\hat{y}$   $[B,\\ T_{out},\\ C]$", C_IO, fs=8.6, bold=True)

    ax.text(0.50, 1.0, "Architecture of PA-STL-DualTimesNet",
            ha="center", va="top", fontsize=11.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_arch()
    print("Fig.2 architecture done (simplified)")
