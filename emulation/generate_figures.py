#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════
Publication-Quality Figure Generator
══════════════════════════════════════════════════════════════════════════

Generates 4 PDF figures for the nanophotonic GNSS receiver paper:

  Fig 1 — Allan deviation (σ_y vs τ), log-log: TCXO vs CSOC
  Fig 2 — Horizontal position error convergence (time series)
  Fig 3 — 2D North/East error scatter with 95% error ellipses
  Fig 4 — Superlinear scaling (clock improvement vs position improvement)

Requirements: numpy, scipy, matplotlib

Usage:
    python3 emulation/generate_figures.py
"""

import numpy as np
import os
import sys

# Handle both module and script execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════
# Style Configuration — Publication Quality
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "text.usetex": False,            # Don't require LaTeX install
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
})

# Colour-blind-safe palette (Tol's vibrant)
C_TCXO = "#EE7733"     # orange
C_CSOC = "#0077BB"      # blue
C_FILL_TCXO = "#EE7733"
C_FILL_CSOC = "#0077BB"
C_THEORY = "#999999"    # grey


def get_paths():
    """Get results and figures directory paths."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base, "results")
    figures_dir = os.path.join(base, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return results_dir, figures_dir


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Allan Deviation
# ═══════════════════════════════════════════════════════════════════════

def figure_1_allan_deviation():
    """Log-log Allan deviation plot: TCXO vs CSOC."""
    results_dir, figures_dir = get_paths()
    data = np.load(os.path.join(results_dir, "allan_deviation.npz"))

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Theoretical curves
    ax.loglog(data["TCXO_taus_theory"], data["TCXO_adev_theory"],
              color=C_TCXO, linestyle="-", linewidth=1.5,
              label="TCXO (theoretical)")
    ax.loglog(data["CSOC_taus_theory"], data["CSOC_adev_theory"],
              color=C_CSOC, linestyle="-", linewidth=1.5,
              label="CSOC (theoretical)")

    # Measured from simulation
    ax.loglog(data["TCXO_taus_meas"], data["TCXO_adev_meas"],
              "o", color=C_TCXO, markersize=2.5, alpha=0.5,
              label="TCXO (simulated)")
    ax.loglog(data["CSOC_taus_meas"], data["CSOC_adev_meas"],
              "s", color=C_CSOC, markersize=2.5, alpha=0.5,
              label="CSOC (simulated)")

    # Annotate improvement
    tau_anno = 10.0
    tcxo_at_10 = np.interp(tau_anno, data["TCXO_taus_theory"], data["TCXO_adev_theory"])
    csoc_at_10 = np.interp(tau_anno, data["CSOC_taus_theory"], data["CSOC_adev_theory"])
    ax.annotate("",
                xy=(tau_anno, csoc_at_10), xytext=(tau_anno, tcxo_at_10),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.8))
    ratio = tcxo_at_10 / csoc_at_10
    ax.text(tau_anno * 1.5, np.sqrt(tcxo_at_10 * csoc_at_10),
            f"~{ratio:.0f}×",
            fontsize=9, ha="left", va="center")

    ax.set_xlabel(r"Averaging Time $\tau$ [s]")
    ax.set_ylabel(r"Allan Deviation $\sigma_y(\tau)$")
    ax.set_title("Clock Stability: TCXO vs. Nanophotonic CSOC")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(0.5, 2000)

    path = os.path.join(figures_dir, "fig1_allan_deviation.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] {path}")
    # Also save PNG for preview
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    ax2.loglog(data["TCXO_taus_theory"], data["TCXO_adev_theory"],
               color=C_TCXO, linestyle="-", linewidth=1.5, label="TCXO (theoretical)")
    ax2.loglog(data["CSOC_taus_theory"], data["CSOC_adev_theory"],
               color=C_CSOC, linestyle="-", linewidth=1.5, label="CSOC (theoretical)")
    ax2.loglog(data["TCXO_taus_meas"], data["TCXO_adev_meas"],
               "o", color=C_TCXO, markersize=2.5, alpha=0.5, label="TCXO (simulated)")
    ax2.loglog(data["CSOC_taus_meas"], data["CSOC_adev_meas"],
               "s", color=C_CSOC, markersize=2.5, alpha=0.5, label="CSOC (simulated)")
    ax2.annotate("", xy=(tau_anno, csoc_at_10), xytext=(tau_anno, tcxo_at_10),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=0.8))
    ax2.text(tau_anno * 1.5, np.sqrt(tcxo_at_10 * csoc_at_10),
             f"~{ratio:.0f}×", fontsize=9, ha="left", va="center")
    ax2.set_xlabel(r"Averaging Time $\tau$ [s]")
    ax2.set_ylabel(r"Allan Deviation $\sigma_y(\tau)$")
    ax2.set_title("Clock Stability: TCXO vs. Nanophotonic CSOC")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, which="both", alpha=0.2)
    ax2.set_xlim(0.5, 2000)
    png_path = os.path.join(figures_dir, "fig1_allan_deviation.png")
    fig2.savefig(png_path)
    plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Convergence Time Series
# ═══════════════════════════════════════════════════════════════════════

def figure_2_convergence():
    """Horizontal position error vs time from cold start."""
    results_dir, figures_dir = get_paths()
    tcxo = np.load(os.path.join(results_dir, "tcxo_results.npz"))
    csoc = np.load(os.path.join(results_dir, "csoc_results.npz"))

    epochs = np.arange(len(tcxo["h_median"]))

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # TCXO
    ax.semilogy(epochs, tcxo["h_median"], color=C_TCXO, linewidth=1.5,
                label="TCXO (median)")
    ax.fill_between(epochs, tcxo["h_p25"], tcxo["h_p75"],
                     color=C_TCXO, alpha=0.15, label="TCXO (25–75%)")

    # CSOC
    ax.semilogy(epochs, csoc["h_median"], color=C_CSOC, linewidth=1.5,
                label="CSOC (median)")
    ax.fill_between(epochs, csoc["h_p25"], csoc["h_p75"],
                     color=C_CSOC, alpha=0.15, label="CSOC (25–75%)")

    # Reference lines
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.text(N_EPOCHS * 0.85, 1.15, "1 m", fontsize=8, color="gray")
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.text(N_EPOCHS * 0.85, 0.057, "5 cm (Micro-RTK)", fontsize=8, color="gray")

    ax.set_xlabel("Epoch [s]")
    ax.set_ylabel("Horizontal Position Error [m]")
    ax.set_title("Cold-Start Convergence: TCXO vs. CSOC")
    ax.legend(loc="upper right", framealpha=0.9, ncol=1)
    ax.set_xlim(0, len(epochs) - 1)
    ax.set_ylim(bottom=0.005)
    ax.grid(True, which="both", alpha=0.2)

    path = os.path.join(figures_dir, "fig2_convergence.pdf")
    fig.savefig(path)
    png_path = os.path.join(figures_dir, "fig2_convergence.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  [✓] {path}")


N_EPOCHS = 600  # must match run_emulation.py


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: 2D Error Scatter with Ellipses
# ═══════════════════════════════════════════════════════════════════════

def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """
    Draw a confidence ellipse based on sample data.
    n_std=2.0 → ~95% confidence for 2D Gaussian.
    """
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_std * np.sqrt(eigvals[0])
    height = 2 * n_std * np.sqrt(eigvals[1])

    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                       width=width, height=height,
                       angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return width, height


def figure_3_error_ellipse():
    """2D North/East scatter with 95% confidence ellipses."""
    results_dir, figures_dir = get_paths()
    tcxo = np.load(os.path.join(results_dir, "tcxo_results.npz"))
    csoc = np.load(os.path.join(results_dir, "csoc_results.npz"))

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    # Scatter (subsample for clarity)
    n_show = min(2000, len(tcxo["e_final"]))
    idx_t = np.random.choice(len(tcxo["e_final"]), n_show, replace=False)
    idx_c = np.random.choice(len(csoc["e_final"]), n_show, replace=False)

    ax.scatter(tcxo["e_final"][idx_t], tcxo["n_final"][idx_t],
               s=3, alpha=0.15, color=C_TCXO, label="TCXO", rasterized=True)
    ax.scatter(csoc["e_final"][idx_c], csoc["n_final"][idx_c],
               s=3, alpha=0.25, color=C_CSOC, label="CSOC", rasterized=True)

    # 95% confidence ellipses
    w_t, h_t = confidence_ellipse(tcxo["e_final"], tcxo["n_final"], ax, n_std=2.0,
                                   edgecolor=C_TCXO, linewidth=1.5,
                                   facecolor="none", linestyle="-",
                                   label="TCXO 95%")
    w_c, h_c = confidence_ellipse(csoc["e_final"], csoc["n_final"], ax, n_std=2.0,
                                   edgecolor=C_CSOC, linewidth=1.5,
                                   facecolor="none", linestyle="-",
                                   label="CSOC 95%")

    # Annotate ellipse sizes
    rms_t = np.sqrt(np.mean(tcxo["e_final"]**2 + tcxo["n_final"]**2))
    rms_c = np.sqrt(np.mean(csoc["e_final"]**2 + csoc["n_final"]**2))

    # Dynamic axis limits
    max_extent = max(abs(tcxo["e_final"]).max(), abs(tcxo["n_final"]).max()) * 1.3
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)

    ax.set_xlabel("East Error [m]")
    ax.set_ylabel("North Error [m]")
    ax.set_title("Post-Convergence Position Error Distribution")
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.axvline(0, color="gray", linewidth=0.4)
    ax.grid(True, alpha=0.2)

    # Text box with stats
    textstr = (f"TCXO: RMS = {rms_t:.3f} m\n"
               f"CSOC: RMS = {rms_c:.3f} m\n"
               f"Ratio: {rms_t/max(rms_c, 1e-9):.1f}×")
    props = dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.9)
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=props)

    ax.legend(loc="lower left", framealpha=0.9, markerscale=3)

    path = os.path.join(figures_dir, "fig3_error_ellipse.pdf")
    fig.savefig(path)
    png_path = os.path.join(figures_dir, "fig3_error_ellipse.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  [✓] {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Superlinear Scaling
# ═══════════════════════════════════════════════════════════════════════

def figure_4_superlinear():
    """Clock improvement ratio vs. position improvement ratio."""
    results_dir, figures_dir = get_paths()
    swept = np.load(os.path.join(results_dir, "swept_results.npz"))

    scales = swept["scales"]
    rms_values = swept["rms_median"]

    # Improvement ratios (relative to worst = scale 1.0, i.e. TCXO)
    clock_improvement = 1.0 / scales  # e.g., scale=0.1 → 10× better clock
    pos_improvement = rms_values[0] / rms_values  # relative to TCXO RMS

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Data points
    ax.loglog(clock_improvement, pos_improvement,
              "o-", color=C_CSOC, linewidth=1.5, markersize=6,
              label="Emulation result", zorder=3)

    # Linear reference
    ci_range = np.logspace(0, np.log10(clock_improvement.max()), 50)
    ax.loglog(ci_range, ci_range,
              "--", color=C_THEORY, linewidth=1.0,
              label="Linear (1:1)")

    # Quadratic reference
    ax.loglog(ci_range, ci_range**1.5,
              ":", color="#CC3311", linewidth=1.0,
              label=r"Superlinear ($\propto x^{1.5}$)")

    # Annotate key point
    idx_10x = np.argmin(np.abs(clock_improvement - 10.0))
    if idx_10x < len(pos_improvement):
        ax.annotate(
            f'{pos_improvement[idx_10x]:.0f}× position\nimprovement',
            xy=(clock_improvement[idx_10x], pos_improvement[idx_10x]),
            xytext=(clock_improvement[idx_10x] * 0.2, pos_improvement[idx_10x] * 2),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.9)
        )

    ax.set_xlabel("Clock Stability Improvement Factor")
    ax.set_ylabel("Position Accuracy Improvement Factor")
    ax.set_title("Superlinear Scaling: Clock → Position Performance")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)

    # Highlight the superlinear region
    ax.fill_between(ci_range, ci_range, ci_range**2,
                     alpha=0.05, color=C_CSOC,
                     label="_nolegend_")

    path = os.path.join(figures_dir, "fig4_superlinear.pdf")
    fig.savefig(path)
    png_path = os.path.join(figures_dir, "fig4_superlinear.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  [✓] {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NanoPhotonic-GNSS — Publication Figure Generator           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    results_dir, figures_dir = get_paths()

    # Check that results exist
    required = ["allan_deviation.npz", "tcxo_results.npz",
                "csoc_results.npz", "swept_results.npz"]
    for f in required:
        if not os.path.exists(os.path.join(results_dir, f)):
            print(f"  [✗] Missing {f} — run emulation first:")
            print(f"      python3 emulation/run_emulation.py")
            return

    print(f"\n  Results: {results_dir}")
    print(f"  Output:  {figures_dir}\n")

    figure_1_allan_deviation()
    figure_2_convergence()
    figure_3_error_ellipse()
    figure_4_superlinear()

    print(f"\n  All figures saved to: {figures_dir}")
    print(f"  Include in LaTeX with:")
    print(r"    \includegraphics[width=\columnwidth]{figures/fig1_allan_deviation.pdf}")


if __name__ == "__main__":
    main()
