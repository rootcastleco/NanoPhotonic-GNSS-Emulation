# Chip-Scale Optical Clock (CSOC) GNSS Emulation Suite

[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--2807--3264-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0009-2807-3264)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Batuhan AYRIBAŞ

A stochastic Python emulation suite designed to mathematically demonstrate the superlinear positioning improvement ($\eta_p \propto \eta_c^{1.5}$) achieved by integrating Chip-Scale Optical Clocks (CSOC) into Global Navigation Satellite System (GNSS) receivers.

This repository contains the source code, Extended Kalman Filter (EKF) implementation, and Monte Carlo simulation framework used to generate the experimental results for the associated manuscript.

## Overview

Traditional GNSS receivers rely on Temperature-Compensated Crystal Oscillators (TCXOs). During periods of signal degradation (e.g., urban canyons, multi-path environments, spoofing attempts), the EKF's geometric trilateration degrades as the local oscillator wanders. 

This project simulates a highly constrained urban-canyon scenario to prove that replacing a TCXO ($h_0 \approx 10^{-9}$) with a nanophotonic CSOC ($h_0 \approx 10^{-10}$) prevents the EKF from incorrectly attributing geometric residuals to clock drift. Because Geometric Dilution of Precision (GDOP) acts as a multi-path multiplier, stripping away the pseudorange timing ambiguity allows the trilateration equations to converge aggressively, yielding a superlinear improvement in positioning accuracy.

## Features

1. **Stochastic Clock Noise Modeling (`clock_noise.py`)** 
   - Time-domain Allan variance synthesis using $h_0, h_{-1}, h_{-2}$ coefficients.
   - Accurately models WFM, FFM, and RWFM for both TCXO and CSOC oscillators.
2. **Synthetic Satellite Geometry (`satellite_geometry.py`)**
   - 8-satellite constellation with realistic line-of-sight vectors.
   - Elevation-dependent pseudorange and Doppler measurement generation.
   - Mathematical coupling of clock instability to correlation jitter.
3. **Optical-Clock-Aware EKF (`gnss_ekf.py`)**
   - 8-state Extended Kalman Filter (3D Position, 3D Velocity, Clock Bias, Clock Drift).
   - Dynamically parameterized process noise matrix ($Q_{clock}$) derived from specific oscillator Allan coefficients.
4. **Monte Carlo Runner & Figure Generation (`run_emulation.py`, `generate_figures.py`)**
   - Headless simulation pipeline performing 50-trial Monte Carlo sweeps across 7 clock quality scale factors.
   - Generates publication-ready PDF figures (Allan deviation, convergence timeseries, 2D error ellipses, and superlinear scaling).

## Installation

Ensure you have Python 3.8+ installed. Clone this repository and install the numerical dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/NanoPhotonic-GNSS-Emulation.git
cd NanoPhotonic-GNSS-Emulation
pip install -r emulation/requirements.txt
```

## Usage

To reproduce the study's experimental results and generate the figures:

```bash
# Run the Monte Carlo stochastic emulation (takes ~30 seconds)
python3 emulation/run_emulation.py

# Generate the publication-quality LaTeX figures
python3 emulation/generate_figures.py
```

The output data will be saved to the `results/` directory as `.npz` arrays, and the corresponding plots will be saved to the `figures/` directory as `.pdf` and `.png` files.

---
*For questions or collaborations, please view my research profile on [ORCID](https://orcid.org/0009-0009-2807-3264).*
