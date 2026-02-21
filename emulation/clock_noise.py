#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════
Clock Noise Generator — Allan-Deviation-Consistent Oscillator Models
══════════════════════════════════════════════════════════════════════════

Generates time-domain clock bias and drift processes that reproduce
target Allan deviation σ_y(τ) signatures for two oscillator classes:

  • TCXO  — Temperature-Compensated Crystal Oscillator (commodity)
  • CSOC  — Chip-Scale Optical Clock (nanophotonic, ~10× better)

Noise types modeled:
  h₀  → White Frequency Modulation  (WFM)   : σ_y(τ) ∝ τ^{-1/2}
  h₋₁ → Flicker Frequency Modulation (FFM)  : σ_y(τ) ∝ τ^0 (flat)
  h₋₂ → Random-Walk Frequency Mod.  (RWFM)  : σ_y(τ) ∝ τ^{+1/2}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════
# Clock Model Dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ClockModel:
    """
    Parameterized clock noise model based on power-law spectral coefficients.

    The fractional frequency PSD is:
        S_y(f) = h_0 + h_{-1}/f + h_{-2}/f^2

    Allan deviation contributions at averaging time τ:
        WFM:   σ_y(τ) = sqrt(h_0 / (2τ))
        FFM:   σ_y(τ) = sqrt(2 * ln(2) * h_{-1})       (τ-independent)
        RWFM:  σ_y(τ) = sqrt((2π²/3) * h_{-2} * τ)
    """
    name: str
    h0: float      # White FM coefficient
    h_neg1: float  # Flicker FM coefficient
    h_neg2: float  # Random-walk FM coefficient

    def allan_deviation(self, tau: np.ndarray) -> np.ndarray:
        """Compute Allan deviation σ_y(τ) from h-coefficients."""
        sigma_wfm_sq = self.h0 / (2.0 * tau)
        sigma_ffm_sq = 2.0 * np.log(2.0) * self.h_neg1 * np.ones_like(tau)
        sigma_rwfm_sq = (2.0 * np.pi**2 / 3.0) * self.h_neg2 * tau
        return np.sqrt(sigma_wfm_sq + sigma_ffm_sq + sigma_rwfm_sq)

    def generate_frequency_noise(self, N: int, dt: float,
                                  rng: Optional[np.random.Generator] = None
                                  ) -> np.ndarray:
        """
        Generate fractional frequency deviation y(t) time series.

        Parameters
        ----------
        N  : number of samples
        dt : sampling interval [s]
        rng: numpy random generator (for reproducibility)

        Returns
        -------
        y : (N,) array of fractional frequency deviations
        """
        if rng is None:
            rng = np.random.default_rng()

        y = np.zeros(N)

        # White FM: each sample is i.i.d. Gaussian
        sigma_wfm = np.sqrt(self.h0 / (2.0 * dt))
        y += rng.normal(0, sigma_wfm, N)

        # Flicker FM: approximate via 1/f filtering
        # Generate white noise and apply 1/sqrt(f) filter in frequency domain
        if self.h_neg1 > 0:
            white = rng.normal(0, 1.0, N)
            freqs = np.fft.rfftfreq(N, d=dt)
            freqs[0] = freqs[1]  # avoid division by zero
            # 1/f noise: multiply spectrum by 1/sqrt(f)
            spectrum = np.fft.rfft(white)
            flicker_filter = 1.0 / np.sqrt(freqs)
            flicker_filter[0] = 0  # zero DC
            spectrum *= flicker_filter
            flicker = np.fft.irfft(spectrum, n=N)
            # Scale to match target Allan deviation
            # At τ=1s, FFM contribution: σ_y = sqrt(2*ln2*h_{-1})
            target_std = np.sqrt(self.h_neg1)
            if np.std(flicker) > 0:
                flicker *= target_std / np.std(flicker)
            y += flicker

        # Random-walk FM: cumulative sum of white noise
        if self.h_neg2 > 0:
            sigma_rwfm = np.sqrt(self.h_neg2 * dt)
            rw_increments = rng.normal(0, sigma_rwfm, N)
            y += np.cumsum(rw_increments)

        return y

    def generate_clock_states(self, N: int, dt: float,
                               rng: Optional[np.random.Generator] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time-domain clock bias x(t) [seconds] and drift y(t).

        The clock bias satisfies dx/dt = y(t), so x(t) = ∫y(t')dt'.

        Parameters
        ----------
        N  : number of epochs
        dt : epoch interval [s]

        Returns
        -------
        bias  : (N,) clock bias in seconds
        drift : (N,) fractional frequency deviation (drift rate)
        """
        drift = self.generate_frequency_noise(N, dt, rng)
        bias = np.cumsum(drift) * dt
        return bias, drift


# ═══════════════════════════════════════════════════════════════════════
# Preset Oscillator Models
# ═══════════════════════════════════════════════════════════════════════

def tcxo_model() -> ClockModel:
    """
    Representative TCXO parameters.
    Target: σ_y(1s) ≈ 1e-9, σ_y(100s) ≈ 3e-10, σ_y(1000s) ≈ 5e-10
    """
    return ClockModel(
        name="TCXO",
        h0=2e-18,       # WFM: dominant at short τ
        h_neg1=5e-20,    # FFM: flicker floor
        h_neg2=1e-22,    # RWFM: long-term wander
    )


def csoc_model() -> ClockModel:
    """
    Nanophotonic Chip-Scale Optical Clock (CSOC) parameters.
    Target: σ_y(1s) ≈ 1e-10, σ_y(100s) ≈ 3e-11, σ_y(1000s) ≈ 5e-11
    ~10× improvement over TCXO across all τ.
    """
    return ClockModel(
        name="CSOC",
        h0=2e-20,        # 100× lower WFM
        h_neg1=5e-22,     # 100× lower FFM
        h_neg2=1e-24,     # 100× lower RWFM
    )


def intermediate_model(scale_factor: float) -> ClockModel:
    """
    Interpolated clock model between TCXO and CSOC.

    Parameters
    ----------
    scale_factor : 1.0 = TCXO, 0.01 = CSOC (h-coefficients scale linearly)
    """
    tcxo = tcxo_model()
    return ClockModel(
        name=f"Intermediate (×{scale_factor:.3f})",
        h0=tcxo.h0 * scale_factor,
        h_neg1=tcxo.h_neg1 * scale_factor,
        h_neg2=tcxo.h_neg2 * scale_factor,
    )


# ═══════════════════════════════════════════════════════════════════════
# Allan Deviation Computation from Time Series
# ═══════════════════════════════════════════════════════════════════════

def compute_overlapping_adev(phase_data: np.ndarray, dt: float,
                              taus: Optional[np.ndarray] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute overlapping Allan deviation from phase (time error) data.

    Parameters
    ----------
    phase_data : (N,) time error x(t) in seconds
    dt         : sampling interval [s]
    taus       : averaging times to evaluate (if None, auto-generated)

    Returns
    -------
    taus_used  : evaluated averaging times
    adev       : Allan deviation at each τ
    """
    N = len(phase_data)
    if taus is None:
        # Generate log-spaced tau values
        max_m = N // 3
        ms = np.unique(np.logspace(0, np.log10(max_m), 50).astype(int))
        ms = ms[ms >= 1]
        taus = ms * dt

    adev_list = []
    taus_used = []

    for tau in taus:
        m = int(round(tau / dt))
        if m < 1 or 2 * m >= N:
            continue

        # Overlapping ADEV
        phase = phase_data
        diffs = phase[2*m:] - 2*phase[m:-(m)] + phase[:-(2*m)]
        if len(diffs) == 0:
            continue

        adev_val = np.sqrt(np.mean(diffs**2) / (2.0 * (m * dt)**2))
        adev_list.append(adev_val)
        taus_used.append(tau)

    return np.array(taus_used), np.array(adev_list)


# ═══════════════════════════════════════════════════════════════════════
# Process Noise Matrices for EKF
# ═══════════════════════════════════════════════════════════════════════

def clock_process_noise_matrix(model: ClockModel, dt: float) -> np.ndarray:
    """
    Compute 2×2 process noise covariance for clock states [bias, drift].

    Based on the two-state clock model:
        bias_{k+1}  = bias_k + drift_k * dt + w_bias
        drift_{k+1} = drift_k + w_drift

    The variances are derived from h-coefficients:
        Var(w_bias)  ≈ h₀ * dt + h₋₂ * dt³/3  (time error variance)
        Var(w_drift) ≈ h₀/dt + h₋₂ * dt        (frequency error variance)
        Cov          ≈ h₋₂ * dt²/2

    Units: bias in meters (×c), drift in m/s (×c).
    """
    c = 299792458.0  # speed of light [m/s]

    q_bias = (model.h0 * dt + model.h_neg2 * dt**3 / 3.0) * c**2
    q_drift = (model.h0 / dt + model.h_neg2 * dt) * c**2
    q_cross = (model.h_neg2 * dt**2 / 2.0) * c**2

    Q_clock = np.array([
        [q_bias,  q_cross],
        [q_cross, q_drift],
    ])

    return Q_clock


if __name__ == "__main__":
    # Quick validation
    taus = np.logspace(0, 3, 100)

    tcxo = tcxo_model()
    csoc = csoc_model()

    print("TCXO Allan Dev @ 1s:    {:.2e}".format(tcxo.allan_deviation(np.array([1.0]))[0]))
    print("TCXO Allan Dev @ 100s:  {:.2e}".format(tcxo.allan_deviation(np.array([100.0]))[0]))
    print("CSOC Allan Dev @ 1s:    {:.2e}".format(csoc.allan_deviation(np.array([1.0]))[0]))
    print("CSOC Allan Dev @ 100s:  {:.2e}".format(csoc.allan_deviation(np.array([100.0]))[0]))

    ratio = tcxo.allan_deviation(np.array([1.0])) / csoc.allan_deviation(np.array([1.0]))
    print(f"\nImprovement ratio @ 1s: {ratio[0]:.1f}×")
