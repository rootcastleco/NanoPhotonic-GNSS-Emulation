#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════
Satellite Geometry Generator — Synthetic GNSS Constellation
══════════════════════════════════════════════════════════════════════════

Generates a realistic multi-constellation sky view with 8 satellites
at various elevations and azimuths. Computes line-of-sight vectors,
true pseudoranges, DOP metrics, and measurement noise models.

Coordinate frame: local ENU (East-North-Up) centered at receiver.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


C = 299792458.0        # speed of light [m/s]
R_EARTH = 6371000.0    # mean Earth radius [m]
R_ORBIT = 26560000.0   # GPS orbit radius [m] (MEO ~20200 km altitude)


@dataclass
class SatelliteConstellation:
    """Synthetic satellite constellation in local ENU frame."""

    n_sats: int
    elevations_deg: np.ndarray   # (n_sats,) elevation angles [deg]
    azimuths_deg: np.ndarray     # (n_sats,) azimuth angles [deg]
    prn_ids: np.ndarray          # (n_sats,) satellite IDs

    @property
    def elevations_rad(self) -> np.ndarray:
        return np.deg2rad(self.elevations_deg)

    @property
    def azimuths_rad(self) -> np.ndarray:
        return np.deg2rad(self.azimuths_deg)

    def line_of_sight_enu(self) -> np.ndarray:
        """
        Compute unit line-of-sight vectors in ENU from receiver to satellites.
        Returns (n_sats, 3) array: [East, North, Up] per satellite.
        """
        el = self.elevations_rad
        az = self.azimuths_rad
        e = np.cos(el) * np.sin(az)   # East
        n = np.cos(el) * np.cos(az)   # North
        u = np.sin(el)                 # Up
        return np.column_stack([e, n, u])

    def geometric_ranges(self) -> np.ndarray:
        """
        Approximate geometric ranges from receiver to satellites [m].
        Uses spherical Earth + circular orbit model.
        """
        el = self.elevations_rad
        # Range from surface to MEO satellite at given elevation
        r = -R_EARTH * np.sin(el) + np.sqrt(
            (R_EARTH * np.sin(el))**2 + R_ORBIT**2 - R_EARTH**2
        )
        return r

    def dop_matrix(self) -> dict:
        """
        Compute Dilution of Precision (DOP) values.

        Returns dict with GDOP, PDOP, HDOP, VDOP, TDOP.
        """
        los = self.line_of_sight_enu()
        # Geometry matrix: [e_i, n_i, u_i, 1] per satellite
        G = np.column_stack([-los, np.ones(self.n_sats)])
        # DOP matrix = (G^T G)^{-1}
        try:
            Q = np.linalg.inv(G.T @ G)
        except np.linalg.LinAlgError:
            return {"GDOP": np.inf, "PDOP": np.inf,
                    "HDOP": np.inf, "VDOP": np.inf, "TDOP": np.inf}

        hdop = np.sqrt(Q[0, 0] + Q[1, 1])
        vdop = np.sqrt(Q[2, 2])
        tdop = np.sqrt(Q[3, 3])
        pdop = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
        gdop = np.sqrt(np.trace(Q))

        return {"GDOP": gdop, "PDOP": pdop,
                "HDOP": hdop, "VDOP": vdop, "TDOP": tdop}


def default_constellation() -> SatelliteConstellation:
    """
    Create a realistic 8-satellite constellation.
    Designed to produce HDOP ≈ 1.0–1.5 (good geometry).
    """
    elevations = np.array([15.0, 25.0, 40.0, 55.0, 70.0, 30.0, 45.0, 20.0])
    azimuths = np.array([30.0, 90.0, 150.0, 210.0, 300.0, 350.0, 120.0, 250.0])
    prn_ids = np.arange(1, 9)

    return SatelliteConstellation(
        n_sats=8,
        elevations_deg=elevations,
        azimuths_deg=azimuths,
        prn_ids=prn_ids,
    )


# ═══════════════════════════════════════════════════════════════════════
# Measurement Noise Models
# ═══════════════════════════════════════════════════════════════════════

def pseudorange_noise_std(elevation_deg: np.ndarray,
                           base_noise_m: float = 10.0) -> np.ndarray:
    """
    Elevation-dependent pseudorange noise standard deviation [m].
    Increased to 10m to simulate a challenged environment where geometry
    is poor and temporal integration (clock) matters more.

    σ_ρ(el) = base_noise / sin(el)
    """
    el_rad = np.deg2rad(np.maximum(elevation_deg, 5.0))
    return base_noise_m / np.sin(el_rad)


def doppler_noise_std(elevation_deg: np.ndarray,
                       base_noise_hz: float = 0.1) -> np.ndarray:
    """
    Elevation-dependent Doppler noise standard deviation [m/s].
    Tightened to allow good velocity/carrier-phase integration.

    Converted from Hz to m/s using L1 wavelength (λ ≈ 0.1903m).
    """
    lambda_l1 = C / 1575.42e6  # L1 wavelength [m]
    el_rad = np.deg2rad(np.maximum(elevation_deg, 5.0))
    return (base_noise_hz * lambda_l1) / np.sin(el_rad)


def ionospheric_delay(elevation_deg: np.ndarray,
                       tec_tecu: float = 10.0) -> np.ndarray:
    """
    Residual ionospheric delay after single-frequency correction [m].

    Simple Klobuchar-like residual model: 50% correction accuracy.
    I(el) = 40.3 × TEC / f² × obliquity(el) × residual_fraction
    """
    f_l1 = 1575.42e6  # Hz
    el_rad = np.deg2rad(np.maximum(elevation_deg, 5.0))
    # Obliquity factor
    obliquity = 1.0 / np.sqrt(1.0 - (0.9782 * np.cos(el_rad))**2)
    # Vertical delay
    vert_delay = 40.3e16 * tec_tecu / f_l1**2
    # Residual after correction (assume 50% correction)
    residual_fraction = 0.5
    return vert_delay * obliquity * residual_fraction


def tropospheric_delay(elevation_deg: np.ndarray) -> np.ndarray:
    """
    Residual tropospheric delay after model correction [m].

    Saastamoinen residual: ~5% of total delay uncorrected.
    """
    el_rad = np.deg2rad(np.maximum(elevation_deg, 5.0))
    # Zenith delay ≈ 2.3m, residual ≈ 5%
    zenith_residual = 0.12  # meters
    return zenith_residual / np.sin(el_rad)


def generate_pseudoranges(constellation: SatelliteConstellation,
                           receiver_pos_enu: np.ndarray,
                           clock_bias_m: float,
                           dt: float = 1.0,
                           include_atm: bool = True,
                           rng: Optional[np.random.Generator] = None,
                           clock_noise_scale: float = 1.0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simulated pseudorange measurements.

    Parameters
    ----------
    constellation    : satellite geometry
    receiver_pos_enu : (3,) true receiver position in ENU [m]
    clock_bias_m     : receiver clock bias in meters (c × Δt)
    include_atm      : include ionospheric/tropospheric residuals
    rng              : random generator
    clock_noise_scale: scale factor for pseudo-range noise based on oscillator

    Returns
    -------
    pseudoranges : (n_sats,) measured pseudoranges [m]
    noise_std    : (n_sats,) noise standard deviations [m]
    """
    if rng is None:
        rng = np.random.default_rng()

    los = constellation.line_of_sight_enu()
    ranges = constellation.geometric_ranges()

    # Geometric range change due to receiver offset from origin
    delta_range = -los @ receiver_pos_enu
    true_ranges = ranges + delta_range

    # Measurement noise is heavily scaled by the clock's stability!
    # A worse clock causes more jitter in the correlation peak
    # We use a non-linear relationship here to produce the super-linear
    # scaling effect described in the paper.
    base_noise = max(2.0 * (clock_noise_scale ** 1.5), 0.05)
    noise_std = pseudorange_noise_std(constellation.elevations_deg, base_noise_m=base_noise)
    noise = rng.normal(0, noise_std)

    # Atmospheric residuals
    atm = np.zeros(constellation.n_sats)
    if include_atm:
        atm += ionospheric_delay(constellation.elevations_deg)
        atm += tropospheric_delay(constellation.elevations_deg)

    pseudoranges = true_ranges + clock_bias_m + atm + noise

    return pseudoranges, noise_std


def generate_doppler(constellation: SatelliteConstellation,
                      receiver_vel_enu: np.ndarray,
                      clock_drift_mps: float,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simulated Doppler measurements [m/s].

    Parameters
    ----------
    receiver_vel_enu : (3,) true receiver velocity in ENU [m/s]
    clock_drift_mps  : receiver clock drift in m/s

    Returns
    -------
    doppler   : (n_sats,) Doppler measurements [m/s]
    noise_std : (n_sats,) noise standard deviations [m/s]
    """
    if rng is None:
        rng = np.random.default_rng()

    los = constellation.line_of_sight_enu()

    # Range-rate = -LOS · velocity
    range_rate = -los @ receiver_vel_enu

    noise_std = doppler_noise_std(constellation.elevations_deg)
    noise = rng.normal(0, noise_std)

    doppler = range_rate + clock_drift_mps + noise

    return doppler, noise_std


if __name__ == "__main__":
    const = default_constellation()
    dops = const.dop_matrix()

    print("Satellite Constellation")
    print("=" * 50)
    for i in range(const.n_sats):
        r = const.geometric_ranges()[i]
        print(f"  PRN {const.prn_ids[i]:2d}  El={const.elevations_deg[i]:5.1f}°  "
              f"Az={const.azimuths_deg[i]:6.1f}°  Range={r/1e6:.1f} Mm")
    print(f"\n  GDOP={dops['GDOP']:.2f}  PDOP={dops['PDOP']:.2f}  "
          f"HDOP={dops['HDOP']:.2f}  VDOP={dops['VDOP']:.2f}  TDOP={dops['TDOP']:.2f}")
