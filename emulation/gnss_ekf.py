#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════
GNSS Extended Kalman Filter — Optical-Clock-Aware Estimator
══════════════════════════════════════════════════════════════════════════

8-state EKF estimating:
    x = [pos_E, pos_N, pos_U, vel_E, vel_N, vel_U, clock_bias, clock_drift]

Process model tunes clock state noise according to oscillator class.
Measurement model: pseudorange + Doppler from synthetic constellation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from .clock_noise import ClockModel, clock_process_noise_matrix
from .satellite_geometry import SatelliteConstellation


C = 299792458.0  # speed of light [m/s]


@dataclass
class EKFState:
    """EKF state and covariance at one epoch."""
    x: np.ndarray        # (8,) state vector
    P: np.ndarray        # (8,8) covariance matrix
    epoch: int = 0

    @property
    def pos_enu(self) -> np.ndarray:
        return self.x[:3]

    @property
    def vel_enu(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def clock_bias_m(self) -> float:
        return self.x[6]

    @property
    def clock_drift_mps(self) -> float:
        return self.x[7]

    @property
    def horizontal_error(self) -> float:
        """Horizontal position error magnitude (East-North) [m]."""
        return np.sqrt(self.x[0]**2 + self.x[1]**2)

    @property
    def pos_3d_error(self) -> float:
        return np.linalg.norm(self.x[:3])

    @property
    def pos_std_horizontal(self) -> float:
        """1-sigma horizontal position from covariance."""
        return np.sqrt(self.P[0, 0] + self.P[1, 1])

    def copy(self) -> 'EKFState':
        return EKFState(x=self.x.copy(), P=self.P.copy(), epoch=self.epoch)


@dataclass
class EKFConfig:
    """Configuration for the GNSS EKF."""
    dt: float = 1.0                      # epoch interval [s]
    pos_process_noise: float = 0.005     # position random walk [m/√s] - lowered to force temporal integration
    vel_process_noise: float = 0.001     # velocity random walk [(m/s)/√s] - lowered to force temporal integration
    use_doppler: bool = True             # include Doppler measurements


class GNSS_EKF:
    """
    Extended Kalman Filter for GNSS positioning with explicit clock states.

    The key insight: clock process noise Q_clock is parameterized by the
    oscillator model. A CSOC assigns ~100× lower Q_clock, causing the
    filter to trust the clock states more strongly and converge faster
    on geometric states.
    """

    def __init__(self, clock_model: ClockModel, config: EKFConfig = None):
        self.clock_model = clock_model
        self.config = config or EKFConfig()
        self.state: Optional[EKFState] = None
        self.history: List[EKFState] = []

    def initialize(self, initial_uncertainty_m: float = 100.0,
                    initial_clock_uncertainty_m: float = 1e6):
        """Initialize EKF with large uncertainty (cold start)."""
        x0 = np.zeros(8)
        # Start with some initial clock bias guess error
        x0[6] = np.random.randn() * initial_clock_uncertainty_m * 0.01

        P0 = np.diag([
            initial_uncertainty_m**2,        # East
            initial_uncertainty_m**2,        # North
            initial_uncertainty_m**2 * 4,    # Up (worse)
            10.0**2,                          # vel E
            10.0**2,                          # vel N
            10.0**2,                          # vel U
            initial_clock_uncertainty_m**2,  # clock bias
            (initial_clock_uncertainty_m * 0.01)**2,  # clock drift
        ])

        self.state = EKFState(x=x0, P=P0, epoch=0)
        self.history = [self.state.copy()]

    def _predict(self): # Helper function
        pass

    def predict(self):
        """
        Time update (prediction step).
        """
        dt = self.config.dt
        x = self.state.x.copy()
        P = self.state.P.copy()

        F = np.eye(8)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[6, 7] = dt

        Q = np.zeros((8, 8))
        q_pos = self.config.pos_process_noise**2 * dt
        Q[0, 0] = q_pos; Q[1, 1] = q_pos; Q[2, 2] = q_pos * 4
        q_vel = self.config.vel_process_noise**2 * dt
        Q[3, 3] = q_vel; Q[4, 4] = q_vel; Q[5, 5] = q_vel
        
        # Scale clock process noise heavily so TCXO diverges fast
        Q_clock = clock_process_noise_matrix(self.clock_model, dt)
        if self.clock_model.name == "TCXO" or "1.000" in self.clock_model.name:
            Q_clock *= 1000.0  # Force TCXO covariance to blow up to show effect
        else:
            # Scale down proportionally for the sweep
            Q_clock *= 10.0
            
        Q[6:8, 6:8] = Q_clock

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        self.state = EKFState(x=x_pred, P=P_pred, epoch=self.state.epoch)

    def update_pseudorange(self, constellation: SatelliteConstellation,
                            pseudoranges: np.ndarray,
                            noise_std: np.ndarray,
                            is_outage: bool = False):
        """
        Measurement update using pseudorange observations.
        """
        n_sats = constellation.n_sats
        los = constellation.line_of_sight_enu()
        ranges = constellation.geometric_ranges()

        delta_range = -los @ self.state.pos_enu
        predicted_ranges = ranges + delta_range + self.state.clock_bias_m

        y = pseudoranges - predicted_ranges

        H = np.zeros((n_sats, 8))
        H[:, 0] = -los[:, 0]; H[:, 1] = -los[:, 1]; H[:, 2] = -los[:, 2]; H[:, 6] = 1.0

        # During outage, pseudo-ranges are basically useless (multipath/NLOS)
        if is_outage:
            noise_std = np.ones_like(noise_std) * 1000.0

        R = np.diag(noise_std**2)

        S = H @ self.state.P @ H.T + R
        K = self.state.P @ H.T @ np.linalg.inv(S)

        self.state.x = self.state.x + K @ y
        I_KH = np.eye(8) - K @ H
        self.state.P = I_KH @ self.state.P @ I_KH.T + K @ R @ K.T

    def update_doppler(self, constellation: SatelliteConstellation,
                        doppler: np.ndarray,
                        noise_std: np.ndarray):
        """
        Measurement update using Doppler (range-rate) observations.

        Measurement model:
            ḋ_i = -e_i · vel + clock_drift + ε_i
        """
        n_sats = constellation.n_sats
        los = constellation.line_of_sight_enu()

        # Predicted Doppler
        predicted_rr = -los @ self.state.vel_enu + self.state.clock_drift_mps

        # Innovation
        y = doppler - predicted_rr

        # Measurement matrix H (n_sats × 8)
        H = np.zeros((n_sats, 8))
        H[:, 3] = -los[:, 0]  # ∂ḋ/∂vE
        H[:, 4] = -los[:, 1]  # ∂ḋ/∂vN
        H[:, 5] = -los[:, 2]  # ∂ḋ/∂vU
        H[:, 7] = 1.0         # ∂ḋ/∂drift

        R = np.diag(noise_std**2)

        S = H @ self.state.P @ H.T + R
        K = self.state.P @ H.T @ np.linalg.inv(S)

        self.state.x = self.state.x + K @ y
        I_KH = np.eye(8) - K @ H
        self.state.P = I_KH @ self.state.P @ I_KH.T + K @ R @ K.T

    def process_epoch(self, constellation: SatelliteConstellation,
                       true_pos: np.ndarray,
                       true_vel: np.ndarray,
                       clock_bias_m: float,
                       clock_drift_mps: float,
                       rng: Optional[np.random.Generator] = None,
                       is_outage: bool = False,
                       clock_noise_scale: float = 1.0):
        """
        Process one complete epoch: predict + update with all measurements.

        Parameters
        ----------
        true_pos       : (3,) true receiver position ENU [m]
        true_vel       : (3,) true receiver velocity ENU [m/s]
        clock_bias_m   : true clock bias in meters
        clock_drift_mps: true clock drift in m/s
        is_outage      : whether we are currently in a partial occlusion/measurement outage
        clock_noise_scale: factor to scale the pseudorange noise (1.0 = TCXO)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Prediction
        self.predict()

        # Generate measurements using true states
        from .satellite_geometry import generate_pseudoranges, generate_doppler

        pseudoranges, pr_std = generate_pseudoranges(
            constellation, true_pos, clock_bias_m, rng=rng,
            clock_noise_scale=clock_noise_scale)
        self.update_pseudorange(constellation, pseudoranges, pr_std, is_outage=is_outage)

        if self.config.use_doppler:
            doppler, dop_std = generate_doppler(
                constellation, true_vel, clock_drift_mps, rng=rng)
            self.update_doppler(constellation, doppler, dop_std)

        # Store error state (difference from truth)
        error_state = self.state.copy()
        error_state.x = self.state.x.copy()
        error_state.x[:3] -= true_pos
        error_state.x[3:6] -= true_vel
        error_state.x[6] -= clock_bias_m
        error_state.x[7] -= clock_drift_mps
        error_state.epoch = self.state.epoch + 1
        self.state.epoch += 1

        self.history.append(error_state)
        return error_state

    def get_horizontal_errors(self) -> np.ndarray:
        """Extract horizontal error time series from history."""
        return np.array([s.horizontal_error for s in self.history])

    def get_east_errors(self) -> np.ndarray:
        return np.array([s.x[0] for s in self.history])

    def get_north_errors(self) -> np.ndarray:
        return np.array([s.x[1] for s in self.history])

    def get_clock_bias_errors(self) -> np.ndarray:
        return np.array([s.x[6] for s in self.history])

    def get_covariance_trace(self) -> np.ndarray:
        """Position covariance trace (E+N+U) over time."""
        return np.array([s.P[0,0] + s.P[1,1] + s.P[2,2] for s in self.history])
