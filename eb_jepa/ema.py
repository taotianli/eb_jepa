"""
EMA variants for target encoder updates in JEPA-style models.

All classes share a common interface:
    ema = SomeEMA(online_model, target_model, **kwargs)
    stats = ema.step(named_grads=None)   # returns dict of diagnostic scalars

Ablation variants implemented:
    A1  StandardEMA          - fixed momentum (baseline)
    A2  ScheduledEMA         - cosine momentum schedule (I-JEPA default)
    A3  KalmanEMA            - scalar Kalman gain (proposed)
    A4  KalmanEMAPerLayer    - per-layer Kalman gain
    A5  GradientAdaptiveEMA  - gradient-norm adaptive momentum
    A6  DoubleEMA            - bias-corrected double EMA
    A7  LayerwiseEMA         - depth-dependent fixed momentum
"""

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _EMABase:
    """Common interface for all EMA variants."""

    def __init__(self, online: nn.Module, target: nn.Module):
        self.online = online
        self.target = target
        # Copy online weights into target at init
        self._sync_target()

    def _sync_target(self):
        with torch.no_grad():
            for op, tp in zip(
                self.online.parameters(), self.target.parameters()
            ):
                tp.data.copy_(op.data)
            for ob, tb in zip(self.online.buffers(), self.target.buffers()):
                tb.data.copy_(ob.data)

    def step(self, named_grads=None) -> dict:
        raise NotImplementedError

    def state_dict(self) -> dict:
        """Return internal state for checkpointing."""
        return {}

    def load_state_dict(self, state: dict):
        pass


# ---------------------------------------------------------------------------
# A1: Standard EMA (fixed momentum)
# ---------------------------------------------------------------------------

class StandardEMA(_EMABase):
    """Fixed-momentum EMA: ξ ← m·ξ + (1-m)·θ"""

    def __init__(self, online: nn.Module, target: nn.Module, momentum: float = 0.996):
        super().__init__(online, target)
        self.momentum = momentum

    @torch.no_grad()
    def step(self, named_grads=None) -> dict:
        m = self.momentum
        for op, tp in zip(self.online.parameters(), self.target.parameters()):
            tp.data.mul_(m).add_(op.data, alpha=1.0 - m)
        return {"momentum": m, "equiv_momentum": m}


# ---------------------------------------------------------------------------
# A2: Scheduled EMA (cosine momentum schedule, I-JEPA default)
# ---------------------------------------------------------------------------

class ScheduledEMA(_EMABase):
    """
    Cosine momentum schedule: momentum rises from m_start to m_end over
    total_steps, matching the I-JEPA paper's EMA schedule.
    """

    def __init__(
        self,
        online: nn.Module,
        target: nn.Module,
        m_start: float = 0.996,
        m_end: float = 1.0,
        total_steps: int = 100_000,
    ):
        super().__init__(online, target)
        self.m_start = m_start
        self.m_end = m_end
        self.total_steps = total_steps
        self._step = 0

    def _current_momentum(self) -> float:
        t = min(self._step / max(self.total_steps, 1), 1.0)
        # cosine interpolation from m_start → m_end
        return self.m_end - (self.m_end - self.m_start) * (
            1.0 + math.cos(math.pi * t)
        ) / 2.0

    @torch.no_grad()
    def step(self, named_grads=None) -> dict:
        m = self._current_momentum()
        for op, tp in zip(self.online.parameters(), self.target.parameters()):
            tp.data.mul_(m).add_(op.data, alpha=1.0 - m)
        self._step += 1
        return {"momentum": m, "equiv_momentum": m, "ema_step": self._step}

    def state_dict(self):
        return {"_step": self._step}

    def load_state_dict(self, state):
        self._step = state.get("_step", 0)


# ---------------------------------------------------------------------------
# A3: Kalman EMA (scalar covariance)
# ---------------------------------------------------------------------------

class KalmanEMA(_EMABase):
    """
    State-space model for the target encoder (scalar covariance version).

    State equation:   ξ_t  = ξ_{t-1} + w_t,   w_t ~ N(0, Q_t)
    Observation:      θ_t  = ξ_t     + v_t,   v_t ~ N(0, R_t)

    Kalman gain:      K_t  = P_{t|t-1} / (P_{t|t-1} + R_t)
    Update:           ξ_t  = ξ_{t-1} + K_t * (θ_t - ξ_{t-1})
    Covariance:       P_{t|t}   = (1 - K_t) * P_{t|t-1}
                      P_{t+1|t} = P_{t|t} + Q_t

    Online noise estimation:
        Q_t  ≈ EMA(||θ_t - θ_{t-1}||² / n_params)   [process noise]
        R_t  ≈ EMA(||∇L||²           / n_params)     [observation noise]
    """

    def __init__(
        self,
        online: nn.Module,
        target: nn.Module,
        init_momentum: float = 0.996,
        q_momentum: float = 0.99,
        r_momentum: float = 0.99,
        min_gain: float = 1e-4,
        max_gain: float = 0.5,
    ):
        super().__init__(online, target)
        self.q_mom = q_momentum
        self.r_mom = r_momentum
        self.min_gain = min_gain
        self.max_gain = max_gain

        # Initialise P from init_momentum: K = 1-m  →  P/(P+R)=1-m
        # With R=1 initially: P = (1-m)/m
        init_K = 1.0 - init_momentum
        self.P = init_K / max(1.0 - init_K, 1e-8)

        # Running estimates (start at small positive values)
        self.Q_ema = 1e-4
        self.R_ema = 1.0

        # Store previous online params for Q estimation
        self._prev_params = self._flatten_params(online)

    @staticmethod
    def _flatten_params(model: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.detach().cpu().flatten() for p in model.parameters()])

    def _estimate_Q(self) -> float:
        curr = self._flatten_params(self.online)
        delta_sq = (curr - self._prev_params).pow(2).sum().item()
        n = max(curr.numel(), 1)
        self._prev_params = curr
        return delta_sq / n

    def _estimate_R(self, named_grads) -> float:
        if named_grads is None:
            return self.R_ema
        grad_sq = sum(
            g.pow(2).sum().item() for g in named_grads.values() if g is not None
        )
        n = max(
            sum(g.numel() for g in named_grads.values() if g is not None), 1
        )
        return grad_sq / n

    @torch.no_grad()
    def step(self, named_grads=None) -> dict:
        # 1. Process noise
        Q_t = self._estimate_Q()
        self.Q_ema = self.q_mom * self.Q_ema + (1.0 - self.q_mom) * Q_t

        # 2. Observation noise
        R_t = self._estimate_R(named_grads)
        self.R_ema = self.r_mom * self.R_ema + (1.0 - self.r_mom) * R_t

        # 3. Kalman gain
        K_t = float(np.clip(
            self.P / (self.P + self.R_ema + 1e-8),
            self.min_gain,
            self.max_gain,
        ))

        # 4. Update target encoder
        for op, tp in zip(self.online.parameters(), self.target.parameters()):
            tp.data.add_(K_t * (op.data - tp.data))

        # 5. Covariance update
        self.P = (1.0 - K_t) * self.P + self.Q_ema

        return {
            "kalman_gain": K_t,
            "equiv_momentum": 1.0 - K_t,
            "Q_t": self.Q_ema,
            "R_t": self.R_ema,
            "P_t": self.P,
        }

    def state_dict(self):
        return {"P": self.P, "Q_ema": self.Q_ema, "R_ema": self.R_ema}

    def load_state_dict(self, state):
        self.P = state.get("P", self.P)
        self.Q_ema = state.get("Q_ema", self.Q_ema)
        self.R_ema = state.get("R_ema", self.R_ema)


# ---------------------------------------------------------------------------
# A4: Kalman EMA per-layer (independent P per named parameter)
# ---------------------------------------------------------------------------

class KalmanEMAPerLayer(_EMABase):
    """
    Same Kalman filter as KalmanEMA but each named parameter group
    maintains its own scalar covariance P_i, Q_i, R_i.
    """

    def __init__(
        self,
        online: nn.Module,
        target: nn.Module,
        init_momentum: float = 0.996,
        q_momentum: float = 0.99,
        r_momentum: float = 0.99,
        min_gain: float = 1e-4,
        max_gain: float = 0.5,
    ):
        super().__init__(online, target)
        self.q_mom = q_momentum
        self.r_mom = r_momentum
        self.min_gain = min_gain
        self.max_gain = max_gain

        init_K = 1.0 - init_momentum
        init_P = init_K / max(1.0 - init_K, 1e-8)

        # Per-parameter state
        self._state = {
            name: {"P": init_P, "Q_ema": 1e-4, "R_ema": 1.0}
            for name, _ in online.named_parameters()
        }
        self._prev = {
            name: p.data.detach().clone()
            for name, p in online.named_parameters()
        }

    @torch.no_grad()
    def step(self, named_grads=None) -> dict:
        gains = []
        for (name, op), (_, tp) in zip(
            self.online.named_parameters(), self.target.named_parameters()
        ):
            s = self._state[name]

            # Q: per-param process noise
            delta_sq = (op.data - self._prev[name]).pow(2).mean().item()
            s["Q_ema"] = self.q_mom * s["Q_ema"] + (1.0 - self.q_mom) * delta_sq
            self._prev[name].copy_(op.data)

            # R: per-param observation noise
            if named_grads is not None and named_grads.get(name) is not None:
                r_raw = named_grads[name].pow(2).mean().item()
                s["R_ema"] = self.r_mom * s["R_ema"] + (1.0 - self.r_mom) * r_raw

            # Kalman gain
            K = float(np.clip(
                s["P"] / (s["P"] + s["R_ema"] + 1e-8),
                self.min_gain,
                self.max_gain,
            ))
            gains.append(K)

            # Update target
            tp.data.add_(K * (op.data - tp.data))

            # Covariance update
            s["P"] = (1.0 - K) * s["P"] + s["Q_ema"]

        mean_K = float(np.mean(gains))
        return {
            "kalman_gain_mean": mean_K,
            "kalman_gain_min": float(np.min(gains)),
            "kalman_gain_max": float(np.max(gains)),
            "equiv_momentum": 1.0 - mean_K,
        }

    def state_dict(self):
        return {"_state": self._state}

    def load_state_dict(self, state):
        if "_state" in state:
            self._state.update(state["_state"])


# ---------------------------------------------------------------------------
# A5: Gradient-Adaptive EMA
# ---------------------------------------------------------------------------

class GradientAdaptiveEMA(_EMABase):
    """
    Momentum adapts inversely to gradient norm:
        m_t = 1 - base_gain / (1 + ||∇L||_norm)
    Large gradients → lower momentum (faster update).
    Small gradients → higher momentum (stable target).
    """

    def __init__(
        self,
        online: nn.Module,
        target: nn.Module,
        base_momentum: float = 0.996,
        grad_momentum: float = 0.99,
        min_momentum: float = 0.9,
        max_momentum: float = 0.9999,
    ):
        super().__init__(online, target)
        self.base_momentum = base_momentum
        self.grad_mom = grad_momentum
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self._grad_norm_ema = 1.0

    @torch.no_grad()
    def step(self, named_grads=None) -> dict:
        if named_grads is not None:
            grad_sq = sum(
                g.pow(2).sum().item() for g in named_grads.values() if g is not None
            )
            n = max(
                sum(g.numel() for g in named_grads.values() if g is not None), 1
            )
            self._grad_norm_ema = (
                self.grad_mom * self._grad_norm_ema
                + (1.0 - self.grad_mom) * math.sqrt(grad_sq / n)
            )

        base_gain = 1.0 - self.base_momentum
        m = 1.0 - base_gain / (1.0 + self._grad_norm_ema)
        m = float(np.clip(m, self.min_momentum, self.max_momentum))

        for op, tp in zip(self.online.parameters(), self.target.parameters()):
            tp.data.mul_(m).add_(op.data, alpha=1.0 - m)

        return {"momentum": m, "equiv_momentum": m, "grad_norm_ema": self._grad_norm_ema}

    def state_dict(self):
        return {"_grad_norm_ema": self._grad_norm_ema}

    def load_state_dict(self, state):
        self._grad_norm_ema = state.get("_grad_norm_ema", self._grad_norm_ema)


# ---------------------------------------------------------------------------
# A6: Double EMA (bias-corrected, removes lag)
# ---------------------------------------------------------------------------

class DoubleEMA(_EMABase):
    """
    Double EMA eliminates the lag of a single EMA:
        S1_t = m * S1_{t-1} + (1-m) * θ_t
        S2_t = m * S2_{t-1} + (1-m) * S1_t
        ξ_t  = 2*S1_t - S2_t

    This is the DES (Double Exponential Smoothing) formula.
    """

    def __init__(
        self,
        online: nn.Module,
        target: nn.Module,
        momentum: float = 0.996,
    ):
        super().__init__(online, target)
        self.momentum = momentum
        # S1 and S2 shadow models
        self._s1 = deepcopy(online)
        self._s2 = deepcopy(online)
        for p in self._s1.parameters():
            p.requires_grad_(False)
        for p in self._s2.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def step(self, named_grads=None) -> dict:
        m = self.momentum
        for op, s1p, s2p, tp in zip(
            self.online.parameters(),
            self._s1.parameters(),
            self._s2.parameters(),
            self.target.parameters(),
        ):
            s1p.data.mul_(m).add_(op.data, alpha=1.0 - m)
            s2p.data.mul_(m).add_(s1p.data, alpha=1.0 - m)
            tp.data.copy_(2.0 * s1p.data - s2p.data)

        return {"momentum": m, "equiv_momentum": m}


# ---------------------------------------------------------------------------
# A7: Layer-wise EMA (shallower layers get lower momentum)
# ---------------------------------------------------------------------------

class LayerwiseEMA(_EMABase):
    """
    Assigns momentum based on layer depth: shallow layers track faster,
    deep layers track slower.

        m_i = m_max - (m_max - m_min) * (1 - i/L)^alpha

    where i is the layer index (0 = shallowest) and L is total layers.
    """

    def __init__(
        self,
        online: nn.Module,
        target: nn.Module,
        m_min: float = 0.99,
        m_max: float = 0.9999,
        alpha: float = 2.0,
    ):
        super().__init__(online, target)
        params = list(online.named_parameters())
        L = max(len(params) - 1, 1)
        self._momentums = []
        for i, _ in enumerate(params):
            depth_frac = i / L  # 0 = shallow, 1 = deep
            m = m_max - (m_max - m_min) * (1.0 - depth_frac) ** alpha
            self._momentums.append(m)

    @torch.no_grad()
    def step(self, named_grads=None) -> dict:
        for (_, op), (_, tp), m in zip(
            self.online.named_parameters(),
            self.target.named_parameters(),
            self._momentums,
        ):
            tp.data.mul_(m).add_(op.data, alpha=1.0 - m)

        return {
            "momentum_min": min(self._momentums),
            "momentum_max": max(self._momentums),
            "equiv_momentum": float(np.mean(self._momentums)),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "standard": StandardEMA,
    "scheduled": ScheduledEMA,
    "kalman": KalmanEMA,
    "kalman_per_layer": KalmanEMAPerLayer,
    "gradient_adaptive": GradientAdaptiveEMA,
    "double": DoubleEMA,
    "layerwise": LayerwiseEMA,
}


def build_ema(ema_type: str, online: nn.Module, target: nn.Module, **kwargs) -> _EMABase:
    """
    Factory function.

    Args:
        ema_type: one of 'standard', 'scheduled', 'kalman', 'kalman_per_layer',
                  'gradient_adaptive', 'double', 'layerwise'
        online:   the context encoder (θ)
        target:   the target encoder (ξ), will be synced to online at init
        **kwargs: passed to the chosen EMA class constructor
    """
    if ema_type not in _REGISTRY:
        raise ValueError(
            f"Unknown ema_type '{ema_type}'. Choose from: {list(_REGISTRY)}"
        )
    return _REGISTRY[ema_type](online, target, **kwargs)
