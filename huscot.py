"""
huscot.py

Lightweight Python implementation of core HuSCoT kernels and an integrated
analysis profile.

MIT License
Copyright © 2025 Steven Srebranig
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import math


# ---------------------------------------------------------------------------
# Data classes for module outputs
# ---------------------------------------------------------------------------

@dataclass
class CSAResult:
    layer_scores: Dict[str, List[float]]
    fragmentation: Dict[str, float]
    fi_mean: float
    coherence_gradient: List[float]
    integration_coefficient: float


@dataclass
class WSAResult:
    capacities: Dict[str, float]
    loads: Dict[str, float]
    overload_ratios: Dict[str, float]


@dataclass
class CSMResult:
    pressure_field: List[List[float]]
    local_gradients: List[List[float]]
    coherence_vectors: List[Tuple[float, ...]]
    vector_magnitudes: List[float]
    interface_tension: float


@dataclass
class WCSResult:
    cohesion: float
    entropy: float
    dissolution_risk: float
    alpha: float
    beta: float


@dataclass
class SECLResult:
    error_series: List[float]
    max_envelope: float
    stable: bool
    control_effort: float


@dataclass
class HCMResult:
    baseline_hist: List[float]
    current_hist: List[float]
    divergence: float
    threshold: float
    drift_detected: bool


@dataclass
class MEOResult:
    x_t: List[float]
    grad_u: List[float]
    eta: float
    x_next: List[float]


@dataclass
class HuSCoTProfile:
    csa: CSAResult
    wsa: WSAResult
    csm: Optional[CSMResult]
    wcs: WCSResult
    secl: Optional[SECLResult]
    hcm: Optional[HCMResult]
    meo: Optional[MEOResult]


# ---------------------------------------------------------------------------
# CSA — Cognitive-Structural Analysis
# ---------------------------------------------------------------------------

def compute_csa(layer_scores: Dict[str, List[float]]) -> CSAResult:
    """
    Compute CSA kernels:
      - Fragmentation Index (per layer + mean)
      - Coherence Gradient
      - Integration Coefficient

    layer_scores: dict of layer_name -> list of scores per segment.
                  All lists must be the same length.
    """
    if not layer_scores:
        raise ValueError("layer_scores cannot be empty.")

    # assume all layers same length
    lengths = {len(v) for v in layer_scores.values()}
    if len(lengths) != 1:
        raise ValueError("All layers must have the same number of segments.")
    n = lengths.pop()

    # Fragmentation Index per layer
    fragmentation: Dict[str, float] = {}
    for name, vals in layer_scores.items():
        if n <= 1:
            fragmentation[name] = 0.0
        else:
            acc = 0.0
            for k in range(n - 1):
                acc += abs(vals[k + 1] - vals[k])
            fragmentation[name] = acc / (n - 1)

    # Mean FI
    fi_mean = sum(fragmentation.values()) / len(fragmentation)

    # Coherence Gradient: CG_k = mean_i(l_{i,k})
    coherence_gradient: List[float] = []
    layer_names = list(layer_scores.keys())
    for k in range(n):
        s = 0.0
        for name in layer_names:
            s += layer_scores[name][k]
        coherence_gradient.append(s / len(layer_names))

    # Integration Coefficient:
    # IC = (1/n) Σ [ (1/L) Σ_i l_{i,k} ]^2
    L = len(layer_names)
    ic_acc = 0.0
    for k in range(n):
        avg_k = 0.0
        for name in layer_names:
            avg_k += layer_scores[name][k]
        avg_k /= L
        ic_acc += avg_k ** 2
    integration_coefficient = ic_acc / n

    return CSAResult(
        layer_scores=layer_scores,
        fragmentation=fragmentation,
        fi_mean=fi_mean,
        coherence_gradient=coherence_gradient,
        integration_coefficient=integration_coefficient,
    )


# ---------------------------------------------------------------------------
# WSA — Weighted Structure Analysis
# ---------------------------------------------------------------------------

def compute_wsa(capacities: Dict[str, float],
                loads: Dict[str, float]) -> WSAResult:
    """
    Compute overload ratios R_i = L_i / C_i for each component class.
    """
    overload_ratios: Dict[str, float] = {}
    for name, C_i in capacities.items():
        if C_i <= 0:
            raise ValueError(f"Capacity for {name} must be > 0.")
        L_i = loads.get(name, 0.0)
        overload_ratios[name] = L_i / C_i
    return WSAResult(capacities=capacities, loads=loads,
                     overload_ratios=overload_ratios)


# ---------------------------------------------------------------------------
# CSM — Cognitive-Structure Mapping
# ---------------------------------------------------------------------------

def compute_csm(pressure_field: List[List[float]]) -> CSMResult:
    """
    Given a pressure field P[i][k] (i = layer/region, k = segment),
    compute local gradients, coherence vectors, and interface tension.
    """
    if not pressure_field:
        raise ValueError("pressure_field cannot be empty.")

    num_layers = len(pressure_field)
    lengths = {len(row) for row in pressure_field}
    if len(lengths) != 1:
        raise ValueError("All rows in pressure_field must have same length.")
    n = lengths.pop()

    # Local gradients: ∇p_{i,k} = p_{i,k+1} - p_{i,k}
    local_gradients: List[List[float]] = []
    for i in range(num_layers):
        row = pressure_field[i]
        grads = []
        for k in range(n - 1):
            grads.append(row[k + 1] - row[k])
        local_gradients.append(grads)

    # Coherence vectors C_k = (∇p_{1,k}, ..., ∇p_{L,k})
    coherence_vectors: List[Tuple[float, ...]] = []
    vector_magnitudes: List[float] = []
    for k in range(n - 1):
        components = [local_gradients[i][k] for i in range(num_layers)]
        coherence_vectors.append(tuple(components))
        mag = math.sqrt(sum(c * c for c in components))
        vector_magnitudes.append(mag)

    interface_tension = sum(vector_magnitudes)

    return CSMResult(
        pressure_field=pressure_field,
        local_gradients=local_gradients,
        coherence_vectors=coherence_vectors,
        vector_magnitudes=vector_magnitudes,
        interface_tension=interface_tension,
    )


# ---------------------------------------------------------------------------
# WCS/DR — Cohesion, Entropy, Dissolution Risk
# ---------------------------------------------------------------------------

def compute_entropy(probs: Sequence[float]) -> float:
    """
    Shannon entropy: E = - Σ p log p
    Assumes probs are non-negative and sum to 1 (or close).
    """
    eps = 1e-12
    total = sum(probs)
    if total <= 0:
        return 0.0
    normed = [max(p / total, eps) for p in probs]
    return -sum(p * math.log(p) for p in normed)


def compute_wcs_dr(cohesion: float,
                   entropy: float,
                   alpha: float = 1.0,
                   beta: float = 1.0) -> WCSResult:
    """
    DR = 1 / (1 + exp( -(αE - βC) ))
    """
    z = alpha * entropy - beta * cohesion
    dr = 1.0 / (1.0 + math.exp(-z))
    return WCSResult(
        cohesion=cohesion,
        entropy=entropy,
        dissolution_risk=dr,
        alpha=alpha,
        beta=beta,
    )


# ---------------------------------------------------------------------------
# SECL — Statistical-Envelope Control Loops
# ---------------------------------------------------------------------------

def compute_secl(error_series: List[float],
                 max_envelope: float) -> SECLResult:
    """
    Simple SECL check:
      - stable if all |e(t)| <= max_envelope
      - control_effort = Σ |e(t)|
    """
    control_effort = sum(abs(e) for e in error_series)
    stable = all(abs(e) <= max_envelope for e in error_series)
    return SECLResult(
        error_series=error_series,
        max_envelope=max_envelope,
        stable=stable,
        control_effort=control_effort,
    )


# ---------------------------------------------------------------------------
# HCM — Histogram Confidence Method
# ---------------------------------------------------------------------------

def compute_hcm(baseline_hist: Sequence[float],
                current_hist: Sequence[float],
                threshold: float) -> HCMResult:
    """
    D = Σ |H_{0b} - H_{tb}|
    Drift if D >= threshold.
    """
    if len(baseline_hist) != len(current_hist):
        raise ValueError("Histograms must have the same length.")
    divergence = sum(abs(a - b) for a, b in zip(baseline_hist, current_hist))
    drift = divergence >= threshold
    return HCMResult(
        baseline_hist=list(baseline_hist),
        current_hist=list(current_hist),
        divergence=divergence,
        threshold=threshold,
        drift_detected=drift,
    )


# ---------------------------------------------------------------------------
# MEO/OO — Microeconomic Equilibrium Optimizer / Opportunity Optimization
# ---------------------------------------------------------------------------

def compute_meo_step(x_t: Sequence[float],
                     grad_u: Sequence[float],
                     eta: float) -> MEOResult:
    """
    x_{t+1} = x_t + η ∇U(x_t)
    """
    if len(x_t) != len(grad_u):
        raise ValueError("x_t and grad_u must have the same dimension.")
    x_next = [xt + eta * g for xt, g in zip(x_t, grad_u)]
    return MEOResult(
        x_t=list(x_t),
        grad_u=list(grad_u),
        eta=eta,
        x_next=x_next,
    )


# ---------------------------------------------------------------------------
# Integrated HuSCoT profile (very simple wiring)
# ---------------------------------------------------------------------------

def build_huscot_profile(
    csa_layers: Dict[str, List[float]],
    capacities: Dict[str, float],
    loads: Dict[str, float],
    cohesion: float,
    entropy_probs: Sequence[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    pressure_field: Optional[List[List[float]]] = None,
    error_series: Optional[List[float]] = None,
    max_envelope: float = 1.0,
    baseline_hist: Optional[Sequence[float]] = None,
    current_hist: Optional[Sequence[float]] = None,
    hcm_threshold: float = 0.5,
    x_t: Optional[Sequence[float]] = None,
    grad_u: Optional[Sequence[float]] = None,
    eta: float = 0.1,
) -> HuSCoTProfile:
    """
    Convenience function to compute a minimal, integrated HuSCoT profile.
    Not all modules are required; some may be None depending on inputs.
    """
    csa = compute_csa(csa_layers)
    wsa = compute_wsa(capacities, loads)

    csm = None
    if pressure_field is not None:
        csm = compute_csm(pressure_field)

    entropy = compute_entropy(entropy_probs)
    wcs = compute_wcs_dr(cohesion=cohesion, entropy=entropy,
                         alpha=alpha, beta=beta)

    secl = None
    if error_series is not None:
        secl = compute_secl(error_series, max_envelope)

    hcm = None
    if baseline_hist is not None and current_hist is not None:
        hcm = compute_hcm(baseline_hist, current_hist, hcm_threshold)

    meo = None
    if x_t is not None and grad_u is not None:
        meo = compute_meo_step(x_t, grad_u, eta)

    return HuSCoTProfile(
        csa=csa,
        wsa=wsa,
        csm=csm,
        wcs=wcs,
        secl=secl,
        hcm=hcm,
        meo=meo,
    )
