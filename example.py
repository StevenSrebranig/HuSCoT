"""
example.py

Minimal demonstration of the HuSCoT kernels on a toy clinic-like scenario.

Run:
    python example.py
"""

from huscot import (
    build_huscot_profile,
)


def main() -> None:
    # ------------------------------------------------------------------
    # Toy CSA layer scores for 5 segments (e.g., Intake, Nursing, etc.)
    # ------------------------------------------------------------------
    csa_layers = {
        "BCL":  [0.55, 0.60, 0.70, 0.40, 0.65],
        "RSL":  [0.40, 0.45, 0.65, 0.30, 0.55],
        "CLIL": [0.35, 0.40, 0.50, 0.25, 0.45],
        "DCL":  [0.50, 0.55, 0.70, 0.45, 0.60],
        "HSEL": [0.45, 0.50, 0.65, 0.35, 0.55],
    }

    # ------------------------------------------------------------------
    # WSA capacities and loads (normalized around 1.0 capacity)
    # ------------------------------------------------------------------
    capacities = {
        "Physicians": 1.0,
        "Nursing": 1.0,
        "Intake": 1.0,
        "Scheduling": 1.0,
        "Admin": 1.0,
    }
    loads = {
        "Physicians": 0.95,
        "Nursing": 1.15,
        "Intake": 1.20,
        "Scheduling": 1.35,
        "Admin": 0.85,
    }

    # ------------------------------------------------------------------
    # Simple cohesion + entropy proxy:
    # cohesion approximated from CSA integration coefficient.
    # entropy estimated from a toy probability vector (e.g., error states).
    # ------------------------------------------------------------------
    # For demonstration, we’ll plug cohesion in later from CSA result.
    entropy_probs = [0.4, 0.3, 0.2, 0.1]  # arbitrary

    # ------------------------------------------------------------------
    # Optional dynamic fields (toy values)
    # ------------------------------------------------------------------
    pressure_field = [
        # layer/region 1..L over segments 1..n
        [0.1, 0.2, 0.5, 0.7, 0.8],   # e.g., Intake
        [0.2, 0.3, 0.6, 0.8, 0.9],   # Nursing
        [0.1, 0.2, 0.4, 0.5, 0.6],   # Physicians
    ]

    error_series = [0.3, 0.4, 0.6, 0.5, 0.35]  # SECL toy error magnitudes
    baseline_hist = [10, 20, 30, 40]
    current_hist = [8, 18, 35, 50]
    hcm_threshold = 10.0  # arbitrary

    # MEO example: 2D opportunity space
    x_t = [0.0, 0.0]
    grad_u = [0.5, 0.2]
    eta = 0.1

    # ------------------------------------------------------------------
    # First, build a profile once to get CSA-derived cohesion
    # ------------------------------------------------------------------
    from huscot import compute_csa, compute_entropy, compute_wcs_dr

    csa_temp = compute_csa(csa_layers)
    cohesion = csa_temp.integration_coefficient  # simple proxy

    entropy = compute_entropy(entropy_probs)
    wcs_temp = compute_wcs_dr(cohesion=cohesion, entropy=entropy)

    print("Initial CSA Integration Coefficient (cohesion proxy):",
          round(csa_temp.integration_coefficient, 3))
    print("Entropy (toy):", round(entropy, 3))
    print("Dissolution Risk (WCS/DR):", round(wcs_temp.dissolution_risk, 3))

    # ------------------------------------------------------------------
    # Now compute a full HuSCoT profile with all available modules
    # ------------------------------------------------------------------
    profile = build_huscot_profile(
        csa_layers=csa_layers,
        capacities=capacities,
        loads=loads,
        cohesion=cohesion,
        entropy_probs=entropy_probs,
        pressure_field=pressure_field,
        error_series=error_series,
        max_envelope=0.6,
        baseline_hist=baseline_hist,
        current_hist=current_hist,
        hcm_threshold=hcm_threshold,
        x_t=x_t,
        grad_u=grad_u,
        eta=eta,
    )

    # ------------------------------------------------------------------
    # Print a compact summary
    # ------------------------------------------------------------------
    print("\n=== HuSCoT Profile Summary ===")
    print("FI (mean):", round(profile.csa.fi_mean, 3))
    print("IC (integration coefficient):",
          round(profile.csa.integration_coefficient, 3))
    print("WSA overload ratios:", {
        k: round(v, 2) for k, v in profile.wsa.overload_ratios.items()
    })
    if profile.csm is not None:
        print("Interface Tension (CSM):", round(profile.csm.interface_tension, 3))
    print("Dissolution Risk (WCS/DR):", round(profile.wcs.dissolution_risk, 3))
    if profile.secl is not None:
        print("SECL stable?:", profile.secl.stable)
        print("SECL control effort:", round(profile.secl.control_effort, 3))
    if profile.hcm is not None:
        print("HCM divergence:", round(profile.hcm.divergence, 3))
        print("HCM drift detected?:", profile.hcm.drift_detected)
    if profile.meo is not None:
        print("MEO next state x_(t+1):", [round(x, 3) for x in profile.meo.x_next])


if __name__ == "__main__":
    main()
