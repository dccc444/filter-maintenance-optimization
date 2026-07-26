"""Monte Carlo evaluation and selection of maintenance policies."""

from __future__ import annotations

import numpy as np

from src.problem2.model import FilterParams, MaintenanceSchedule
from src.problem2.simulate import monte_carlo_simulate, summarise_results

from .policies import PolicySpec, decision_from_spec, schedule_from_spec


def summarise_policy_results(
    results: list[dict],
    device: str,
    strategy: str,
    spec: PolicySpec | None,
) -> dict:
    """Estimate renewal-equivalent annual cost and reliability metrics."""
    lifetime = np.asarray(
        [result["total_lifetime_days"] for result in results], dtype=float
    )
    years = lifetime / 365.25
    costs = np.asarray([result["total_cost"] for result in results], dtype=float)
    medium = np.asarray(
        [result["medium_maintenance_count"] for result in results], dtype=float
    )
    major = np.asarray(
        [result["major_maintenance_count"] for result in results], dtype=float
    )
    observed = np.asarray([result["event_observed"] for result in results], dtype=bool)
    base = summarise_results(results, device)
    total_years = float(years.sum())
    record = {
        "device": device,
        "strategy": strategy,
        "candidate_id": spec.candidate_id if spec else "BASELINE",
        "family": spec.family if spec else "当前固定方案",
        "policy_label": spec.label if spec else "按历史固定维护规律继续",
        "n_runs": len(results),
        "renewal_annual_cost": float(costs.sum() / total_years),
        "median_annual_cost": float(
            np.median([result["annualized_cost"] for result in results])
        ),
        "p90_annual_cost": float(
            np.quantile([result["annualized_cost"] for result in results], 0.90)
        ),
        "median_lifetime_days": base["median_lifetime_days"],
        "p10_lifetime_days": float(np.quantile(lifetime, 0.10)),
        "p90_lifetime_days": float(np.quantile(lifetime, 0.90)),
        "right_censored_share": float((~observed).mean()),
        "eol_within_2y_probability": float(
            np.mean(observed & (lifetime <= 2 * 365.25))
        ),
        "medium_maintenances_per_year": float(medium.sum() / total_years),
        "major_maintenances_per_year": float(major.sum() / total_years),
        "mean_simulated_permeability": float(
            np.mean([result["mean_simulated_permeability"] for result in results])
        ),
        "mean_annual_below_threshold_days": float(
            np.mean([result["annual_below_threshold_days"] for result in results])
        ),
    }
    if spec:
        record.update(spec.to_record())
    return record


def evaluate_policy(
    params: FilterParams,
    baseline_schedule: MaintenanceSchedule,
    warmup_history: list[float],
    strategy: str,
    spec: PolicySpec | None,
    n_runs: int,
    max_years: int,
    seed: int = 2026,
) -> dict:
    """Run one candidate with common random numbers."""
    schedule = (
        schedule_from_spec(spec, baseline_schedule)
        if spec is not None
        else baseline_schedule
    )
    decision = decision_from_spec(spec) if spec is not None else None
    results, _ = monte_carlo_simulate(
        params,
        schedule,
        n_runs=n_runs,
        seed=seed,
        warmup_history=warmup_history,
        max_years=max_years,
        maintenance_decision=decision,
    )
    return summarise_policy_results(results, params.device, strategy, spec)


def choose_best_specs(
    records: list[dict],
    specs: list[PolicySpec],
) -> dict[tuple[str, str], PolicySpec]:
    """Choose the minimum renewal-cost candidate for each device and family."""
    spec_lookup = {spec.candidate_id: spec for spec in specs}
    best: dict[tuple[str, str], PolicySpec] = {}
    devices = sorted({record["device"] for record in records})
    families = sorted({record["family"] for record in records})
    for device in devices:
        for family in families:
            subset = [
                record
                for record in records
                if record["device"] == device and record["family"] == family
            ]
            feasible = [
                record for record in subset if record["right_censored_share"] <= 0.20
            ]
            candidates = feasible or subset
            winner = min(
                candidates,
                key=lambda record: (
                    record["renewal_annual_cost"],
                    -record["median_lifetime_days"],
                ),
            )
            best[(device, family)] = spec_lookup[winner["candidate_id"]]
    return best
