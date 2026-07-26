"""Pricing identities and scenario construction for Problem 4."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_PURCHASE_COST = 300.0
BASE_MEDIUM_COST = 3.0
BASE_MAJOR_COST = 12.0


def add_renewal_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover replacement intensity from the baseline renewal-cost identity."""
    result = frame.copy()
    maintenance_component = (
        BASE_MEDIUM_COST * result["medium_maintenances_per_year"]
        + BASE_MAJOR_COST * result["major_maintenances_per_year"]
    )
    result["replacement_rate_per_year"] = (
        result["renewal_annual_cost"] - maintenance_component
    ) / BASE_PURCHASE_COST
    if (result["replacement_rate_per_year"] < -1e-9).any():
        raise ValueError("Negative replacement rate reconstructed from candidate table")
    result["replacement_rate_per_year"] = result[
        "replacement_rate_per_year"
    ].clip(lower=0.0)
    return result


def reprice(
    frame: pd.DataFrame,
    purchase_cost: float,
    medium_cost: float,
    major_cost: float,
) -> pd.Series:
    """Reprice fixed simulated trajectories without changing their dynamics."""
    return (
        purchase_cost * frame["replacement_rate_per_year"]
        + medium_cost * frame["medium_maintenances_per_year"]
        + major_cost * frame["major_maintenances_per_year"]
    )


def coarse_price_scenarios() -> pd.DataFrame:
    """Coarse grid used only to identify candidates requiring full MC review."""
    factors = (0.5, 0.75, 1.0, 1.25, 1.5)
    records = []
    for factor in factors:
        records.extend(
            [
                {
                    "scenario_id": f"purchase_{factor:.2f}",
                    "scenario_type": "购置价单因素",
                    "purchase_factor": factor,
                    "medium_factor": 1.0,
                    "major_factor": 1.0,
                },
                {
                    "scenario_id": f"medium_{factor:.2f}",
                    "scenario_type": "中维护价单因素",
                    "purchase_factor": 1.0,
                    "medium_factor": factor,
                    "major_factor": 1.0,
                },
                {
                    "scenario_id": f"major_{factor:.2f}",
                    "scenario_type": "大维护价单因素",
                    "purchase_factor": 1.0,
                    "medium_factor": 1.0,
                    "major_factor": factor,
                },
            ]
        )
    for purchase_factor in factors:
        for maintenance_factor in factors:
            records.append(
                {
                    "scenario_id": (
                        f"purchase_maintenance_{purchase_factor:.2f}_"
                        f"{maintenance_factor:.2f}"
                    ),
                    "scenario_type": "购置价×维护价",
                    "purchase_factor": purchase_factor,
                    "medium_factor": maintenance_factor,
                    "major_factor": maintenance_factor,
                }
            )
    for medium_factor in factors:
        for major_factor in factors:
            records.append(
                {
                    "scenario_id": (
                        f"medium_major_{medium_factor:.2f}_{major_factor:.2f}"
                    ),
                    "scenario_type": "中维护价×大维护价",
                    "purchase_factor": 1.0,
                    "medium_factor": medium_factor,
                    "major_factor": major_factor,
                }
            )
    return _with_prices(pd.DataFrame(records))


def full_price_scenarios() -> pd.DataFrame:
    """Fine grids used for the final sensitivity surfaces."""
    single_factors = np.round(np.linspace(0.5, 1.5, 21), 2)
    double_factors = np.round(np.linspace(0.5, 1.5, 11), 2)
    records = []
    for factor in single_factors:
        records.extend(
            [
                {
                    "scenario_id": f"purchase_{factor:.2f}",
                    "scenario_type": "购置价单因素",
                    "purchase_factor": factor,
                    "medium_factor": 1.0,
                    "major_factor": 1.0,
                },
                {
                    "scenario_id": f"medium_{factor:.2f}",
                    "scenario_type": "中维护价单因素",
                    "purchase_factor": 1.0,
                    "medium_factor": factor,
                    "major_factor": 1.0,
                },
                {
                    "scenario_id": f"major_{factor:.2f}",
                    "scenario_type": "大维护价单因素",
                    "purchase_factor": 1.0,
                    "medium_factor": 1.0,
                    "major_factor": factor,
                },
            ]
        )
    for purchase_factor in double_factors:
        for maintenance_factor in double_factors:
            records.append(
                {
                    "scenario_id": (
                        f"purchase_maintenance_{purchase_factor:.2f}_"
                        f"{maintenance_factor:.2f}"
                    ),
                    "scenario_type": "购置价×维护价",
                    "purchase_factor": purchase_factor,
                    "medium_factor": maintenance_factor,
                    "major_factor": maintenance_factor,
                }
            )
    for medium_factor in double_factors:
        for major_factor in double_factors:
            records.append(
                {
                    "scenario_id": (
                        f"medium_major_{medium_factor:.2f}_{major_factor:.2f}"
                    ),
                    "scenario_type": "中维护价×大维护价",
                    "purchase_factor": 1.0,
                    "medium_factor": medium_factor,
                    "major_factor": major_factor,
                }
            )
    return _with_prices(pd.DataFrame(records))


def _with_prices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["purchase_cost"] = BASE_PURCHASE_COST * result["purchase_factor"]
    result["medium_cost"] = BASE_MEDIUM_COST * result["medium_factor"]
    result["major_cost"] = BASE_MAJOR_COST * result["major_factor"]
    return result


def select_top_pairs(
    screening: pd.DataFrame,
    scenarios: pd.DataFrame,
    top_n: int = 3,
) -> set[tuple[str, str]]:
    """Select top candidates per device/scenario for higher precision review."""
    candidates = add_renewal_rates(screening)
    pairs: set[tuple[str, str]] = set()
    for scenario in scenarios.itertuples(index=False):
        priced = candidates.assign(
            repriced_cost=reprice(
                candidates,
                scenario.purchase_cost,
                scenario.medium_cost,
                scenario.major_cost,
            )
        ).sort_values(["device", "repriced_cost"])
        top = priced.groupby("device", sort=False).head(top_n)
        pairs.update(zip(top["device"], top["candidate_id"]))
    return pairs


def optimise_scenarios(
    refined: pd.DataFrame,
    scenarios: pd.DataFrame,
    original_policy: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-optimise every device and calculate original-policy regret."""
    candidates = add_renewal_rates(refined)
    device_records = []
    summary_records = []
    for scenario in scenarios.itertuples(index=False):
        scenario_rows = []
        for device, group in candidates.groupby("device", sort=False):
            priced = group.assign(
                scenario_cost=reprice(
                    group,
                    scenario.purchase_cost,
                    scenario.medium_cost,
                    scenario.major_cost,
                )
            )
            optimum = priced.sort_values(
                ["scenario_cost", "median_lifetime_days"],
                ascending=[True, False],
            ).iloc[0]
            original = priced.loc[
                priced["candidate_id"] == original_policy[device]
            ].iloc[0]
            regret = float(original["scenario_cost"] - optimum["scenario_cost"])
            record = {
                "scenario_id": scenario.scenario_id,
                "scenario_type": scenario.scenario_type,
                "purchase_factor": scenario.purchase_factor,
                "medium_factor": scenario.medium_factor,
                "major_factor": scenario.major_factor,
                "purchase_cost": scenario.purchase_cost,
                "medium_cost": scenario.medium_cost,
                "major_cost": scenario.major_cost,
                "device": device,
                "optimal_candidate_id": optimum["candidate_id"],
                "optimal_family": optimum["family"],
                "optimal_policy_label": optimum["policy_label"],
                "optimal_annual_cost": float(optimum["scenario_cost"]),
                "original_candidate_id": original_policy[device],
                "original_annual_cost": float(original["scenario_cost"]),
                "absolute_regret": max(regret, 0.0),
                "relative_regret": max(
                    regret / float(optimum["scenario_cost"]), 0.0
                ),
                "replacement_rate_per_year": float(
                    optimum["replacement_rate_per_year"]
                ),
                "medium_maintenances_per_year": float(
                    optimum["medium_maintenances_per_year"]
                ),
                "major_maintenances_per_year": float(
                    optimum["major_maintenances_per_year"]
                ),
                "median_lifetime_days": float(optimum["median_lifetime_days"]),
                "eol_within_2y_probability": float(
                    optimum["eol_within_2y_probability"]
                ),
            }
            scenario_rows.append(record)
            device_records.append(record)

        rows = pd.DataFrame(scenario_rows)
        optimal_cost = float(rows["optimal_annual_cost"].sum())
        original_cost = float(rows["original_annual_cost"].sum())
        family_counts = rows["optimal_family"].value_counts()
        summary_records.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_type": scenario.scenario_type,
                "purchase_factor": scenario.purchase_factor,
                "medium_factor": scenario.medium_factor,
                "major_factor": scenario.major_factor,
                "purchase_cost": scenario.purchase_cost,
                "medium_cost": scenario.medium_cost,
                "major_cost": scenario.major_cost,
                "optimal_factory_annual_cost": optimal_cost,
                "original_plan_annual_cost": original_cost,
                "absolute_regret": max(original_cost - optimal_cost, 0.0),
                "relative_regret": max(
                    (original_cost - optimal_cost) / optimal_cost, 0.0
                ),
                "original_plan_near_optimal_5pct": (
                    (original_cost - optimal_cost) / optimal_cost <= 0.05 + 1e-12
                ),
                "replacement_rate_per_year": float(
                    rows["replacement_rate_per_year"].sum()
                ),
                "medium_maintenances_per_year": float(
                    rows["medium_maintenances_per_year"].sum()
                ),
                "major_maintenances_per_year": float(
                    rows["major_maintenances_per_year"].sum()
                ),
                "average_median_lifetime_years": float(
                    rows["median_lifetime_days"].mean() / 365.25
                ),
                "fixed_policy_devices": int(family_counts.get("固定周期", 0)),
                "threshold_policy_devices": int(family_counts.get("透水率触发", 0)),
                "state_policy_devices": int(family_counts.get("状态触发", 0)),
            }
        )
    return pd.DataFrame(device_records), pd.DataFrame(summary_records)


def one_factor_applicability(summary: pd.DataFrame) -> pd.DataFrame:
    """Return multiplier intervals where the Problem 3 plan is within 5%."""
    factor_column = {
        "购置价单因素": "purchase_factor",
        "中维护价单因素": "medium_factor",
        "大维护价单因素": "major_factor",
    }
    records = []
    for scenario_type, factor in factor_column.items():
        subset = summary.loc[
            (summary["scenario_type"] == scenario_type)
            & summary["original_plan_near_optimal_5pct"]
        ]
        records.append(
            {
                "scenario_type": scenario_type,
                "factor": factor,
                "near_optimal_lower_factor": (
                    float(subset[factor].min()) if len(subset) else np.nan
                ),
                "near_optimal_upper_factor": (
                    float(subset[factor].max()) if len(subset) else np.nan
                ),
                "near_optimal_lower_price": (
                    float(
                        subset[
                            {
                                "purchase_factor": "purchase_cost",
                                "medium_factor": "medium_cost",
                                "major_factor": "major_cost",
                            }[factor]
                        ].min()
                    )
                    if len(subset)
                    else np.nan
                ),
                "near_optimal_upper_price": (
                    float(
                        subset[
                            {
                                "purchase_factor": "purchase_cost",
                                "medium_factor": "medium_cost",
                                "major_factor": "major_cost",
                            }[factor]
                        ].max()
                    )
                    if len(subset)
                    else np.nan
                ),
                "maximum_regret_in_tested_range": float(
                    summary.loc[
                        summary["scenario_type"] == scenario_type,
                        "relative_regret",
                    ].max()
                ),
            }
        )
    return pd.DataFrame(records)
