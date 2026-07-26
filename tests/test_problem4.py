from __future__ import annotations

import pandas as pd

from src.problem4.sensitivity import add_renewal_rates, reprice


def _candidate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "renewal_annual_cost": [90.0],
            "medium_maintenances_per_year": [10.0],
            "major_maintenances_per_year": [2.0],
        }
    )


def test_replacement_rate_reconstructs_baseline_cost() -> None:
    candidate = add_renewal_rates(_candidate())
    assert abs(candidate.loc[0, "replacement_rate_per_year"] - 0.12) < 1e-12
    value = reprice(candidate, 300.0, 3.0, 12.0).iloc[0]
    assert abs(value - 90.0) < 1e-12


def test_repricing_responds_to_each_cost_component() -> None:
    candidate = add_renewal_rates(_candidate())
    base = reprice(candidate, 300.0, 3.0, 12.0).iloc[0]
    assert reprice(candidate, 600.0, 3.0, 12.0).iloc[0] > base
    assert reprice(candidate, 300.0, 6.0, 12.0).iloc[0] > base
    assert reprice(candidate, 300.0, 3.0, 24.0).iloc[0] > base
