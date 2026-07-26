from __future__ import annotations

from src.problem2.model import FilterParams
from src.problem3.policies import PolicySpec, decision_from_spec


def _params() -> FilterParams:
    return FilterParams(
        device="A1",
        alpha=0.02,
        beta=0.2,
        seasonal_sin_1y=0.0,
        seasonal_cos_1y=0.0,
        seasonal_sin_2y=0.0,
        seasonal_cos_2y=0.0,
        medium_recovery=15.0,
        major_recovery=30.0,
        medium_damage=0.2,
        major_damage=0.6,
        sigma=1.0,
        recent_level=80.0,
        days_since_maintenance=0.0,
        mediums_since_major=0,
        last_maintenance_type="中维护",
        historical_irreversible_decline_per_year=10.0,
        damage_share_assumption=0.2,
    )


def _context(**updates) -> dict:
    values = {
        "day": 1000,
        "day_of_year": 100,
        "C": 80.0,
        "F": 20.0,
        "permeability": 60.0,
        "recent_7d_mean": 60.0,
        "annual_mean": 70.0,
        "days_since_event": 40,
        "mediums_since_major": 2,
        "params": _params(),
    }
    values.update(updates)
    return values


def test_threshold_policy_respects_cooldown_and_thresholds() -> None:
    spec = PolicySpec(
        candidate_id="T",
        family="透水率触发",
        label="test",
        medium_trigger=65.0,
        major_trigger=45.0,
        min_gap=21,
        max_gap=150,
    )
    decision = decision_from_spec(spec)
    assert decision(_context(days_since_event=10, recent_7d_mean=40.0)) is None
    assert decision(_context(recent_7d_mean=60.0)) == "中维护"
    assert decision(_context(recent_7d_mean=40.0)) == "大维护"


def test_state_policy_chooses_major_only_when_it_adds_value() -> None:
    spec = PolicySpec(
        candidate_id="S",
        family="状态触发",
        label="test",
        fouling_trigger=10.0,
        reserve_threshold=70.0,
        min_gap=21,
    )
    decision = decision_from_spec(spec)
    assert decision(_context(F=25.0)) == "大维护"
    weak_major = _params()
    weak_major.major_recovery = 10.0
    assert decision(_context(F=25.0, params=weak_major)) == "中维护"
