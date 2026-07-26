"""Candidate maintenance policies for Problem 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.problem2.model import FilterParams, MaintenanceSchedule
from src.problem2.simulate import MaintenanceDecision, seasonal_effect


@dataclass(frozen=True)
class PolicySpec:
    """Serializable description of one candidate maintenance policy."""

    candidate_id: str
    family: str
    label: str
    medium_gap: float | None = None
    major_gap: float | None = None
    medium_between_major: int = 4
    medium_trigger: float | None = None
    major_trigger: float | None = None
    fouling_trigger: float | None = None
    reserve_threshold: float | None = None
    min_gap: int = 21
    max_gap: int = 150

    def to_record(self) -> dict:
        return asdict(self)


def generate_candidate_specs() -> list[PolicySpec]:
    """Generate a compact, interpretable grid for three policy families."""
    specs: list[PolicySpec] = []
    candidate = 0

    for gap in (45, 60, 75, 90, 120):
        for medium_between_major in (2, 4, 6):
            candidate += 1
            specs.append(
                PolicySpec(
                    candidate_id=f"F{candidate:02d}",
                    family="固定周期",
                    label=f"每{gap}日维护，{medium_between_major}次中维护后大维护",
                    medium_gap=float(gap),
                    major_gap=float(gap),
                    medium_between_major=medium_between_major,
                )
            )

    threshold_pairs = ((55, 42), (65, 47), (75, 52))
    for medium_trigger, major_trigger in threshold_pairs:
        for min_gap in (21, 35):
            for max_gap in (90, 150):
                candidate += 1
                specs.append(
                    PolicySpec(
                        candidate_id=f"T{candidate:02d}",
                        family="透水率触发",
                        label=(
                            f"7日均值≤{medium_trigger}/{major_trigger}触发中/大维护，"
                            f"冷却{min_gap}日，上限{max_gap}日"
                        ),
                        medium_trigger=float(medium_trigger),
                        major_trigger=float(major_trigger),
                        medium_between_major=4,
                        min_gap=min_gap,
                        max_gap=max_gap,
                    )
                )

    for fouling_trigger in (10, 20, 30):
        for reserve_threshold in (50, 60):
            for min_gap in (21, 35):
                candidate += 1
                specs.append(
                    PolicySpec(
                        candidate_id=f"S{candidate:02d}",
                        family="状态触发",
                        label=(
                            f"堵塞损失≥{fouling_trigger}触发，维护后储备阈值"
                            f"{reserve_threshold}，冷却{min_gap}日"
                        ),
                        fouling_trigger=float(fouling_trigger),
                        reserve_threshold=float(reserve_threshold),
                        medium_between_major=4,
                        min_gap=min_gap,
                        max_gap=150,
                    )
                )
    return specs


def schedule_from_spec(
    spec: PolicySpec,
    baseline: MaintenanceSchedule,
) -> MaintenanceSchedule:
    """Convert a fixed-cycle candidate into the Problem 2 schedule interface."""
    if spec.family != "固定周期":
        return baseline
    return MaintenanceSchedule(
        device=baseline.device,
        medium_gap_mean=float(spec.medium_gap),
        medium_gap_std=0.0,
        major_gap_mean=float(spec.major_gap),
        major_gap_std=0.0,
        medium_between_major=int(spec.medium_between_major),
    )


def decision_from_spec(spec: PolicySpec) -> MaintenanceDecision | None:
    """Build a stateless condition-based decision rule."""
    if spec.family == "固定周期":
        return None

    if spec.family == "透水率触发":

        def threshold_decision(context: dict) -> str | None:
            if context["days_since_event"] < spec.min_gap:
                return None
            level = context["recent_7d_mean"]
            if level <= float(spec.major_trigger):
                return "大维护"
            if level <= float(spec.medium_trigger):
                return "中维护"
            if context["days_since_event"] >= spec.max_gap:
                if context["mediums_since_major"] >= spec.medium_between_major:
                    return "大维护"
                return "中维护"
            return None

        return threshold_decision

    if spec.family == "状态触发":

        def state_decision(context: dict) -> str | None:
            if context["days_since_event"] < spec.min_gap:
                return None
            if (
                context["F"] < float(spec.fouling_trigger)
                and context["days_since_event"] < spec.max_gap
            ):
                return None

            params: FilterParams = context["params"]
            season = seasonal_effect(context["day_of_year"], params)
            post_medium = (
                context["C"]
                - params.medium_damage
                - max(context["F"] - params.medium_recovery, 0.0)
                + season
            )
            post_major = (
                context["C"]
                - params.major_damage
                - max(context["F"] - params.major_recovery, 0.0)
                + season
            )
            major_has_material_advantage = post_major >= post_medium + 1.0
            reserve_is_low = post_medium < float(spec.reserve_threshold)
            major_cycle_due = (
                context["mediums_since_major"] >= spec.medium_between_major
            )
            if major_has_material_advantage and (reserve_is_low or major_cycle_due):
                return "大维护"
            return "中维护"

        return state_decision

    raise ValueError(f"Unknown policy family: {spec.family}")
