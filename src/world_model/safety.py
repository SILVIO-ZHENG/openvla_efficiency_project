"""Rule-based safety assessment for candidate actions."""

from dataclasses import dataclass

from .actions import RobotAction
from .state import WorldState


@dataclass(frozen=True)
class SafetyAssessment:
    """Safety label and short explanation for a candidate action."""

    is_safe: bool
    risk_label: str
    reason: str


def assess_action_safety(
    state: WorldState,
    action: RobotAction,
    min_z: float = 0.0,
    max_step: float = 0.25,
) -> SafetyAssessment:
    """Assess whether an action is safe using simple geometric rules."""

    next_z = state.end_effector_z + action.delta_z
    max_delta = max(abs(action.delta_x), abs(action.delta_y), abs(action.delta_z))

    if next_z < min_z:
        return SafetyAssessment(
            is_safe=False,
            risk_label="risky",
            reason="predicted end-effector height is below the minimum safe height",
        )

    if max_delta > max_step:
        return SafetyAssessment(
            is_safe=False,
            risk_label="risky",
            reason="candidate action step is larger than the configured safety limit",
        )

    return SafetyAssessment(
        is_safe=True,
        risk_label="safe",
        reason="candidate action satisfies the current rule-based safety checks",
    )
