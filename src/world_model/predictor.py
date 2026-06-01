"""Tiny world model for next-state prediction and risk assessment."""

from dataclasses import dataclass

from .actions import RobotAction
from .safety import SafetyAssessment, assess_action_safety
from .state import WorldState


@dataclass(frozen=True)
class TinyWorldModel:
    """Rule-based baseline for action-conditioned world-state prediction."""

    min_z: float = 0.0
    max_step: float = 0.25

    def predict_next_state(self, state: WorldState, action: RobotAction) -> WorldState:
        """Predict the next world state after applying a candidate action."""

        object_in_gripper = state.object_in_gripper
        if action.gripper_open is True:
            object_in_gripper = False
        elif action.gripper_open is False:
            object_in_gripper = True

        return WorldState(
            end_effector_x=state.end_effector_x + action.delta_x,
            end_effector_y=state.end_effector_y + action.delta_y,
            end_effector_z=state.end_effector_z + action.delta_z,
            object_in_gripper=object_in_gripper,
            metadata=dict(state.metadata),
        )

    def assess_safety(self, state: WorldState, action: RobotAction) -> SafetyAssessment:
        """Return a safe/risky label for the candidate action."""

        return assess_action_safety(
            state=state,
            action=action,
            min_z=self.min_z,
            max_step=self.max_step,
        )

    def predict(self, state: WorldState, action: RobotAction) -> tuple[WorldState, SafetyAssessment]:
        """Predict next state and safety assessment together."""

        return self.predict_next_state(state, action), self.assess_safety(state, action)
