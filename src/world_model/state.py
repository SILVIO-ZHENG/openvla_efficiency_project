"""World-state representation for lightweight prediction."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldState:
    """Minimal structured state used by the tiny world model."""

    end_effector_x: float
    end_effector_y: float
    end_effector_z: float
    object_in_gripper: bool = False
    metadata: dict[str, float | str | bool] = field(default_factory=dict)
