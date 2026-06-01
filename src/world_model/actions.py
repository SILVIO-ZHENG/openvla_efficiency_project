"""Action representation for the tiny world model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotAction:
    """Candidate robot action proposed by a policy."""

    name: str
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0
    gripper_open: bool | None = None
