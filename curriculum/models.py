"""Data models for curriculum-driven training."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpponentPlayerSpec:
    """Describe which opponent Player implementation to use for a stage.

    Exactly one of ``id`` or ``class_path`` must be provided.
    """

    id: str | None = None
    class_path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if (self.id is None) == (self.class_path is None):
            raise ValueError("Exactly one of 'id' or 'class_path' must be provided.")
        if self.id is not None and not self.id.strip():
            raise ValueError("'id' must be a non-empty string.")
        if self.class_path is not None and not self.class_path.strip():
            raise ValueError("'class_path' must be a non-empty string.")
        object.__setattr__(self, "kwargs", dict(self.kwargs))

    @property
    def identifier(self) -> str:
        """Return the registered id or import path for this player spec."""
        return self.id if self.id is not None else self.class_path  # type: ignore[return-value]


@dataclass(frozen=True)
class CurriculumStage:
    """A single curriculum stage with timestep boundaries."""

    name: str
    start_timestep: int
    end_timestep: int | None
    opponent_player: OpponentPlayerSpec

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Stage name must be a non-empty string.")
        if self.start_timestep < 0:
            raise ValueError("Stage start_timestep must be >= 0.")
        if self.end_timestep is not None and self.end_timestep <= self.start_timestep:
            raise ValueError("Stage end_timestep must be greater than start_timestep.")

    def contains(self, num_timesteps: int) -> bool:
        """Return whether ``num_timesteps`` falls inside this stage."""
        if num_timesteps < self.start_timestep:
            return False
        if self.end_timestep is None:
            return True
        return num_timesteps < self.end_timestep
