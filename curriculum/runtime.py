"""Curriculum runtime implementations."""

from typing import Protocol, Sequence

from curriculum.models import CurriculumStage


class Curriculum(Protocol):
    """Small runtime contract for staged curricula."""

    def initialize(self) -> CurriculumStage:
        """Return the initial stage and mark it active."""

    def maybe_advance(self, num_timesteps: int) -> CurriculumStage | None:
        """Return a new stage when the curriculum crosses a boundary."""

    def stage_for_timesteps(self, num_timesteps: int) -> CurriculumStage:
        """Resolve which stage should be active at ``num_timesteps``."""


class LinearCurriculum:
    """A fixed ordered list of stages with timestep boundaries."""

    def __init__(self, stages: Sequence[CurriculumStage]):
        if not stages:
            raise ValueError("LinearCurriculum requires at least one stage.")

        ordered_stages = list(stages)
        if ordered_stages[0].start_timestep != 0:
            raise ValueError("The first curriculum stage must start at timestep 0.")

        previous_end = 0
        for index, stage in enumerate(ordered_stages):
            if stage.start_timestep != previous_end:
                raise ValueError(
                    "Curriculum stages must be contiguous and ordered by timestep."
                )
            if stage.end_timestep is None and index != len(ordered_stages) - 1:
                raise ValueError("Only the final curriculum stage may omit end_timestep.")
            previous_end = stage.end_timestep if stage.end_timestep is not None else previous_end

        self._stages = tuple(ordered_stages)
        self._active_index: int | None = None

    @property
    def stages(self) -> tuple[CurriculumStage, ...]:
        """Return the ordered curriculum stages."""
        return self._stages

    def initialize(self) -> CurriculumStage:
        """Activate and return the first stage."""
        self._active_index = 0
        return self._stages[0]

    def stage_for_timesteps(self, num_timesteps: int) -> CurriculumStage:
        """Resolve the stage active at ``num_timesteps``."""
        if num_timesteps < 0:
            raise ValueError("num_timesteps must be >= 0.")

        for stage in self._stages:
            if stage.contains(num_timesteps):
                return stage
        return self._stages[-1]

    def maybe_advance(self, num_timesteps: int) -> CurriculumStage | None:
        """Advance once when crossing a stage boundary."""
        stage = self.stage_for_timesteps(num_timesteps)
        stage_index = self._stages.index(stage)

        if self._active_index is None:
            self._active_index = stage_index
            return stage

        if stage_index == self._active_index:
            return None

        self._active_index = stage_index
        return stage
