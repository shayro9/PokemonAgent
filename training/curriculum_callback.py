"""SB3 callback for curriculum-driven opponent switching."""

from stable_baselines3.common.callbacks import BaseCallback

from curriculum.models import CurriculumStage
from curriculum.runtime import Curriculum


class CurriculumCallback(BaseCallback):
    """Advance the curriculum and schedule opponent changes on the env."""

    def __init__(self, curriculum: Curriculum, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.curriculum = curriculum

    def _on_training_start(self) -> None:
        self._apply_stage(self.curriculum.initialize())

    def _on_step(self) -> bool:
        stage = self.curriculum.maybe_advance(self.num_timesteps)
        if stage is not None:
            self._apply_stage(stage)
        return True

    def _apply_stage(self, stage: CurriculumStage) -> None:
        env = self.training_env
        if hasattr(env, "env_method"):
            env.env_method("schedule_opponent_player", stage.opponent_player)
        elif hasattr(env, "schedule_opponent_player"):
            env.schedule_opponent_player(stage.opponent_player)
        else:
            raise TypeError(
                "CurriculumCallback requires an environment built by env.env_builder."
            )

        if self.verbose:
            print(
                f"[Curriculum] Stage '{stage.name}' active at timestep {self.num_timesteps} "
                f"-> opponent {stage.opponent_player.identifier}"
            )
