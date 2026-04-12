"""Tests for the curriculum callback."""

import unittest
from unittest.mock import MagicMock

from curriculum.models import CurriculumStage, OpponentPlayerSpec
from training.curriculum_callback import CurriculumCallback


class DummyEnv:
    def __init__(self):
        self.calls = []

    def schedule_opponent_player(self, spec):
        self.calls.append(spec)


class DummyVecEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, name, *args):
        self.calls.append((name, args))


class TestCurriculumCallback(unittest.TestCase):
    """Verify curriculum stages are forwarded to the env."""

    def test_on_training_start_schedules_initial_stage(self):
        stage = CurriculumStage(
            name="warmup",
            start_timestep=0,
            end_timestep=10,
            opponent_player=OpponentPlayerSpec(id="random"),
        )
        curriculum = MagicMock()
        curriculum.initialize.return_value = stage
        model = MagicMock()
        env = DummyEnv()
        model.get_env.return_value = env
        model.num_timesteps = 0
        callback = CurriculumCallback(curriculum=curriculum)
        callback.init_callback(model)

        callback.on_training_start({}, {})

        self.assertEqual(env.calls, [stage.opponent_player])

    def test_on_step_uses_env_method_for_vector_envs(self):
        stage = CurriculumStage(
            name="midgame",
            start_timestep=10,
            end_timestep=None,
            opponent_player=OpponentPlayerSpec(id="heuristic"),
        )
        curriculum = MagicMock()
        curriculum.initialize.return_value = stage
        curriculum.maybe_advance.return_value = stage
        model = MagicMock()
        env = DummyVecEnv()
        model.get_env.return_value = env
        model.num_timesteps = 10
        callback = CurriculumCallback(curriculum=curriculum)
        callback.init_callback(model)

        callback.on_step()

        self.assertEqual(
            env.calls,
            [("schedule_opponent_player", (stage.opponent_player,))],
        )
