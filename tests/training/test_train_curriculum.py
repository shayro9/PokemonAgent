"""Curriculum-specific tests for training.train."""

import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from curriculum.models import CurriculumStage, OpponentPlayerSpec
from training.train import train_model, _print_curriculum_summary
from tests.conftest import MockTeamGenerator, make_mock_model


class TestTrainModelCurriculum(unittest.TestCase):
    """Verify train_model wires curricula into env creation and callbacks."""

    @patch("training.train.CurriculumCallback")
    @patch("training.train.WandbCallback")
    @patch("training.train.BattleMetricsCallback")
    @patch("training.train.wandb")
    @patch("training.train.build_env")
    @patch("training.train.MaskablePPO")
    @patch("training.train.set_random_seed")
    def test_train_model_passes_initial_curriculum_spec_to_env(
        self,
        mock_set_seed,
        mock_ppo_class,
        mock_build_env,
        mock_wandb,
        mock_metrics_cb,
        mock_wandb_cb,
        mock_curriculum_cb,
    ):
        env = MagicMock()
        mock_build_env.return_value = env
        model = make_mock_model()
        mock_ppo_class.return_value = model
        mock_wandb.init.return_value = MagicMock()
        stage = CurriculumStage(
            name="warmup",
            start_timestep=0,
            end_timestep=None,
            opponent_player=OpponentPlayerSpec(id="random"),
        )
        curriculum = MagicMock()
        curriculum.stage_for_timesteps.return_value = stage

        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=MockTeamGenerator(["opp1"]),
            timesteps=100,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
            curriculum=curriculum,
        )

        self.assertEqual(
            mock_build_env.call_args[1]["opponent_player_spec"],
            stage.opponent_player,
        )
        mock_curriculum_cb.assert_called_once_with(curriculum=curriculum)

    @patch("training.train.CurriculumCallback")
    @patch("training.train.WandbCallback")
    @patch("training.train.BattleMetricsCallback")
    @patch("training.train.wandb")
    @patch("training.train.build_env")
    @patch("training.train.MaskablePPO")
    @patch("training.train.set_random_seed")
    def test_train_model_adds_curriculum_callback_to_learn(
        self,
        mock_set_seed,
        mock_ppo_class,
        mock_build_env,
        mock_wandb,
        mock_metrics_cb,
        mock_wandb_cb,
        mock_curriculum_cb,
    ):
        env = MagicMock()
        mock_build_env.return_value = env
        model = make_mock_model()
        mock_ppo_class.return_value = model
        mock_wandb.init.return_value = MagicMock()
        curriculum = MagicMock()
        curriculum.stage_for_timesteps.return_value = CurriculumStage(
            name="warmup",
            start_timestep=0,
            end_timestep=None,
            opponent_player=OpponentPlayerSpec(id="random"),
        )

        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=MockTeamGenerator(["opp1"]),
            timesteps=100,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
            curriculum=curriculum,
        )

        callbacks = model.learn.call_args[1]["callback"]
        self.assertEqual(callbacks[0], mock_curriculum_cb.return_value)


class TestCurriculumPrinting(unittest.TestCase):
    """Verify curriculum startup logging is readable."""

    def test_print_curriculum_summary_lists_all_stages(self):
        curriculum = MagicMock()
        curriculum.stages = [
            CurriculumStage(
                name="warmup",
                start_timestep=0,
                end_timestep=10,
                opponent_player=OpponentPlayerSpec(id="random"),
            ),
            CurriculumStage(
                name="endgame",
                start_timestep=10,
                end_timestep=None,
                opponent_player=OpponentPlayerSpec(id="heuristic"),
            ),
        ]

        with patch("sys.stdout", new=StringIO()) as fake_stdout:
            _print_curriculum_summary(curriculum)
            output = fake_stdout.getvalue()

        self.assertIn("Curriculum stages:", output)
        self.assertIn("warmup: [0, 10) -> random", output)
        self.assertIn("endgame: [10, end) -> heuristic", output)

    @patch("training.train.CurriculumCallback")
    @patch("training.train.WandbCallback")
    @patch("training.train.BattleMetricsCallback")
    @patch("training.train.wandb")
    @patch("training.train.build_env")
    @patch("training.train.MaskablePPO")
    @patch("training.train.set_random_seed")
    def test_train_model_prints_curriculum_at_start(
        self,
        mock_set_seed,
        mock_ppo_class,
        mock_build_env,
        mock_wandb,
        mock_metrics_cb,
        mock_wandb_cb,
        mock_curriculum_cb,
    ):
        env = MagicMock()
        mock_build_env.return_value = env
        model = make_mock_model()
        mock_ppo_class.return_value = model
        mock_wandb.init.return_value = MagicMock()
        stage = CurriculumStage(
            name="warmup",
            start_timestep=0,
            end_timestep=None,
            opponent_player=OpponentPlayerSpec(id="random"),
        )
        curriculum = MagicMock()
        curriculum.stage_for_timesteps.return_value = stage
        curriculum.stages = [stage]

        with patch("sys.stdout", new=StringIO()) as fake_stdout:
            train_model(
                model_path="test_model",
                battle_format="gen1ou",
                opponent_generator=MockTeamGenerator(["opp1"]),
                timesteps=100,
                rounds_per_opponent=10,
                eval_every_timesteps=0,
                seed=42,
                curriculum=curriculum,
            )
            output = fake_stdout.getvalue()

        self.assertIn("Curriculum stages:", output)
        self.assertIn("warmup: [0, end) -> random", output)
