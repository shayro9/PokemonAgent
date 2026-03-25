"""Unit tests for training.train module."""
import unittest
from unittest.mock import MagicMock, patch, call
import time

from training.train import train_model, LR, N_STEPS, BATCH_SIZE, GAMMA, ENT_COEF
from tests.conftest import MockTeamGenerator, make_mock_model


class TestTrainModel(unittest.TestCase):
    """Test train_model function."""

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_returns_model(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify train_model returns a trained MaskablePPO model."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        
        opponent_gen = MockTeamGenerator(["opp1"])
        
        result = train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            timesteps=100,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
        )
        
        self.assertEqual(result, model)

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_initializes_random_seed(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify random seeds are initialized."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        
        opponent_gen = MockTeamGenerator(["opp1"])
        
        with patch('training.train.random.seed'):
            with patch('training.train.np.random.seed'):
                train_model(
                    model_path="test_model",
                    battle_format="gen1ou",
                    opponent_generator=opponent_gen,
                    timesteps=100,
                    rounds_per_opponent=10,
                    eval_every_timesteps=0,
                    seed=42,
                )
        
        mock_set_seed.assert_called_once_with(42)

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_builds_environment(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify environment is built with correct parameters."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        
        opponent_gen = MockTeamGenerator(["opp1"])
        agent_gen = MockTeamGenerator(["agent1"])
        
        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            agent_team_generator=agent_gen,
            timesteps=100,
            rounds_per_opponent=500,
            eval_every_timesteps=0,
            seed=42,
        )
        
        mock_build_env.assert_called_once()
        # battle_format is positional arg 0
        call_args = mock_build_env.call_args[0]
        call_kwargs = mock_build_env.call_args[1]
        self.assertEqual(call_args[0], "gen1ou")  # battle_format
        self.assertEqual(call_args[1], opponent_gen)  # opponent_generator
        self.assertEqual(call_args[2], 500)  # rounds_per_opponent
        self.assertEqual(call_kwargs['agent_team_generator'], agent_gen)

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_creates_maskableppo_with_correct_policy(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify MaskablePPO is created with AttentionPointerPolicy."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        
        opponent_gen = MockTeamGenerator(["opp1"])
        
        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            timesteps=100,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
        )
        
        mock_ppo_class.assert_called_once()
        call_args = mock_ppo_class.call_args
        # First positional arg should be the policy class
        policy_class = call_args[0][0]
        self.assertEqual(policy_class.__name__, "AttentionPointerPolicy")

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_sets_hyperparameters(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify model is created with correct hyperparameters."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        
        opponent_gen = MockTeamGenerator(["opp1"])
        
        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            timesteps=100,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
        )
        
        call_kwargs = mock_ppo_class.call_args[1]
        self.assertEqual(call_kwargs['n_steps'], N_STEPS)
        self.assertEqual(call_kwargs['batch_size'], BATCH_SIZE)
        self.assertEqual(call_kwargs['gamma'], GAMMA)
        self.assertEqual(call_kwargs['ent_coef'], ENT_COEF)

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_initializes_wandb(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify wandb is initialized with correct config."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        opponent_gen = MockTeamGenerator(["opp1"])
        
        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            timesteps=2000,
            rounds_per_opponent=100,
            eval_every_timesteps=0,
            seed=42,
        )
        
        mock_wandb.init.assert_called_once()
        init_kwargs = mock_wandb.init.call_args[1]
        self.assertEqual(init_kwargs['project'], "pokemon-rl")
        self.assertIn("gen1ou", init_kwargs['name'])
        self.assertEqual(init_kwargs['config']['timesteps'], 2000)
        self.assertEqual(init_kwargs['config']['battle_format'], "gen1ou")

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_learns_without_eval(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify model.learn is called when eval_every_timesteps=0."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        
        opponent_gen = MockTeamGenerator(["opp1"])
        
        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            timesteps=200,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
        )
        
        model.learn.assert_called_once()
        learn_kwargs = model.learn.call_args[1]
        self.assertEqual(learn_kwargs['total_timesteps'], 200)

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    @patch('training.train.evaluate_model')
    def test_train_model_learns_with_eval_checkpoints(
        self, mock_eval, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify model.learn is called multiple times with eval checkpoints."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        mock_eval.return_value = []
        
        opponent_gen = MockTeamGenerator(["opp1"])
        eval_gen = MockTeamGenerator(["eval1"])
        
        eval_kwargs = {
            "battle_format": "gen1ou",
            "opponent_generator": eval_gen,
            "eval_episodes": 5,
            "max_steps": 100,
        }
        
        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            timesteps=200,
            rounds_per_opponent=10,
            eval_every_timesteps=100,
            eval_kwargs=eval_kwargs,
            seed=42,
        )
        
        # Should call learn twice: 100 timesteps each
        self.assertEqual(model.learn.call_count, 2)

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_saves_model(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify model is saved to specified path."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        
        opponent_gen = MockTeamGenerator(["opp1"])
        
        train_model(
            model_path="test_model_path",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            timesteps=100,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
        )
        
        model.save.assert_called_once_with("test_model_path")

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_finishes_wandb_run(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify wandb run is finished after training."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        opponent_gen = MockTeamGenerator(["opp1"])
        
        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            timesteps=100,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
        )
        
        mock_run.finish.assert_called_once()

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_uses_learning_rate_schedule(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify learning rate schedule is callable and returns numeric values."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        
        opponent_gen = MockTeamGenerator(["opp1"])
        
        train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            timesteps=100,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
        )
        
        call_args, call_kwargs = mock_ppo_class.call_args
        lr_schedule = call_kwargs['learning_rate']
        
        # Should be callable
        self.assertTrue(callable(lr_schedule))
        
        # Test that it returns numeric values for different progress values
        lr_at_0 = lr_schedule(0.0)
        lr_at_0_5 = lr_schedule(0.5)
        lr_at_1 = lr_schedule(1.0)
        
        # All should be numbers and positive
        self.assertIsInstance(lr_at_0, (int, float))
        self.assertIsInstance(lr_at_0_5, (int, float))
        self.assertIsInstance(lr_at_1, (int, float))
        self.assertGreater(lr_at_0, 0)
        self.assertGreater(lr_at_0_5, 0)
        self.assertGreater(lr_at_1, 0)


class TestTrainModelIntegration(unittest.TestCase):
    """Integration tests for train_model."""

    @patch('training.train.WandbCallback')
    @patch('training.train.BattleMetricsCallback')
    @patch('training.train.wandb')
    @patch('training.train.build_env')
    @patch('training.train.MaskablePPO')
    @patch('training.train.set_random_seed')
    def test_train_model_with_both_team_generators(
        self, mock_set_seed, mock_ppo_class, mock_build_env, mock_wandb,
        mock_metrics_cb, mock_wandb_cb
    ):
        """Verify train_model handles both single and battle team generators."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        model = make_mock_model()
        mock_ppo_class.return_value = model
        
        mock_wandb.init.return_value = MagicMock()
        
        opponent_gen = MockTeamGenerator(["opp1"])
        agent_gen = MockTeamGenerator(["agent1"])
        battle_gen = MockTeamGenerator(["battle1", "battle2"])
        
        result = train_model(
            model_path="test_model",
            battle_format="gen1ou",
            opponent_generator=opponent_gen,
            agent_team_generator=agent_gen,
            battle_team_generator=battle_gen,
            timesteps=100,
            rounds_per_opponent=10,
            eval_every_timesteps=0,
            seed=42,
        )
        
        self.assertIsNotNone(result)
        mock_build_env.assert_called_once()

if __name__ == "__main__":
    unittest.main()
