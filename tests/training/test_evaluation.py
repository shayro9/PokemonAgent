"""Unit tests for training.evaluation module."""
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

from training.evaluation import (
    EvalResult, _play_episode, build_fixed_eval_pool,
    _generate_eval_pool, evaluate_model, print_eval_summary,
)
from tests.conftest import (
    make_mock_battle, make_mock_model,
    MockTeamGenerator,
)


class TestEvalResult(unittest.TestCase):
    """Test EvalResult dataclass."""

    def test_episodes_property(self):
        """Verify episodes count is sum of wins, losses, draws."""
        result = EvalResult(timestep=1000, wins=5, losses=3, draws=2)
        self.assertEqual(result.episodes, 10)

    def test_episodes_zero_when_no_results(self):
        """Verify episodes is 0 when no wins/losses/draws."""
        result = EvalResult(timestep=0, wins=0, losses=0, draws=0)
        self.assertEqual(result.episodes, 0)

    def test_win_rate_calculation(self):
        """Verify win_rate includes wins and half of draws."""
        # 5 wins + 2 draws (counts as 1) = 6 out of 10 = 0.6
        result = EvalResult(timestep=1000, wins=5, losses=3, draws=2)
        self.assertAlmostEqual(result.win_rate, 0.6)

    def test_win_rate_perfect(self):
        """Verify win_rate is 1.0 for all wins."""
        result = EvalResult(timestep=1000, wins=10, losses=0, draws=0)
        self.assertEqual(result.win_rate, 1.0)

    def test_win_rate_zero(self):
        """Verify win_rate is 0.0 for all losses."""
        result = EvalResult(timestep=1000, wins=0, losses=10, draws=0)
        self.assertEqual(result.win_rate, 0.0)

    def test_win_rate_zero_episodes(self):
        """Verify win_rate returns 0.0 when no episodes."""
        result = EvalResult(timestep=0, wins=0, losses=0, draws=0)
        self.assertEqual(result.win_rate, 0.0)

    def test_win_rate_only_draws(self):
        """Verify win_rate with only draws is 0.5."""
        result = EvalResult(timestep=1000, wins=0, losses=0, draws=10)
        self.assertEqual(result.win_rate, 0.5)


class TestPlayEpisode(unittest.TestCase):
    """Test _play_episode function."""

    @patch('training.evaluation.get_action_masks')
    def test_play_episode_returns_tuple(self, mock_get_masks):
        """Verify _play_episode returns (total_reward, truncated) tuple."""
        env = MagicMock()
        env.reset.return_value = ([1.0, 0.5], {})
        env.step.return_value = ([0.5, 0.2], 1.0, True, False, {})
        
        model = make_mock_model()
        model.predict.return_value = (0, None)
        
        mock_get_masks.return_value = [True] * 26
        
        result = _play_episode(env, model, max_steps=10)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        total_reward, truncated = result
        self.assertIsInstance(total_reward, float)
        self.assertIsInstance(truncated, bool)

    @patch('training.evaluation.get_action_masks')
    def test_play_episode_terminates_on_done(self, mock_get_masks):
        """Verify episode terminates when terminated=True."""
        env = MagicMock()
        env.reset.return_value = ([1.0, 0.5], {})
        # First step: terminated=True
        env.step.return_value = ([0.5, 0.2], 1.5, True, False, {})
        
        model = make_mock_model()
        model.predict.return_value = (0, None)
        mock_get_masks.return_value = [True] * 26
        
        result = _play_episode(env, model, max_steps=10)
        
        total_reward, truncated = result
        self.assertEqual(total_reward, 1.5)
        self.assertFalse(truncated)  # Not truncated, terminated

    @patch('training.evaluation.get_action_masks')
    def test_play_episode_truncates_on_max_steps(self, mock_get_masks):
        """Verify episode truncates when max_steps is reached."""
        env = MagicMock()
        env.reset.return_value = ([1.0, 0.5], {})
        # Never terminates
        env.step.return_value = ([0.5, 0.2], 1.0, False, False, {})
        
        model = make_mock_model()
        model.predict.return_value = (0, None)
        mock_get_masks.return_value = [True] * 26
        
        result = _play_episode(env, model, max_steps=3)
        
        total_reward, truncated = result
        self.assertTrue(truncated)  # Should be truncated

    @patch('training.evaluation.get_action_masks')
    def test_play_episode_accumulates_rewards(self, mock_get_masks):
        """Verify rewards are accumulated across steps."""
        env = MagicMock()
        env.reset.return_value = ([1.0], {})
        # Step 1: reward=1.0, Step 2: reward=2.0, Step 3: terminated
        env.step.side_effect = [
            ([0.5], 1.0, False, False, {}),
            ([0.5], 2.0, False, False, {}),
            ([0.5], 0.5, True, False, {}),
        ]
        
        model = make_mock_model()
        model.predict.return_value = (0, None)
        mock_get_masks.return_value = [True] * 26
        
        result = _play_episode(env, model, max_steps=10)
        
        total_reward, _ = result
        self.assertAlmostEqual(total_reward, 3.5)  # 1.0 + 2.0 + 0.5


class TestGenerateEvalPool(unittest.TestCase):
    """Test _generate_eval_pool function."""

    def test_generate_eval_pool_size(self):
        """Verify generated pool has correct size."""
        gen = MockTeamGenerator(["team1", "team2"])
        
        pool = _generate_eval_pool(pool_size=5, opponent_generator=gen)
        
        self.assertEqual(len(pool), 5)

    def test_generate_eval_pool_cycles_teams(self):
        """Verify pool cycles through generator."""
        gen = MockTeamGenerator(["team_a", "team_b"])
        
        pool = _generate_eval_pool(pool_size=5, opponent_generator=gen)
        
        # Should cycle: team_a, team_b, team_a, team_b, team_a
        self.assertEqual(pool, ["team_a", "team_b", "team_a", "team_b", "team_a"])

    def test_generate_eval_pool_empty_size(self):
        """Verify pool is empty when pool_size=0."""
        gen = MockTeamGenerator(["team1"])
        
        pool = _generate_eval_pool(pool_size=0, opponent_generator=gen)
        
        self.assertEqual(pool, [])

    def test_generate_eval_pool_no_generator_raises(self):
        """Verify ValueError is raised if generator is None."""
        with self.assertRaisesRegex(ValueError, "Cannot generate eval pool"):
            _generate_eval_pool(pool_size=5, opponent_generator=None)


class TestBuildFixedEvalPool(unittest.TestCase):
    """Test build_fixed_eval_pool function."""

    def test_build_fixed_eval_pool_with_generator(self):
        """Verify pool is built from generator."""
        gen = MockTeamGenerator(["team1", "team2"])
        
        pool = build_fixed_eval_pool(opponent_generator=gen, eval_episodes=3)
        
        self.assertEqual(len(pool), 3)

    def test_build_fixed_eval_pool_no_generator_raises(self):
        """Verify ValueError is raised without generator."""
        with self.assertRaisesRegex(ValueError, "Must provide"):
            build_fixed_eval_pool(opponent_generator=None, eval_episodes=5)


class TestEvaluateModel(unittest.TestCase):
    """Test evaluate_model function."""

    @patch('training.evaluation.build_env')
    @patch('training.evaluation._play_episode')
    @patch('training.evaluation._get_last_battle')
    def test_evaluate_model_returns_list_with_one_result(
        self, mock_get_battle, mock_play_episode, mock_build_env
    ):
        """Verify evaluate_model returns list with single EvalResult."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        gen = MockTeamGenerator(["team1"])
        model = make_mock_model()
        
        # Mock episodes: win, loss, draw
        mock_play_episode.side_effect = [
            (1.5, False),   # Win
            (-1.0, False),  # Loss
            (0.0, False),   # Draw
        ]
        
        battle_win = make_mock_battle(won=True, finished=True)
        battle_loss = make_mock_battle(lost=True, finished=True)
        battle_draw = make_mock_battle(won=False, lost=False, finished=True)
        
        mock_get_battle.side_effect = [battle_win, battle_loss, battle_draw]
        
        results = evaluate_model(
            model=model,
            timestep=1000,
            battle_format="gen1ou",
            opponent_generator=gen,
            eval_episodes=3,
            max_steps=100,
        )
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], EvalResult)

    @patch('training.evaluation.build_env')
    @patch('training.evaluation._play_episode')
    @patch('training.evaluation._get_last_battle')
    def test_evaluate_model_counts_wins_losses_draws(
        self, mock_get_battle, mock_play_episode, mock_build_env
    ):
        """Verify model evaluation counts battle outcomes correctly."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        gen = MockTeamGenerator(["team1"])
        model = make_mock_model()
        
        # Mock 5 episodes: 2 wins, 2 losses, 1 draw
        mock_play_episode.side_effect = [(0, False)] * 5
        
        battles = [
            make_mock_battle(won=True, finished=True),
            make_mock_battle(won=True, finished=True),
            make_mock_battle(lost=True, finished=True),
            make_mock_battle(lost=True, finished=True),
            make_mock_battle(won=False, lost=False, finished=True),  # Draw
        ]
        mock_get_battle.side_effect = battles
        
        results = evaluate_model(
            model=model,
            timestep=2000,
            battle_format="gen1ou",
            opponent_generator=gen,
            eval_episodes=5,
            max_steps=100,
        )
        
        result = results[0]
        self.assertEqual(result.wins, 2)
        self.assertEqual(result.losses, 2)
        self.assertEqual(result.draws, 1)
        self.assertEqual(result.timestep, 2000)

    @patch('training.evaluation.build_env')
    @patch('training.evaluation._play_episode')
    @patch('training.evaluation._get_last_battle')
    def test_evaluate_model_handles_unfinished_battles(
        self, mock_get_battle, mock_play_episode, mock_build_env
    ):
        """Verify unfinished battles are not counted."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        gen = MockTeamGenerator(["team1"])
        model = make_mock_model()
        
        mock_play_episode.return_value = (0, False)
        
        # One unfinished battle
        unfinished = make_mock_battle(finished=False)
        mock_get_battle.return_value = unfinished
        
        results = evaluate_model(
            model=model,
            timestep=1000,
            battle_format="gen1ou",
            opponent_generator=gen,
            eval_episodes=1,
            max_steps=100,
        )
        
        result = results[0]
        self.assertEqual(result.episodes, 0)

    @patch('training.evaluation.build_env')
    def test_evaluate_model_resets_battle_team_generator(self, mock_build_env):
        """Verify battle_team_generator is reset if provided."""
        env = MagicMock()
        mock_build_env.return_value = env
        
        gen = MockTeamGenerator(["team1"])
        battle_gen = MockTeamGenerator(["battle1"])
        model = make_mock_model()
        
        with patch('training.evaluation._play_episode', return_value=(0, False)):
            with patch('training.evaluation._get_last_battle', return_value=None):
                evaluate_model(
                    model=model,
                    timestep=1000,
                    battle_format="gen1ou",
                    opponent_generator=gen,
                    eval_episodes=1,
                    max_steps=100,
                    battle_team_generator=battle_gen,
                )
        
        self.assertEqual(battle_gen.reset_count, 1)


class TestPrintEvalSummary(unittest.TestCase):
    """Test print_eval_summary function."""

    def test_print_eval_summary_single_result(self):
        """Verify print_eval_summary prints correctly for single result."""
        result = EvalResult(timestep=1000, wins=8, losses=2, draws=0)
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            print_eval_summary([result])
            output = mock_stdout.getvalue()
        
        self.assertIn("1000", output)
        self.assertIn("8", output)
        self.assertIn("2", output)

    def test_print_eval_summary_multiple_results(self):
        """Verify print_eval_summary prints multiple results."""
        results = [
            EvalResult(timestep=1000, wins=8, losses=2, draws=0),
            EvalResult(timestep=2000, wins=9, losses=1, draws=0),
        ]
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            print_eval_summary(results)
            output = mock_stdout.getvalue()
        
        self.assertIn("1000", output)
        self.assertIn("2000", output)
        self.assertIn("Overall", output)

    def test_print_eval_summary_calculates_overall_stats(self):
        """Verify overall statistics are calculated correctly."""
        results = [
            EvalResult(timestep=1000, wins=5, losses=3, draws=2),
            EvalResult(timestep=2000, wins=4, losses=4, draws=2),
        ]
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            print_eval_summary(results)
            output = mock_stdout.getvalue()
        
        # Total: 9 wins, 7 losses, 4 draws = 20 episodes
        # Win rate: (9 + 2) / 20 = 0.55 = 55%
        self.assertIn("Overall", output)

    def test_print_eval_summary_handles_zero_episodes(self):
        """Verify print_eval_summary handles zero episodes."""
        result = EvalResult(timestep=1000, wins=0, losses=0, draws=0)
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            print_eval_summary([result])
            output = mock_stdout.getvalue()
        
        # Should not crash, should show 0 win rate
        self.assertIn("1000", output)


if __name__ == "__main__":
    unittest.main()
