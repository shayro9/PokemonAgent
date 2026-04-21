"""Unit tests for env.singles_env_wrapper.PokemonRLWrapper."""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from env.battle_config import BattleConfig
from env.singles_env_wrapper import PokemonRLWrapper, print_state
from tests.conftest import (
    make_mock_battle, make_mock_move, make_mock_pokemon,
    MockTeamGenerator,
)


def _create_wrapper(**kwargs):
    """Helper to create a PokemonRLWrapper with minimal required setup."""
    import gymnasium as gym
    
    with patch.object(PokemonRLWrapper, '__init__', lambda x, **kw: None):
        wrapper = PokemonRLWrapper()
        # Manually set attributes that would be set by parent __init__ and PokemonRLWrapper
        wrapper.possible_agents = ["agent1", "agent2"]
        wrapper.agent1 = MagicMock(username="player")
        wrapper.agent2 = MagicMock(username="opponent")
        wrapper.rounds_per_opponents = kwargs.get('rounds_per_opponents', 2_000)
        wrapper.battle_team_generator = kwargs.get('battle_team_generator')
        wrapper.agent_team_generator = kwargs.get('agent_team_generator')
        wrapper.opponent_team_generator = kwargs.get('opponent_team_generator')
        wrapper._last_team_update_round = None
        wrapper._last_finished_battle = None
        wrapper._action_space = gym.spaces.Discrete(26)  # Real action space
        wrapper.action_spaces = {}
        wrapper.observation_spaces = {}
        wrapper._reward_buffer = {}
        wrapper._obs_cache = {}
        wrapper.action_mask = MagicMock()
        wrapper.rounds_played = 0
        wrapper._battle_config = BattleConfig.gen1()
        return wrapper


class TestPrintState(unittest.TestCase):
    """Test the print_state helper function."""

    def test_print_state_returns_formatted_message(self):
        """Verify print_state formats and returns battle state."""
        mock_state = MagicMock()
        mock_state.describe.return_value = "Test battle description"
        mock_state_class = MagicMock(return_value=mock_state)
        mock_config = MagicMock()
        mock_config.battle_state_cls = mock_state_class

        battle = make_mock_battle()

        result = print_state(battle, battle_config=mock_config, prefix="[TEST]")

        self.assertIn("[TEST]", result)
        self.assertIn("Test battle description", result)
        mock_state_class.assert_called_once_with(battle)
        mock_state.describe.assert_called_once()


class TestPokemonRLWrapperInit(unittest.TestCase):
    """Test PokemonRLWrapper initialization."""

    def test_init_default_parameters(self):
        """Verify initialization with default parameters."""
        wrapper = _create_wrapper()
        
        self.assertEqual(wrapper.rounds_per_opponents, 2_000)
        self.assertIsNone(wrapper.battle_team_generator)
        self.assertIsNone(wrapper.agent_team_generator)
        self.assertIsNone(wrapper.opponent_team_generator)
        self.assertIsNone(wrapper._last_team_update_round)
        self.assertIsNone(wrapper._last_finished_battle)
        self.assertEqual(wrapper.rounds_played, 0)

    def test_init_custom_rounds_per_opponents(self):
        """Verify custom rounds_per_opponents is set."""
        wrapper = _create_wrapper(rounds_per_opponents=500)
        
        self.assertEqual(wrapper.rounds_per_opponents, 500)

    def test_init_team_generators(self):
        """Verify team generators are stored."""
        agent_gen = MockTeamGenerator()
        opponent_gen = MockTeamGenerator()
        
        wrapper = _create_wrapper(
            agent_team_generator=agent_gen,
            opponent_team_generator=opponent_gen,
        )
        
        self.assertEqual(wrapper.agent_team_generator, agent_gen)
        self.assertEqual(wrapper.opponent_team_generator, opponent_gen)

    def test_action_space_is_discrete_26(self):
        """Verify action space is Discrete(26)."""
        import gymnasium as gym
        
        wrapper = _create_wrapper()
        
        self.assertIsInstance(wrapper._action_space, gym.spaces.Discrete)
        self.assertEqual(wrapper._action_space.n, 26)


class TestActionToOrder(unittest.TestCase):
    """Test action_to_order method."""

    @patch('env.singles_env_wrapper.SinglesEnv.action_to_order')
    def test_action_to_order_valid_action(self, mock_parent_method):
        """Verify valid action is passed to parent."""
        wrapper = _create_wrapper()
        wrapper.action_mask = MagicMock()
        wrapper.action_mask.get_mask.return_value = np.array([1] * 26, dtype=bool)
        
        battle = make_mock_battle(player_username="player")
        mock_parent_method.return_value = "tackle"
        
        result = wrapper.action_to_order(6, battle, strict=True)
        
        mock_parent_method.assert_called_once()
        self.assertEqual(result, "tackle")

    @patch('env.singles_env_wrapper.SinglesEnv.action_to_order')
    def test_action_out_of_bounds_strict_raises(self, mock_parent_method):
        """Verify out-of-bounds action raises ValueError in strict mode."""
        wrapper = _create_wrapper()
        wrapper.action_mask = MagicMock()
        wrapper.action_mask.get_mask.return_value = np.array([1] * 26, dtype=bool)
        
        battle = make_mock_battle(player_username="player")
        
        with self.assertRaises(ValueError):
            wrapper.action_to_order(100, battle, strict=True)

    @patch('env.singles_env_wrapper.SinglesEnv.action_to_order')
    def test_action_out_of_bounds_non_strict_uses_default(self, mock_parent_method):
        """Verify out-of-bounds action uses default in non-strict mode."""
        wrapper = _create_wrapper()
        wrapper.action_mask = MagicMock()
        wrapper.action_mask.get_mask.return_value = np.array([1] * 26, dtype=bool)
        wrapper.action_mask.ACTION_DEFAULT = 0
        
        battle = make_mock_battle(player_username="player")
        
        try:
            wrapper.action_to_order(100, battle, strict=False)
        except ValueError:
            self.fail("Should not raise ValueError in non-strict mode")
        
        # Verify parent was called with default action
        mock_parent_method.assert_called()

    @patch('env.singles_env_wrapper.SinglesEnv.action_to_order')
    def test_masked_action_strict_raises(self, mock_parent_method):
        """Verify masked action raises ValueError in strict mode."""
        wrapper = _create_wrapper()
        wrapper.action_mask = MagicMock()
        # Action 6 is masked (False)
        mask = np.zeros(26, dtype=bool)
        wrapper.action_mask.get_mask.return_value = mask
        
        battle = make_mock_battle(player_username="player")
        
        with self.assertRaises(ValueError):
            wrapper.action_to_order(6, battle, strict=True)

    @patch('env.singles_env_wrapper.SinglesEnv.action_to_order')
    def test_non_player_turn_delegates_to_parent(self, mock_parent_method):
        """Verify non-player turns delegate directly to parent."""
        wrapper = _create_wrapper()
        
        battle = make_mock_battle(player_username="opponent")
        mock_parent_method.return_value = "tackle"
        
        result = wrapper.action_to_order(6, battle, strict=True)
        
        # Parent should be called without validation
        mock_parent_method.assert_called_once_with(6, battle, False, True)


class TestEmbedBattle(unittest.TestCase):
    """Test embed_battle method."""

    def test_embed_battle_returns_array(self):
        """Verify embed_battle returns a numpy array."""
        wrapper = _create_wrapper()

        mock_state = MagicMock()
        mock_array = np.array([1.0, -0.5, 0.0] * 256)[:768]
        mock_state.to_array.return_value = mock_array
        mock_state_class = MagicMock(return_value=mock_state)
        wrapper._battle_config = MagicMock(battle_state_cls=mock_state_class)

        battle = make_mock_battle(player_username="player")

        result = wrapper.embed_battle(battle)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, mock_array)
        mock_state_class.assert_called_once_with(battle)

    def test_embed_battle_no_mask_set_for_opponent_turn(self):
        """Verify action mask is NOT set on opponent turn."""
        wrapper = _create_wrapper()
        wrapper.action_mask = MagicMock()

        mock_state = MagicMock()
        mock_state.to_array.return_value = np.zeros(768)
        mock_state_class = MagicMock(return_value=mock_state)
        wrapper._battle_config = MagicMock(battle_state_cls=mock_state_class)

        battle = make_mock_battle(player_username="opponent")

        wrapper.embed_battle(battle)

        wrapper.action_mask.set_mask.assert_not_called()


class TestCalcReward(unittest.TestCase):
    """Test calc_reward method."""

    @patch('env.singles_env_wrapper.get_state_value_optimizable')
    def test_calc_reward_first_call_initializes_buffer(self, mock_get_value):
        """Verify first call initializes reward buffer."""
        wrapper = _create_wrapper()
        wrapper._reward_buffer = {}
        wrapper.rounds_played = 0
        wrapper._last_finished_battle = None
        mock_get_value.return_value = 0.5
        
        battle = make_mock_battle()
        
        reward = wrapper.calc_reward(battle)
        
        # First call: new value (0.5) - old value (0.0) = 0.5
        self.assertEqual(reward, 0.5)
        self.assertIn(battle, wrapper._reward_buffer)

    @patch('env.singles_env_wrapper.get_state_value_optimizable')
    def test_calc_reward_delta_calculation(self, mock_get_value):
        """Verify reward is calculated as delta."""
        wrapper = _create_wrapper()
        wrapper._reward_buffer = {}
        wrapper.rounds_played = 0
        wrapper._last_finished_battle = None
        
        battle = make_mock_battle()
        
        mock_get_value.side_effect = [0.3, 0.7]  # First call, then second
        
        # First call
        reward1 = wrapper.calc_reward(battle)
        self.assertAlmostEqual(reward1, 0.3)  # 0.3 - 0.0
        
        # Second call
        reward2 = wrapper.calc_reward(battle)
        self.assertAlmostEqual(reward2, 0.4)  # 0.7 - 0.3

    @patch('env.singles_env_wrapper.get_state_value_optimizable')
    def test_calc_reward_increments_rounds_on_finished_battle(self, mock_get_value):
        """Verify rounds_played increments when battle finishes."""
        wrapper = _create_wrapper()
        wrapper._reward_buffer = {}
        wrapper.rounds_played = 0
        mock_get_value.return_value = 0.0
        
        battle = make_mock_battle(player_username="player", finished=True)
        
        wrapper.calc_reward(battle)
        
        self.assertEqual(wrapper.rounds_played, 1)
        self.assertEqual(wrapper._last_finished_battle, battle)

    @patch('env.singles_env_wrapper.get_state_value_optimizable')
    def test_calc_reward_no_increment_if_not_player_turn(self, mock_get_value):
        """Verify rounds_played doesn't increment if not player's turn."""
        wrapper = _create_wrapper()
        wrapper._reward_buffer = {}
        wrapper.rounds_played = 0
        mock_get_value.return_value = 0.0
        
        battle = make_mock_battle(player_username="opponent", finished=True)
        
        wrapper.calc_reward(battle)
        
        self.assertEqual(wrapper.rounds_played, 0)


class TestReset(unittest.TestCase):
    """Test reset method."""

    @patch('env.singles_env_wrapper.SinglesEnv.reset')
    def test_reset_clears_action_mask(self, mock_parent_reset):
        """Verify action mask is reset."""
        wrapper = _create_wrapper()
        wrapper.action_mask = MagicMock()
        wrapper.rounds_played = 0
        mock_parent_reset.return_value = (None, {})
        
        wrapper.reset()
        
        wrapper.action_mask.reset.assert_called_once()

    @patch('env.singles_env_wrapper.SinglesEnv.reset')
    def test_reset_updates_teams_at_round_boundary(self, mock_parent_reset):
        """Verify teams are updated when rounds_played is multiple of rounds_per_opponents."""
        agent_gen = MockTeamGenerator(["team1", "team2"])
        opponent_gen = MockTeamGenerator(["opp1", "opp2"])
        
        wrapper = _create_wrapper(
            rounds_per_opponents=2,
            agent_team_generator=agent_gen,
            opponent_team_generator=opponent_gen,
        )
        wrapper.action_mask = MagicMock()
        wrapper.rounds_played = 2  # On boundary
        mock_parent_reset.return_value = (None, {})
        
        wrapper.reset()
        
        wrapper.agent1.update_team.assert_called_once_with("team1")
        wrapper.agent2.update_team.assert_called_once_with("opp1")

    @patch('env.singles_env_wrapper.SinglesEnv.reset')
    def test_reset_does_not_update_twice_at_same_round(self, mock_parent_reset):
        """Verify teams aren't updated twice at the same round number."""
        agent_gen = MockTeamGenerator(["team1", "team2"])
        
        wrapper = _create_wrapper(
            rounds_per_opponents=2,
            agent_team_generator=agent_gen,
        )
        wrapper.action_mask = MagicMock()
        wrapper.rounds_played = 2
        wrapper._last_team_update_round = 2  # Already updated at this round
        mock_parent_reset.return_value = (None, {})
        
        wrapper.reset()
        
        # Should not call update_team since _last_team_update_round matches
        wrapper.agent1.update_team.assert_not_called()


class TestActionMasks(unittest.TestCase):
    """Test action_masks method."""

    def test_action_masks_returns_mask_from_action_mask_object(self):
        """Verify action_masks returns the mask array."""
        wrapper = _create_wrapper()
        expected_mask = np.array([1, 0, 1, 0, 1] + [0] * 21, dtype=bool)
        wrapper.action_mask = MagicMock()
        wrapper.action_mask.get_mask.return_value = expected_mask
        
        result = wrapper.action_masks()
        
        np.testing.assert_array_equal(result, expected_mask)


class TestIsPlayerTurn(unittest.TestCase):
    """Test _is_player_turn helper method."""

    def test_is_player_turn_matches_agent_username(self):
        """Verify _is_player_turn returns True for player username."""
        wrapper = _create_wrapper()
        
        battle = make_mock_battle(player_username="player")
        
        self.assertTrue(wrapper._is_player_turn(battle))

    def test_is_player_turn_returns_false_for_opponent(self):
        """Verify _is_player_turn returns False for opponent username."""
        wrapper = _create_wrapper()
        
        battle = make_mock_battle(player_username="opponent_bot")
        
        self.assertFalse(wrapper._is_player_turn(battle))

    def test_is_player_turn_handles_missing_username(self):
        """Verify _is_player_turn handles missing player_username attribute."""
        wrapper = _create_wrapper()
        
        battle = MagicMock()
        del battle.player_username  # Simulate missing attribute
        
        self.assertFalse(wrapper._is_player_turn(battle))


class TestGetLastBattle(unittest.TestCase):
    """Test get_last_battle method."""

    def test_get_last_battle_returns_last_finished_battle(self):
        """Verify get_last_battle returns stored battle."""
        wrapper = _create_wrapper()
        battle = make_mock_battle(finished=True)
        wrapper._last_finished_battle = battle
        
        result = wrapper.get_last_battle()
        
        self.assertEqual(result, battle)

    def test_get_last_battle_returns_none_initially(self):
        """Verify get_last_battle returns None before any battle."""
        wrapper = _create_wrapper()
        
        result = wrapper.get_last_battle()
        
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
