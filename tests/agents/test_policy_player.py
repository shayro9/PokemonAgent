import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from agents.policy_player import PolicyPlayer


class TestPolicyPlayer(unittest.TestCase):
    @patch("agents.policy_player.Player.__init__", return_value=None)
    @patch("agents.policy_player.MaskablePPO.load")
    def test_choose_move_uses_static_helpers_without_wrapper(self, mock_load, _mock_player_init):
        battle = MagicMock()
        battle._wait = False
        battle.valid_orders = []
        battle_config = MagicMock()
        state = MagicMock()
        state.to_array.return_value = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        battle_config.battle_state_cls.return_value = state

        model = MagicMock()
        model.predict.return_value = (np.int64(6), None)
        mock_load.return_value = model

        player = PolicyPlayer(
            model_path="models\\6v6_gen1_1M_meta.zip",
            battle_config=battle_config,
            verbose=False,
        )

        with (
            patch("agents.policy_player.SinglesEnv.get_action_mask", return_value=[0, 0, 0, 0, 0, 0, 1]),
            patch("agents.policy_player.SinglesEnv.action_to_order", return_value="move-order") as mock_action_to_order,
        ):
            result = player.choose_move(battle)

        self.assertEqual(result, "move-order")
        observed = model.predict.call_args.args[0]
        np.testing.assert_array_equal(observed["observation"], state.to_array.return_value)
        np.testing.assert_array_equal(observed["action_mask"], np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool))
        mock_action_to_order.assert_called_once_with(np.int64(6), battle, strict=False)

    @patch("agents.policy_player.Player.__init__", return_value=None)
    @patch("agents.policy_player.MaskablePPO.load")
    def test_choose_move_uses_wrapper_when_provided(self, mock_load, _mock_player_init):
        battle = MagicMock()
        battle._wait = False
        battle.valid_orders = []
        battle_config = MagicMock()
        state = MagicMock()
        state.to_array.return_value = np.array([1.0, 2.0], dtype=np.float32)
        battle_config.battle_state_cls.return_value = state

        model = MagicMock()
        model.predict.return_value = (np.int64(7), None)
        mock_load.return_value = model

        wrapper = MagicMock()
        wrapper.get_action_mask.return_value = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
        wrapper.action_to_order.return_value = "wrapper-order"

        player = PolicyPlayer(
            model_path="models\\6v6_gen1_1M_meta.zip",
            wrapper=wrapper,
            battle_config=battle_config,
            verbose=False,
        )

        with patch("agents.policy_player.SinglesEnv.get_action_mask") as mock_get_action_mask:
            result = player.choose_move(battle)

        self.assertEqual(result, "wrapper-order")
        wrapper.get_action_mask.assert_called_once_with(battle)
        wrapper.action_to_order.assert_called_once_with(np.int64(7), battle, strict=False)
        mock_get_action_mask.assert_not_called()

    @patch("agents.policy_player.Player.__init__", return_value=None)
    @patch("agents.policy_player.MaskablePPO.load")
    def test_choose_move_returns_default_on_wait_turn(self, mock_load, _mock_player_init):
        battle = MagicMock()
        battle._wait = True
        battle.valid_orders = [MagicMock()]
        battle_config = MagicMock()
        state = MagicMock()
        state.to_array.return_value = np.array([1.0], dtype=np.float32)
        battle_config.battle_state_cls.return_value = state

        model = MagicMock()
        mock_load.return_value = model

        player = PolicyPlayer(
            model_path="models\\6v6_gen1_1M_meta.zip",
            battle_config=battle_config,
            verbose=False,
        )

        with patch("agents.policy_player.SinglesEnv.action_to_order", return_value="default-order") as mock_action_to_order:
            result = player.choose_move(battle)

        self.assertEqual(result, "default-order")
        model.predict.assert_not_called()
        mock_action_to_order.assert_called_once_with(np.int64(-2), battle, strict=False)

    @patch("agents.policy_player.Player.__init__", return_value=None)
    @patch("agents.policy_player.MaskablePPO.load")
    def test_choose_move_returns_default_when_only_default_order_is_valid(
        self,
        mock_load,
        _mock_player_init,
    ):
        battle = MagicMock()
        battle._wait = False
        default_order = MagicMock()
        default_order.__str__.return_value = "/choose default"
        battle.valid_orders = [default_order]
        battle_config = MagicMock()
        state = MagicMock()
        state.to_array.return_value = np.array([1.0], dtype=np.float32)
        battle_config.battle_state_cls.return_value = state

        model = MagicMock()
        mock_load.return_value = model

        player = PolicyPlayer(
            model_path="models\\6v6_gen1_1M_meta.zip",
            battle_config=battle_config,
            verbose=False,
        )

        with patch("agents.policy_player.SinglesEnv.action_to_order", return_value="default-order") as mock_action_to_order:
            result = player.choose_move(battle)

        self.assertEqual(result, "default-order")
        model.predict.assert_not_called()
        mock_action_to_order.assert_called_once_with(np.int64(-2), battle, strict=False)


if __name__ == "__main__":
    unittest.main()
