import unittest
from unittest.mock import MagicMock
import numpy as np
from env.action_mask_gen_1 import ActionMaskGen1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_move_mock(move_id: str) -> MagicMock:
    m = MagicMock()
    m.id = move_id
    return m


class MockPokemon:
    """Matches the internal logic of poke-env where move objects are persistent."""

    def __init__(self, move_ids: list[str]) -> None:
        self.moves: dict[str, MagicMock] = {
            mid: _make_move_mock(mid) for mid in move_ids
        }

    def available(self, *ids: str) -> list[MagicMock]:
        return [self.moves[mid] for mid in ids]


def _make_battle_mock(pokemon, available_ids, switch_ids=None, all_team_ids=None):
    battle = MagicMock()
    battle.active_pokemon.moves = pokemon.moves
    battle.available_moves = pokemon.available(*available_ids)

    # Setup Team/Switches
    all_team_ids = all_team_ids or []
    switch_ids = switch_ids or []
    team_dict = {pid: MagicMock(species=pid) for pid in all_team_ids}
    battle.team = team_dict
    battle.available_switches = [team_dict[pid] for pid in switch_ids if pid in team_dict]

    return battle


# ===========================================================================
# ActionMaskGen1 — Tests
# ===========================================================================

class TestActionMaskGen1(unittest.TestCase):

    def setUp(self):
        self.mask_gen = ActionMaskGen1(allow_switches=True)

    def test_constants(self):
        """Verify ranges match the Gen 1 implementation."""
        self.assertEqual(list(ActionMaskGen1.ACTION_MOVE_RANGE), [6, 7, 8, 9])
        self.assertEqual(list(ActionMaskGen1.ACTION_SWITCH_RANGE), [0, 1, 2, 3, 4, 5])

    def test_struggle_fallback(self):
        """Verify the move_action=True fallback for Struggle [1, 0, 0, 0]."""
        pkmn = MockPokemon(["tackle"])
        battle = _make_battle_mock(pkmn, [])  # No moves available

        # We need to manually ensure _set_mask_range is called with move_action=True
        # set_mask calls it this way for moves.
        self.mask_gen.set_mask(battle)
        mask = self.mask_gen.get_mask()

        self.assertTrue(mask[6], "Struggle fallback should enable the first move slot (action 6)")
        self.assertFalse(mask[7], "Other move slots should remain False")

    def test_switch_masking(self):
        """Verify that switches map to actions 0-5 correctly."""
        pkmn = MockPokemon(["tackle"])
        # Team has 3 mons, only the 2nd one is available to switch
        battle = _make_battle_mock(pkmn, ["tackle"], switch_ids=["mon2"], all_team_ids=["mon1", "mon2", "mon3"])

        self.mask_gen.set_mask(battle)
        mask = self.mask_gen.get_mask()

        self.assertFalse(mask[0], "mon1 (slot 0) should be False")
        self.assertTrue(mask[1], "mon2 (slot 1) should be True")
        self.assertFalse(mask[2], "mon3 (slot 2) should be False")

    def test_reset_functionality(self):
        pkmn = MockPokemon(["tackle"])
        battle = _make_battle_mock(pkmn, ["tackle"])
        self.mask_gen.set_mask(battle)

        self.assertTrue(np.any(self.mask_gen.get_mask()))
        self.mask_gen.reset()
        self.assertFalse(np.any(self.mask_gen.get_mask()), "Reset should clear the boolean mask")

    def test_describe_output(self):
        """Ensure describe handles the mapping of names to indices correctly."""
        pkmn = MockPokemon(["surf"])
        battle = _make_battle_mock(pkmn, ["surf"])
        self.mask_gen.set_mask(battle)

        description = self.mask_gen.describe(battle)
        self.assertIn("[06] MOVE   0: (surf)", description)
        self.assertIn("Valid actions: [6]", description)


if __name__ == "__main__":
    unittest.main()