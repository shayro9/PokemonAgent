import unittest
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from env.action_mask_gen_1 import ActionMaskGen1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_move_mock(move_id: str) -> MagicMock:
    m = MagicMock()
    m.id = move_id
    return m


def _make_battle_mock(
    available_move_ids: list[str],
    all_move_ids: list[str],
    available_switch_ids: list[str] | None = None,
    all_pokemon_ids: list[str] | None = None,
) -> MagicMock:
    """Build a minimal battle mock for ActionMaskGen1.set_mask().

    available_move_ids  — moves the agent can legally use this turn
    all_move_ids        — full move dict on the active Pokémon (revealed moves)
    available_switch_ids — slots the agent can switch to (None = empty)
    all_pokemon_ids     — full team dict (None = empty)
    """
    # Build move objects
    all_moves   = {mid: _make_move_mock(mid) for mid in all_move_ids}
    avail_moves = [all_moves[mid] for mid in available_move_ids if mid in all_moves]

    battle = MagicMock()
    battle.active_pokemon.moves = all_moves
    battle.available_moves = avail_moves

    # Switches
    all_pokemon_ids   = all_pokemon_ids   or []
    available_switch_ids = available_switch_ids or []
    all_pokemon   = {pid: MagicMock() for pid in all_pokemon_ids}
    avail_switches = [all_pokemon[pid] for pid in available_switch_ids if pid in all_pokemon]
    battle.team = all_pokemon
    battle.available_switches = avail_switches

    return battle


# ===========================================================================
# ActionMaskGen1 — class constants
# ===========================================================================

class TestActionMaskGen1Constants(unittest.TestCase):

    def test_action_space_is_26(self):
        self.assertEqual(ActionMaskGen1.ACTION_SPACE, 26)

    def test_move_range(self):
        self.assertEqual(list(ActionMaskGen1.ACTION_MOVE_RANGE), [6, 7, 8, 9])

    def test_switch_range(self):
        self.assertEqual(list(ActionMaskGen1.ACTION_SWITCH_RANGE), [0, 1, 2, 3, 4, 5])


# ===========================================================================
# ActionMaskGen1 — initial state
# ===========================================================================

class TestActionMaskGen1InitialState(unittest.TestCase):

    def setUp(self):
        self.mask_gen = ActionMaskGen1()

    def test_initial_mask_is_all_false(self):
        np.testing.assert_array_equal(
            self.mask_gen.get_mask(),
            np.zeros(ActionMaskGen1.ACTION_SPACE, dtype=bool),
        )

    def test_initial_mask_dtype_is_bool(self):
        self.assertEqual(self.mask_gen.get_mask().dtype, bool)

    def test_initial_mask_length(self):
        self.assertEqual(len(self.mask_gen.get_mask()), 26)

    def test_allow_switches_defaults_to_false(self):
        self.assertFalse(ActionMaskGen1().allow_switches)

    def test_allow_switches_can_be_enabled(self):
        self.assertTrue(ActionMaskGen1(allow_switches=True).allow_switches)


# ===========================================================================
# ActionMaskGen1 — reset
# ===========================================================================

class TestActionMaskGen1Reset(unittest.TestCase):

    def test_reset_clears_all_bits(self):
        mask_gen = ActionMaskGen1()
        battle = _make_battle_mock(
            available_move_ids=["tackle", "ember"],
            all_move_ids=["tackle", "ember", "growl", "tail-whip"],
        )
        mask_gen.set_mask(battle)
        self.assertTrue(np.any(mask_gen.get_mask()))   # sanity: some bits set

        mask_gen.reset()
        np.testing.assert_array_equal(
            mask_gen.get_mask(),
            np.zeros(ActionMaskGen1.ACTION_SPACE, dtype=bool),
        )

    def test_reset_returns_none(self):
        self.assertIsNone(ActionMaskGen1().reset())


# ===========================================================================
# ActionMaskGen1 — set_mask: move slots
# ===========================================================================

class TestActionMaskGen1SetMaskMoves(unittest.TestCase):

    def test_all_4_moves_available(self):
        mask_gen = ActionMaskGen1()
        battle = _make_battle_mock(
            available_move_ids=["a", "b", "c", "d"],
            all_move_ids=["a", "b", "c", "d"],
        )
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        for action in ActionMaskGen1.ACTION_MOVE_RANGE:
            self.assertTrue(mask[action], msg=f"action {action} should be True")

    def test_only_first_move_available(self):
        mask_gen = ActionMaskGen1()
        battle = _make_battle_mock(
            available_move_ids=["a"],
            all_move_ids=["a", "b", "c", "d"],
        )
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertTrue(mask[move_actions[0]])
        for action in move_actions[1:]:
            self.assertFalse(mask[action], msg=f"action {action} should be False")

    def test_partial_moves_available(self):
        mask_gen = ActionMaskGen1()
        # Moves b and d are available (slots 1 and 3 in the full move dict)
        battle = _make_battle_mock(
            available_move_ids=["b", "d"],
            all_move_ids=["a", "b", "c", "d"],
        )
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()
        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)

        self.assertFalse(mask[move_actions[0]])   # "a" not available
        self.assertTrue(mask[move_actions[1]])    # "b" available
        self.assertFalse(mask[move_actions[2]])   # "c" not available
        self.assertTrue(mask[move_actions[3]])    # "d" available

    def test_non_move_actions_remain_false(self):
        mask_gen = ActionMaskGen1()
        battle = _make_battle_mock(
            available_move_ids=["a", "b", "c", "d"],
            all_move_ids=["a", "b", "c", "d"],
        )
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        non_move_actions = [i for i in range(26) if i not in ActionMaskGen1.ACTION_MOVE_RANGE]
        for action in non_move_actions:
            self.assertFalse(mask[action], msg=f"non-move action {action} should be False")

    def test_fewer_than_4_known_moves(self):
        """Active Pokémon only has 2 known moves — slots 2 and 3 must be False."""
        mask_gen = ActionMaskGen1()
        battle = _make_battle_mock(
            available_move_ids=["a", "b"],
            all_move_ids=["a", "b"],
        )
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()
        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)

        self.assertTrue(mask[move_actions[0]])
        self.assertTrue(mask[move_actions[1]])
        self.assertFalse(mask[move_actions[2]])
        self.assertFalse(mask[move_actions[3]])

    def test_set_mask_twice_overwrites_previous(self):
        """Calling set_mask a second time should reflect the new battle state."""
        mask_gen = ActionMaskGen1()

        battle1 = _make_battle_mock(
            available_move_ids=["a", "b", "c", "d"],
            all_move_ids=["a", "b", "c", "d"],
        )
        mask_gen.set_mask(battle1)
        self.assertEqual(np.sum(mask_gen.get_mask()), 4)

        # Second battle: only 1 move available
        battle2 = _make_battle_mock(
            available_move_ids=["x"],
            all_move_ids=["x", "y", "z", "w"],
        )
        mask_gen.set_mask(battle2)
        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertTrue(mask_gen.get_mask()[move_actions[0]])
        self.assertFalse(mask_gen.get_mask()[move_actions[1]])


# ===========================================================================
# ActionMaskGen1 — set_mask: switch slots
# ===========================================================================

class TestActionMaskGen1SetMaskSwitches(unittest.TestCase):

    def test_switches_ignored_when_not_allowed(self):
        mask_gen = ActionMaskGen1(allow_switches=False)
        battle = _make_battle_mock(
            available_move_ids=["a"],
            all_move_ids=["a", "b", "c", "d"],
            available_switch_ids=["mon1", "mon2"],
            all_pokemon_ids=["mon1", "mon2", "mon3"],
        )
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        for action in ActionMaskGen1.ACTION_SWITCH_RANGE:
            self.assertFalse(mask[action], msg=f"switch action {action} should be False")

    def test_switches_set_when_allowed(self):
        mask_gen = ActionMaskGen1(allow_switches=True)
        battle = _make_battle_mock(
            available_move_ids=["a"],
            all_move_ids=["a", "b", "c", "d"],
            available_switch_ids=["mon1", "mon2"],
            all_pokemon_ids=["mon1", "mon2", "mon3"],
        )
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        switch_actions = list(ActionMaskGen1.ACTION_SWITCH_RANGE)
        self.assertTrue(mask[switch_actions[0]])
        self.assertTrue(mask[switch_actions[1]])

    def test_no_available_switches(self):
        mask_gen = ActionMaskGen1(allow_switches=True)
        battle = _make_battle_mock(
            available_move_ids=["a"],
            all_move_ids=["a"],
            available_switch_ids=[],
            all_pokemon_ids=["mon1"],
        )
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        for action in ActionMaskGen1.ACTION_SWITCH_RANGE:
            self.assertFalse(mask[action])

    def test_switches_and_moves_coexist(self):
        mask_gen = ActionMaskGen1(allow_switches=True)
        battle = _make_battle_mock(
            available_move_ids=["a", "b"],
            all_move_ids=["a", "b", "c", "d"],
            available_switch_ids=["mon1"],
            all_pokemon_ids=["mon1", "mon2"],
        )
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        # First switch slot should be True
        self.assertTrue(mask[list(ActionMaskGen1.ACTION_SWITCH_RANGE)[0]])
        # First two move slots should be True
        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertTrue(mask[move_actions[0]])
        self.assertTrue(mask[move_actions[1]])
        self.assertFalse(mask[move_actions[2]])


# ===========================================================================
# ActionMaskGen1 — get_mask integrity
# ===========================================================================

class TestActionMaskGen1GetMask(unittest.TestCase):

    def test_returns_numpy_array(self):
        self.assertIsInstance(ActionMaskGen1().get_mask(), np.ndarray)

    def test_dtype_is_bool(self):
        self.assertEqual(ActionMaskGen1().get_mask().dtype, bool)

    def test_length_is_action_space(self):
        self.assertEqual(len(ActionMaskGen1().get_mask()), ActionMaskGen1.ACTION_SPACE)

    def test_valid_action_count_after_full_move_set(self):
        mask_gen = ActionMaskGen1()
        battle = _make_battle_mock(
            available_move_ids=["a", "b", "c", "d"],
            all_move_ids=["a", "b", "c", "d"],
        )
        mask_gen.set_mask(battle)
        self.assertEqual(int(np.sum(mask_gen.get_mask())), 4)


# ===========================================================================
# ActionMaskGen1 — describe()
# ===========================================================================

class TestActionMaskGen1Describe(unittest.TestCase):

    def test_describe_returns_string(self):
        self.assertIsInstance(ActionMaskGen1().describe(), str)

    def test_describe_contains_action_mask_header(self):
        self.assertIn("ACTION MASK", ActionMaskGen1().describe())

    def test_describe_contains_moves_section(self):
        self.assertIn("Moves", ActionMaskGen1().describe())

    def test_describe_shows_valid_actions(self):
        mask_gen = ActionMaskGen1()
        battle = _make_battle_mock(
            available_move_ids=["tackle"],
            all_move_ids=["tackle", "ember"],
        )
        mask_gen.set_mask(battle)
        desc = mask_gen.describe(battle)
        self.assertIn("tackle", desc)

    def test_describe_no_valid_actions_shows_none(self):
        desc = ActionMaskGen1().describe()
        self.assertIn("none", desc)

    def test_describe_valid_actions_list_present(self):
        mask_gen = ActionMaskGen1()
        battle = _make_battle_mock(
            available_move_ids=["a", "b", "c", "d"],
            all_move_ids=["a", "b", "c", "d"],
        )
        mask_gen.set_mask(battle)
        desc = mask_gen.describe(battle)
        self.assertIn("Valid actions", desc)

    def test_describe_with_allow_switches_shows_switches_section(self):
        mask_gen = ActionMaskGen1(allow_switches=True)
        desc = mask_gen.describe()
        self.assertIn("Switch", desc)

    def test_describe_without_allow_switches_no_switches_section(self):
        mask_gen = ActionMaskGen1(allow_switches=False)
        desc = mask_gen.describe()
        self.assertNotIn("Switch", desc)


# ===========================================================================
# PokemonRLWrapper integration — action_masks() delegates to ActionMaskGen1
# ===========================================================================

class TestPokemonRLWrapperActionMasks(unittest.TestCase):
    """
    Lightweight integration tests for the wrapper's action_masks() path.
    We avoid spinning up a real Showdown server by patching at the boundary.
    """

    def _make_wrapper(self):
        """Return a PokemonRLWrapper with all network I/O patched out."""
        from unittest.mock import patch, MagicMock

        with patch("env.singles_env_wrapper.SinglesEnv.__init__", return_value=None):
            from env.singles_env_wrapper import PokemonRLWrapper
            import gymnasium as gym
            from env.states.gen1.battle_state_gen_1 import BattleStateGen1

            wrapper = PokemonRLWrapper.__new__(PokemonRLWrapper)
            wrapper.opponent_teams = []
            wrapper.battle_team_generator = None
            wrapper.agent_team_generator = None
            wrapper.opponent_team_generator = None
            wrapper._last_team_update_round = None
            wrapper._last_finished_battle = None
            wrapper._trackers = {}
            wrapper._action_space = gym.spaces.Discrete(26)
            wrapper.possible_agents = ["agent1", "agent2"]
            wrapper.action_spaces = {a: wrapper._action_space for a in wrapper.possible_agents}
            wrapper.observation_spaces = {}
            wrapper.rounds_played = 0
            wrapper.rounds_per_opponents = 2000
            wrapper.action_mask = ActionMaskGen1()
            return wrapper

    def test_action_masks_returns_array(self):
        wrapper = self._make_wrapper()
        result = wrapper.action_masks()
        self.assertIsInstance(result, np.ndarray)

    def test_action_masks_length_is_26(self):
        wrapper = self._make_wrapper()
        self.assertEqual(len(wrapper.action_masks()), 26)

    def test_action_masks_initially_all_false(self):
        wrapper = self._make_wrapper()
        np.testing.assert_array_equal(
            wrapper.action_masks(),
            np.zeros(26, dtype=bool),
        )

    def test_action_masks_reflects_set_mask(self):
        wrapper = self._make_wrapper()
        battle = _make_battle_mock(
            available_move_ids=["a", "b", "c", "d"],
            all_move_ids=["a", "b", "c", "d"],
        )
        wrapper.action_mask.set_mask(battle)
        mask = wrapper.action_masks()

        for action in ActionMaskGen1.ACTION_MOVE_RANGE:
            self.assertTrue(mask[action])

    def test_reset_clears_action_mask(self):
        wrapper = self._make_wrapper()
        battle = _make_battle_mock(
            available_move_ids=["a", "b", "c", "d"],
            all_move_ids=["a", "b", "c", "d"],
        )
        wrapper.action_mask.set_mask(battle)
        self.assertTrue(np.any(wrapper.action_masks()))

        # Manually trigger the mask-reset part of reset()
        wrapper.action_mask.reset()
        np.testing.assert_array_equal(
            wrapper.action_masks(),
            np.zeros(26, dtype=bool),
        )


# ===========================================================================
# MockPokemon — shared move-instance tests
#
# These tests construct a realistic mock Pokemon whose .moves dict and
# battle.available_moves share the exact same move objects.  This exercises
# the identity-based `item in available_items` check inside _set_mask_range.
# ===========================================================================

class MockPokemon:
    """Minimal stand-in for a poke-env Pokemon with a real moves dict."""

    def __init__(self, move_ids: list[str]) -> None:
        # Each MagicMock is created once and reused — identity is preserved.
        self.moves: dict[str, MagicMock] = {
            mid: _make_move_mock(mid) for mid in move_ids
        }

    def available(self, *ids: str) -> list[MagicMock]:
        """Return the move objects for the given ids (same instances as .moves)."""
        return [self.moves[mid] for mid in ids]


def _battle_from_pokemon(
    pokemon: MockPokemon,
    available_ids: list[str],
) -> MagicMock:
    """Build a battle mock that wires pokemon.moves and available_moves correctly."""
    battle = MagicMock()
    battle.active_pokemon.moves = pokemon.moves
    battle.available_moves = pokemon.available(*available_ids)
    battle.available_switches = []
    battle.team = {}
    return battle


# ---------------------------------------------------------------------------
# Single unavailable move — each slot in turn
# ---------------------------------------------------------------------------

class TestMockPokemonSingleUnavailableMove(unittest.TestCase):
    """Block exactly one move slot and verify only that slot reads False."""

    def _run(self, blocked_idx: int) -> None:
        all_ids = ["tackle", "ember", "growl", "tail-whip"]
        available_ids = [mid for i, mid in enumerate(all_ids) if i != blocked_idx]

        pokemon = MockPokemon(all_ids)
        battle  = _battle_from_pokemon(pokemon, available_ids)

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        for i, action in enumerate(move_actions):
            if i == blocked_idx:
                self.assertFalse(
                    mask[action],
                    msg=f"slot {i} (action {action}) should be False — move was blocked",
                )
            else:
                self.assertTrue(
                    mask[action],
                    msg=f"slot {i} (action {action}) should be True — move was available",
                )

    def test_slot_0_unavailable(self):  self._run(0)
    def test_slot_1_unavailable(self):  self._run(1)
    def test_slot_2_unavailable(self):  self._run(2)
    def test_slot_3_unavailable(self):  self._run(3)


# ---------------------------------------------------------------------------
# All moves unavailable
# ---------------------------------------------------------------------------

class TestMockPokemonAllMovesUnavailable(unittest.TestCase):

    def test_all_move_slots_false(self):
        pokemon = MockPokemon(["a", "b", "c", "d"])
        battle  = _battle_from_pokemon(pokemon, [])   # nothing available

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        for action in ActionMaskGen1.ACTION_MOVE_RANGE:
            self.assertFalse(mask[action], msg=f"action {action} should be False")

    def test_no_valid_actions_at_all(self):
        pokemon = MockPokemon(["a", "b", "c", "d"])
        battle  = _battle_from_pokemon(pokemon, [])

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)

        self.assertEqual(int(np.sum(mask_gen.get_mask())), 0)


# ---------------------------------------------------------------------------
# Exactly one move available — other slots must be False
# ---------------------------------------------------------------------------

class TestMockPokemonOnlyOneMoveAvailable(unittest.TestCase):

    def _run(self, available_idx: int) -> None:
        all_ids = ["tackle", "ember", "growl", "tail-whip"]
        pokemon = MockPokemon(all_ids)
        battle  = _battle_from_pokemon(pokemon, [all_ids[available_idx]])

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        for i, action in enumerate(move_actions):
            if i == available_idx:
                self.assertTrue(mask[action],  msg=f"slot {i} should be True")
            else:
                self.assertFalse(mask[action], msg=f"slot {i} should be False")

    def test_only_slot_0(self): self._run(0)
    def test_only_slot_1(self): self._run(1)
    def test_only_slot_2(self): self._run(2)
    def test_only_slot_3(self): self._run(3)


# ---------------------------------------------------------------------------
# Slot ordering — available moves must map to the correct action indices
# regardless of dict-insertion order
# ---------------------------------------------------------------------------

class TestMockPokemonSlotOrdering(unittest.TestCase):

    def test_available_moves_map_to_correct_action_indices(self):
        """Slots are determined by position in .moves.values(), not by which
        moves happen to be in available_moves."""
        # moves in dict order: [surf, blizzard, thunder, recover]
        # only middle two are available → slots 1 and 2 → actions 7 and 8
        pokemon = MockPokemon(["surf", "blizzard", "thunder", "recover"])
        battle  = _battle_from_pokemon(pokemon, ["blizzard", "thunder"])

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertFalse(mask[move_actions[0]], "surf (slot 0) unavailable → action 6 False")
        self.assertTrue( mask[move_actions[1]], "blizzard (slot 1) available → action 7 True")
        self.assertTrue( mask[move_actions[2]], "thunder (slot 2) available → action 8 True")
        self.assertFalse(mask[move_actions[3]], "recover (slot 3) unavailable → action 9 False")

    def test_only_last_slot_available(self):
        pokemon = MockPokemon(["a", "b", "c", "d"])
        battle  = _battle_from_pokemon(pokemon, ["d"])

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertFalse(mask[move_actions[0]])
        self.assertFalse(mask[move_actions[1]])
        self.assertFalse(mask[move_actions[2]])
        self.assertTrue( mask[move_actions[3]])

    def test_only_first_slot_available(self):
        pokemon = MockPokemon(["a", "b", "c", "d"])
        battle  = _battle_from_pokemon(pokemon, ["a"])

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertTrue( mask[move_actions[0]])
        self.assertFalse(mask[move_actions[1]])
        self.assertFalse(mask[move_actions[2]])
        self.assertFalse(mask[move_actions[3]])


# ---------------------------------------------------------------------------
# Identity — a different object with the same id must NOT be treated as available
# ---------------------------------------------------------------------------

class TestMockPokemonIdentityNotEquality(unittest.TestCase):
    """_set_mask_range uses `item in available_items`.  A different MagicMock
    object that happens to share the same .id must not be considered available,
    because Python list `in` falls back to __eq__ / identity for MagicMock."""

    def test_different_object_same_id_is_not_available(self):
        pokemon = MockPokemon(["tackle", "ember", "growl", "tail-whip"])

        # Build a *new* mock with the same id — a different object
        impostor = _make_move_mock("tackle")
        self.assertIsNot(impostor, pokemon.moves["tackle"])

        battle = MagicMock()
        battle.active_pokemon.moves = pokemon.moves
        battle.available_moves = [impostor]   # impostor, not the real instance
        battle.available_switches = []
        battle.team = {}

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        # None of the real move slots should be True
        for action in ActionMaskGen1.ACTION_MOVE_RANGE:
            self.assertFalse(
                mask[action],
                msg=f"action {action}: impostor object must not unlock any slot",
            )

    def test_correct_instance_is_available(self):
        """Sanity-check the inverse: the real instance does unlock its slot."""
        pokemon = MockPokemon(["tackle", "ember", "growl", "tail-whip"])

        battle = MagicMock()
        battle.active_pokemon.moves = pokemon.moves
        battle.available_moves = [pokemon.moves["tackle"]]   # real instance
        battle.available_switches = []
        battle.team = {}

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertTrue(mask[move_actions[0]], "real tackle instance should unlock slot 0")
        for action in move_actions[1:]:
            self.assertFalse(mask[action])


# ---------------------------------------------------------------------------
# Fewer than 4 known moves — unavailable slots must stay False
# ---------------------------------------------------------------------------

class TestMockPokemonFewerThan4Moves(unittest.TestCase):

    def test_2_moves_known_both_available(self):
        pokemon = MockPokemon(["surf", "recover"])
        battle  = _battle_from_pokemon(pokemon, ["surf", "recover"])

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertTrue( mask[move_actions[0]])
        self.assertTrue( mask[move_actions[1]])
        self.assertFalse(mask[move_actions[2]])
        self.assertFalse(mask[move_actions[3]])

    def test_2_moves_known_first_unavailable(self):
        pokemon = MockPokemon(["surf", "recover"])
        battle  = _battle_from_pokemon(pokemon, ["recover"])   # surf locked out

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertFalse(mask[move_actions[0]], "surf (slot 0) should be False")
        self.assertTrue( mask[move_actions[1]], "recover (slot 1) should be True")
        self.assertFalse(mask[move_actions[2]])
        self.assertFalse(mask[move_actions[3]])

    def test_1_move_known_and_unavailable(self):
        pokemon = MockPokemon(["splash"])
        battle  = _battle_from_pokemon(pokemon, [])   # splash locked (e.g. taunted)

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        for action in ActionMaskGen1.ACTION_MOVE_RANGE:
            self.assertFalse(mask[action])

    def test_3_moves_middle_unavailable(self):
        pokemon = MockPokemon(["a", "b", "c"])
        battle  = _battle_from_pokemon(pokemon, ["a", "c"])   # b is locked

        mask_gen = ActionMaskGen1()
        mask_gen.set_mask(battle)
        mask = mask_gen.get_mask()

        move_actions = list(ActionMaskGen1.ACTION_MOVE_RANGE)
        self.assertTrue( mask[move_actions[0]], "a available")
        self.assertFalse(mask[move_actions[1]], "b unavailable")
        self.assertTrue( mask[move_actions[2]], "c available")
        self.assertFalse(mask[move_actions[3]], "slot 3 unknown")


if __name__ == "__main__":
    unittest.main()
