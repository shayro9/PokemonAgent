import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

from embedding import calc_types_vector, embed_move
from env_wrapper import MAX_MOVES, MOVE_EMBED_LEN, PokemonRLWrapper
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status


class StateEmbeddingTests(unittest.TestCase):
    @staticmethod
    def _make_battle(*, available_moves):
        my = SimpleNamespace(
            current_hp_fraction=0.75,
            stats={"atk": 200, "def": 150, "spa": 180, "spd": 170, "spe": 120},
            types=[PokemonType.FIRE],
            weight=90,
        )
        opp = SimpleNamespace(
            current_hp_fraction=0.35,
            base_stats={"hp": 100, "atk": 120, "def": 110, "spa": 130, "spd": 90, "spe": 95},
            boosts={
                "atk": 1,
                "def": 0,
                "spa": -1,
                "spd": 0,
                "spe": 2,
                "accuracy": 0,
                "evasion": 0,
            },
            preparing=False,
            types=[PokemonType.GRASS],
            weight=60,
        )
        return SimpleNamespace(
            active_pokemon=my,
            opponent_active_pokemon=opp,
            available_moves=available_moves,
            gen=9,
        )

    @staticmethod
    def _expected_state_length(battle):
        return (
            1 + len(battle.active_pokemon.stats)
            + 1 + len(battle.opponent_active_pokemon.base_stats)
            + len(battle.opponent_active_pokemon.boosts)
            + 1
            + MAX_MOVES * MOVE_EMBED_LEN
            + len(calc_types_vector(
                battle.active_pokemon.types,
                battle.opponent_active_pokemon.types,
                battle.gen,
            ))
            + 6
        )

    def test_embed_move_has_expected_length(self):
        move = SimpleNamespace(
            base_power=120,
            accuracy=0.8,
            max_pp=8,
            priority=1,
            category=MoveCategory.SPECIAL,
            type=PokemonType.FIRE,
            status=Status.BRN,
            boosts={"atk": -1, "spa": -2},
            self_boost={"spa": 1},
            recoil=0,
            drain=0,
            n_hit=(2, 5),
        )

        vec = embed_move(move, [PokemonType.GRASS, PokemonType.STEEL], gen=9)

        self.assertEqual(len(vec), MOVE_EMBED_LEN)
        self.assertTrue(np.all(np.isfinite(vec)))
        self.assertTrue(np.all(vec <= 1.0))
        self.assertTrue(np.all(vec >= -1.0))

    def test_embed_move_edge_cases_with_missing_fields(self):
        move = SimpleNamespace(
            base_power=None,
            accuracy=True,
            max_pp=None,
            priority=None,
            category=MoveCategory.STATUS,
            type=PokemonType.NORMAL,
            status=None,
            boosts=None,
            self_boost=None,
            recoil=None,
            drain=None,
            n_hit=None,
        )

        vec = embed_move(move, [PokemonType.GHOST], gen=9)

        self.assertEqual(len(vec), MOVE_EMBED_LEN)
        self.assertTrue(np.all(np.isfinite(vec)))
        self.assertTrue(np.all(vec <= 1.0))
        self.assertTrue(np.all(vec >= -1.0))

    def test_calc_types_vector_encodes_immunity(self):
        vec = calc_types_vector(
            [PokemonType.GROUND],
            [PokemonType.FLYING],
            gen=9,
        )

        self.assertEqual(vec.shape, (4,))
        self.assertEqual(float(vec[0]), -1.0)

    def test_embed_battle_uses_dynamic_length_not_hardcoded(self):
        battle = self._make_battle(available_moves=[])
        wrapper = object.__new__(PokemonRLWrapper)

        state = wrapper.embed_battle(battle)

        self.assertEqual(len(state), self._expected_state_length(battle))
        self.assertTrue(np.all(np.isfinite(state)))

    def test_embed_battle_pads_unavailable_moves_with_zeros(self):
        only_move = SimpleNamespace(
            base_power=90,
            accuracy=1.0,
            max_pp=15,
            priority=0,
            category=MoveCategory.PHYSICAL,
            type=PokemonType.FIRE,
            status=None,
            boosts=None,
            self_boost=None,
            recoil=0,
            drain=0,
            n_hit=1,
        )
        battle = self._make_battle(available_moves=[only_move])
        wrapper = object.__new__(PokemonRLWrapper)

        state = wrapper.embed_battle(battle)

        move_start = (
            1 + len(battle.active_pokemon.stats)
            + 1 + len(battle.opponent_active_pokemon.base_stats)
            + len(battle.opponent_active_pokemon.boosts)
            + 1
        )
        first_move = state[move_start: move_start + MOVE_EMBED_LEN]
        remaining_moves = state[move_start + MOVE_EMBED_LEN: move_start + MAX_MOVES * MOVE_EMBED_LEN]

        self.assertFalse(np.allclose(first_move, 0.0))
        self.assertTrue(np.allclose(remaining_moves, 0.0))

    def test_print_state_includes_dynamic_total_dimensions(self):
        battle = self._make_battle(available_moves=[])
        wrapper = object.__new__(PokemonRLWrapper)
        expected_total = self._expected_state_length(battle)

        with io.StringIO() as buf, redirect_stdout(buf):
            message = wrapper.print_state(battle, prefix="[test]")
            printed = buf.getvalue()

        self.assertIn(f"TOTAL DIMENSIONS: {expected_total}", message)
        self.assertIn(f"TOTAL DIMENSIONS: {expected_total}", printed)


if __name__ == "__main__":
    unittest.main()
