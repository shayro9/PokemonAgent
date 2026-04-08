"""
Tests for env/battle_config.py — BattleConfig dataclass.

Covers:
  1. gen1() factory produces a valid, correct config
  2. All properties return the expected Gen 1 values
  3. Config is frozen (immutable)
  4. Properties are consistent with the state classes they reference
  5. A synthetic custom config works end-to-end (Gen N extensibility)
"""

import pytest
from dataclasses import FrozenInstanceError

from env.battle_config import BattleConfig
from env.states.gen1.battle_state_gen_1 import BattleStateGen1
from env.states.gen1.my_pokemon_state_gen_1 import MyPokemonStateGen1
from env.states.move_state import MoveState
from env.states.state_utils import MAX_TEAM_SIZE, MAX_MOVES


# ─── 1. Factory ──────────────────────────────────────────────────────────────

class TestGen1Factory:

    def test_gen1_returns_battle_config(self):
        cfg = BattleConfig.gen1()
        assert isinstance(cfg, BattleConfig)

    def test_gen1_has_gen_1(self):
        assert BattleConfig.gen1().gen == 1

    def test_gen1_battle_state_cls_is_gen1(self):
        assert BattleConfig.gen1().battle_state_cls is BattleStateGen1

    def test_gen1_my_pokemon_state_cls_is_gen1(self):
        assert BattleConfig.gen1().my_pokemon_state_cls is MyPokemonStateGen1

    def test_gen1_action_space_size_is_10(self):
        assert BattleConfig.gen1().action_space_size == 10

    def test_gen1_factory_returns_equal_instances(self):
        """Two calls to gen1() must return equal (value-equal) configs."""
        assert BattleConfig.gen1() == BattleConfig.gen1()


# ─── 2. Properties match known Gen 1 values ──────────────────────────────────

class TestGen1Properties:

    def setup_method(self):
        self.cfg = BattleConfig.gen1()

    def test_obs_dim_matches_battle_state_cls(self):
        assert self.cfg.obs_dim == BattleStateGen1.array_len()

    def test_arena_opponent_len_matches_battle_state_cls(self):
        assert self.cfg.arena_opponent_len == BattleStateGen1.battle_before_me_len()

    def test_my_pokemon_len_matches_my_pokemon_state_cls(self):
        assert self.cfg.my_pokemon_len == MyPokemonStateGen1.array_len()

    def test_context_len_is_arena_opponent_plus_my_pokemon(self):
        assert self.cfg.context_len == self.cfg.arena_opponent_len + self.cfg.my_pokemon_len

    def test_move_len_matches_move_state(self):
        assert self.cfg.move_len == MoveState.array_len()

    def test_my_moves_start_correct(self):
        expected = MyPokemonStateGen1.array_len() - MoveState.array_len() * MAX_MOVES
        assert self.cfg.my_moves_start == expected

    def test_n_switch_actions_equals_max_team_size(self):
        assert self.cfg.n_switch_actions == MAX_TEAM_SIZE

    def test_obs_dim_is_positive(self):
        assert self.cfg.obs_dim > 0

    def test_context_len_less_than_obs_dim(self):
        """Context is a slice of the full observation."""
        assert self.cfg.context_len < self.cfg.obs_dim

    def test_my_moves_start_less_than_my_pokemon_len(self):
        """Move features start somewhere inside the pokemon block."""
        assert 0 < self.cfg.my_moves_start < self.cfg.my_pokemon_len

    def test_obs_layout_adds_up(self):
        """Verify: arena_opp + 6*(my_pokemon + alive_flag) = obs_dim."""
        expected = (
            self.cfg.arena_opponent_len
            + self.cfg.my_pokemon_len * MAX_TEAM_SIZE
            + MAX_TEAM_SIZE  # alive vector
        )
        assert expected == self.cfg.obs_dim


# ─── 3. Immutability ─────────────────────────────────────────────────────────

class TestBattleConfigImmutability:

    def test_cannot_set_gen(self):
        cfg = BattleConfig.gen1()
        with pytest.raises(FrozenInstanceError):
            cfg.gen = 2

    def test_cannot_set_battle_state_cls(self):
        cfg = BattleConfig.gen1()
        with pytest.raises(FrozenInstanceError):
            cfg.battle_state_cls = object

    def test_cannot_set_action_space_size(self):
        cfg = BattleConfig.gen1()
        with pytest.raises(FrozenInstanceError):
            cfg.action_space_size = 99


# ─── 4. Custom (future-gen) config via direct construction ───────────────────

class TestCustomConfig:
    """Verify that a custom config (simulating Gen 2) can be constructed and used."""

    def test_custom_config_with_gen1_classes(self):
        """A manually constructed config using Gen 1 classes should equal gen1()."""
        custom = BattleConfig(
            gen=1,
            battle_state_cls=BattleStateGen1,
            my_pokemon_state_cls=MyPokemonStateGen1,
            action_space_size=10,
        )
        assert custom == BattleConfig.gen1()

    def test_custom_gen_number(self):
        """Can construct a config with a different gen number."""
        cfg = BattleConfig(
            gen=2,
            battle_state_cls=BattleStateGen1,   # placeholder — real Gen 2 would differ
            my_pokemon_state_cls=MyPokemonStateGen1,
            action_space_size=12,               # hypothetical larger action space
        )
        assert cfg.gen == 2
        assert cfg.action_space_size == 12
        # Properties still work via the referenced classes
        assert cfg.obs_dim == BattleStateGen1.array_len()

    def test_properties_are_derived_from_class_refs(self):
        """Properties compute from the class references, not from hardcoded values."""
        cfg = BattleConfig(
            gen=99,
            battle_state_cls=BattleStateGen1,
            my_pokemon_state_cls=MyPokemonStateGen1,
            action_space_size=10,
        )
        assert cfg.obs_dim == BattleStateGen1.array_len()
        assert cfg.arena_opponent_len == BattleStateGen1.battle_before_me_len()
        assert cfg.my_pokemon_len == MyPokemonStateGen1.array_len()
