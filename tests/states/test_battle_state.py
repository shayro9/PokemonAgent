"""
Unit tests for BattleState (battle_state.py).

Runs with: python test_battle_state.py

All dependencies are stubbed (no poke-env required).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Fake components (stubs)
# ---------------------------------------------------------------------------

class FakeArenaState:
    def __init__(self, battle):
        self.data = np.array([1.0, 2.0], dtype=np.float32)

    def to_array(self) -> np.ndarray:
        return self.data

    def array_len(self) -> int:
        return len(self.data)

    def describe(self) -> str:
        return "ArenaState"


class FakeTeamState:
    def __init__(self, members, state_cls, max_size):
        self.max_size = max_size
        self.data = np.ones(max_size * 3, dtype=np.float32)  # fixed size block

    def to_array(self) -> np.ndarray:
        return self.data

    def array_len(self) -> int:
        return len(self.data)

    def describe(self) -> str:
        return "TeamState"


class FakeBattle:
    def __init__(self, team_size=3, opp_size=2):
        self.team = {f"a{i}": object() for i in range(team_size)}
        self.opponent_team = {f"b{i}": object() for i in range(opp_size)}


# ---------------------------------------------------------------------------
# Monkey patching helper
# ---------------------------------------------------------------------------

def make_battle_state():
    from env.states import battle_state as bs

    # Patch dependencies
    bs.ArenaState = FakeArenaState
    bs.TeamState = FakeTeamState

    return bs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBattleStateConstruction:
    def test_construction_does_not_crash(self):
        bs = make_battle_state()
        battle = FakeBattle()

        state = bs.BattleState(battle)
        assert state is not None

    def test_team_sizes_passed_correctly(self):
        bs = make_battle_state()
        battle = FakeBattle(team_size=4, opp_size=5)

        state = bs.BattleState(battle)

        assert state.my_bench.max_size == bs.MAX_TEAM_SIZE
        assert state.opp_bench.max_size == bs.MAX_TEAM_SIZE


# ---------------------------------------------------------------------------

class TestArrayOutput:
    def test_to_array_returns_numpy_array(self):
        bs = make_battle_state()
        state = bs.BattleState(FakeBattle())

        arr = state.to_array()
        assert isinstance(arr, np.ndarray)

    def test_to_array_dtype(self):
        bs = make_battle_state()
        state = bs.BattleState(FakeBattle())

        arr = state.to_array()
        assert arr.dtype == np.float32

    def test_to_array_length_matches_array_len(self):
        bs = make_battle_state()
        state = bs.BattleState(FakeBattle())

        arr = state.to_array()
        assert len(arr) == state.array_len()

    def test_array_is_concatenation(self):
        bs = make_battle_state()
        state = bs.BattleState(FakeBattle())

        arr = state.to_array()

        expected = np.concatenate([
            state.arena_state.to_array(),
            state.my_bench.to_array(),
            state.opp_bench.to_array(),
        ]).astype(np.float32)

        np.testing.assert_array_equal(arr, expected)


# ---------------------------------------------------------------------------

class TestArrayLen:
    def test_array_len_matches_parts(self):
        bs = make_battle_state()
        state = bs.BattleState(FakeBattle())

        expected = (
            state.arena_state.array_len()
            + state.my_bench.array_len()
            + state.opp_bench.array_len()
        )

        assert state.array_len() == expected


# ---------------------------------------------------------------------------

class TestDescribe:
    def test_describe_returns_string(self):
        bs = make_battle_state()
        state = bs.BattleState(FakeBattle())

        desc = state.describe()
        assert isinstance(desc, str)

    def test_describe_contains_sections(self):
        bs = make_battle_state()
        state = bs.BattleState(FakeBattle())

        desc = state.describe()

        assert "BattleState" in desc
        assert "My Bench" in desc
        assert "Opp Bench" in desc

    def test_repr_returns_string(self):
        bs = make_battle_state()
        state = bs.BattleState(FakeBattle())

        assert isinstance(repr(state), str)
