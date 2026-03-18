from __future__ import annotations

import numpy as np
from poke_env.battle import Battle

from env.states.arena_state import ArenaState
from env.states.my_pokemon_state import MyPokemonState
from env.states.opponent_pokemon_state import OpponentPokemonState
from env.states.team_state import TeamState

MAX_TEAM_SIZE = 6
MAX_MOVES = 4

class BattleState:
    """
    Full snapshot of a single battle turn, ready for embedding.
    """

    GEN: int = 1
    MAX_MOVES: int = MAX_MOVES
    MAX_TEAM_SIZE: int = MAX_TEAM_SIZE

    def __init__(self, battle: Battle) -> None:
        # --- Arena ---
        self.arena_state = ArenaState(battle)

        my_bench = list(battle.team.values())
        opp_bench = list(battle.opponent_team.values())

        self.my_bench = TeamState(my_bench, MyPokemonState, self.MAX_TEAM_SIZE)
        self.opp_bench = TeamState(opp_bench, OpponentPokemonState, self.MAX_TEAM_SIZE)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Return the full flat float32 feature vector for this turn."""
        arr = np.concatenate([
            self.arena_state.to_array(),  # arena
            self.my_bench.to_array(),  # my bench
            self.opp_bench.to_array(),  # opp bench
        ]).astype(np.float32)

        assert len(arr) == self.array_len(), (
            f"BattleState.to_array(): expected {self.array_len()}, got {len(arr)}"
        )
        return arr

    def array_len(self) -> int:
        """Expected flat vector length (static after construction)."""
        return (
                self.arena_state.array_len()
                + self.my_bench.array_len()
                + self.opp_bench.array_len()
        )

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def describe(self) -> str:
        sections = [
            "=== BattleState ===",
            self.arena_state.describe(),
            "--- My Bench ---",
            self.my_bench.describe(),
            "--- Opp Bench ---",
            self.opp_bench.describe(),
            f"Total array length : {self.array_len()}",
        ]
        return "\n".join(sections)

    def __repr__(self) -> str:
        return self.describe()
