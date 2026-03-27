from __future__ import annotations

import numpy as np
from poke_env.battle import AbstractBattle, Move, Pokemon

from env.states.gen1.arena_state_gen1 import ArenaStateGen1
from env.states.gen1.my_pokemon_state_gen_1 import MyPokemonStateGen1
from env.states.gen1.opponent_pokemon_state_gen_1 import OpponentPokemonStateGen1
from env.states.move_state import MoveState
from env.states.team_state import TeamState
from env.states.state_utils import MAX_TEAM_SIZE, MAX_MOVES


class BattleStateGen1:
    """
    Full snapshot of a single battle turn, ready for embedding.

    Layout of ``to_array()``
    ------------------------
        arena_state        (ArenaState.array_len)
        opp_moves          (MAX_MOVES × MoveState.array_len)
        opp_bench_encoded  (5 × OpponentPokemonState.array_len — benched Pokémon only)
        my_bench_encoded   (5 × (MyPokemonState.array_len + 4×MoveState.array_len) — benched Pokémon + moves)
    """

    GEN: int = 1
    MAX_MOVES: int = MAX_MOVES
    MAX_TEAM_SIZE: int = MAX_TEAM_SIZE

    def __init__(self, battle: AbstractBattle) -> None:
        self.battle: AbstractBattle = battle

        # --- Active Pokémon ---
        self.my_active : Pokemon = battle.active_pokemon
        self.opp_active: Pokemon = battle.opponent_active_pokemon

        # --- Bench (excluding active) ---
        active_species = battle.active_pokemon.species
        opp_active_species = battle.opponent_active_pokemon.species

        self.my_bench: list[Pokemon] = list(battle.team.values())
        self.opp_bench: list[Pokemon] = list(battle.opponent_team.values())

        self.my_available_moves: list[Move] = battle.available_moves
        self.opp_moves: list[Move] = list(battle.opponent_active_pokemon.moves.values())

        #-------- States --------
        self.arena_state    : ArenaStateGen1 = ArenaStateGen1(self.battle)
        # Bench: 5 slots (excluding active)
        self.my_bench_state  : TeamState = TeamState(self.my_bench , MyPokemonStateGen1      , self.MAX_TEAM_SIZE)
        self.opp_bench_state : TeamState = TeamState(self.opp_bench, OpponentPokemonStateGen1, self.MAX_TEAM_SIZE)
        # Moves
        self.opp_moves_state: list[MoveState]  = self._encode_moves(self.opp_moves         , self.opp_active, self.my_active)
        self.my_moves_state : list[MoveState]  = self._encode_moves(self.my_available_moves, self.my_active , self.opp_active)

    def _encode_moves(self, available_moves: list[Move], attacking_pokemon: Pokemon, defending_pokemon: Pokemon) -> list[MoveState]:
        """Build up to MAX_MOVES MoveState objects, zero-padded."""
        all_moves = list(attacking_pokemon.moves.values()) + [None] * MAX_MOVES
        moves_list = [m if m in available_moves else None for m in all_moves[:MAX_MOVES]]
        attacking_types = attacking_pokemon.types
        defending_types = defending_pokemon.types

        states = [MoveState(m, defending_types, attacking_types, self.GEN) for m in moves_list]
        return states

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Return the full flat float32 feature vector for this turn."""
        self.my_bench_state.encode_moves(self.opp_active, gen=self.GEN, available_moves=self.my_available_moves)

        arr = np.concatenate([
            self.arena_state.to_array(),
            np.concatenate([m.to_array() for m in self.opp_moves_state]),
            self.opp_bench_state.to_array(),
            self.my_bench_state.to_array(),
        ]).astype(np.float32)

        assert len(arr) == self.array_len(), (
            f"BattleState.to_array(): expected {self.array_len()}, got {len(arr)}"
        )
        return arr

    @classmethod
    def array_len(cls) -> int:
        """Expected flat vector length (static after construction)."""
        # Each my team member (active + 5 bench) includes their 4 moves
        return (
                ArenaStateGen1.array_len()                                          # arena_state
                + MoveState.array_len() * MAX_MOVES                                 # opp_moves
                + TeamState.compute_array_len(OpponentPokemonStateGen1, 6) # opp_bench (5 slots, no moves)
                + TeamState.compute_array_len(MyPokemonStateGen1, 6)       # my_bench (5 slots × pokemon+moves)
        )


    @classmethod
    def battle_before_me_len(cls) -> int:
        """Length of context (everything except my_moves) for policy slicing."""

        return (
                ArenaStateGen1.array_len()                                                  # arena
                + MoveState.array_len() * MAX_MOVES                                         # opp_moves
                + TeamState.compute_array_len(OpponentPokemonStateGen1, MAX_TEAM_SIZE)      # opp_bench (6 slots, no moves)
        )

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def describe(self) -> str:
        move_lines = "\n".join(
            f"  Move {i}: {ms.id} {ms}" for i, ms in enumerate(self.my_moves_state)
        )
        sections = [
            "=== BattleState ===",
            self.arena_state.describe(),
            "--- My Bench ---",
            self.my_bench_state.describe(),
            "--- Opp Bench ---",
            self.opp_bench_state.describe(),
            "--- Moves ---",
            move_lines,
            f"Total array length : {self.array_len()}",
        ]
        return "\n".join(sections)

    def __repr__(self) -> str:
        return self.describe()
