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
        my_active          (MyPokemonState.array_len)
        opp_active         (OpponentPokemonState.array_len)
        my_bench           (Team.array_len  — active excluded)
        opp_bench          (Team.array_len  — active excluded)
        move_0 … move_3    (MAX_MOVES × MoveState.array_len)

    Switches are currently ignored.
    """

    GEN: int = 1
    MAX_MOVES: int = MAX_MOVES
    MAX_TEAM_SIZE: int = MAX_TEAM_SIZE

    def __init__(self, battle: AbstractBattle) -> None:
        self.battle: AbstractBattle = battle

        # --- Active Pokémon ---
        self.my_active : Pokemon = battle.active_pokemon
        self.opp_active: Pokemon = battle.opponent_active_pokemon

        # --- Bench ---
        active_species = battle.active_pokemon.species
        opp_active_species = battle.opponent_active_pokemon.species

        self.my_bench: list[Pokemon] = [p for p in battle.team.values()
                        if p.species != active_species]
        self.opp_bench: list[Pokemon] = [p for p in battle.opponent_team.values()
                         if p.species != opp_active_species]

        self.my_available_moves: list[Move] = battle.available_moves
        self.opp_moves: list[Move] = list(battle.opponent_active_pokemon.moves.values())

        #-------- States --------
        self.arena_state    : ArenaStateGen1 = ArenaStateGen1(self.battle)
        self.my_team_state  : TeamState = TeamState(self.my_bench  + [self.my_active] , MyPokemonStateGen1      , self.MAX_TEAM_SIZE)
        self.opp_team_state : TeamState = TeamState(self.opp_bench + [self.opp_active], OpponentPokemonStateGen1, self.MAX_TEAM_SIZE)
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
        arr = np.concatenate([
            self.arena_state.to_array(),
            self.my_team_state.to_array(),
            self.opp_team_state.to_array(),
            np.concatenate([m.to_array() for m in self.opp_moves_state]),
            np.concatenate([m.to_array() for m in self.my_moves_state]),
        ]).astype(np.float32)

        assert len(arr) == self.array_len(), (
            f"BattleState.to_array(): expected {self.array_len()}, got {len(arr)}"
        )
        return arr

    @classmethod
    def array_len(cls) -> int:
        """Expected flat vector length (static after construction)."""
        return (
                ArenaStateGen1.array_len()
                + TeamState.compute_array_len(MyPokemonStateGen1, cls.MAX_TEAM_SIZE)
                + TeamState.compute_array_len(OpponentPokemonStateGen1, cls.MAX_TEAM_SIZE)
                + MoveState.array_len() * MAX_MOVES
                + MoveState.array_len() * MAX_MOVES
        )

    @classmethod
    def battle_context_len(cls) -> int:
        """Expected flat vector length (static after construction)."""
        return (
                ArenaStateGen1.array_len()
                + TeamState.compute_array_len(MyPokemonStateGen1, cls.MAX_TEAM_SIZE)
                + TeamState.compute_array_len(OpponentPokemonStateGen1, cls.MAX_TEAM_SIZE)
                + MoveState.array_len() * MAX_MOVES
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
            self.my_team_state.describe(),
            "--- Opp Bench ---",
            self.opp_team_state.describe(),
            "--- Moves ---",
            move_lines,
            f"Total array length : {self.array_len()}",
        ]
        return "\n".join(sections)

    def __repr__(self) -> str:
        return self.describe()
