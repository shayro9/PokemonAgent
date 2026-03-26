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
        my_bench_encoded   (5 × MyPokemonState.array_len — benched Pokémon only)
        opp_bench_encoded  (5 × OpponentPokemonState.array_len — benched Pokémon only)
        opp_moves          (MAX_MOVES × MoveState.array_len)
        my_moves           (MAX_MOVES × MoveState.array_len)

    Bench encoding note:
    - my_bench_encoded contains the 5 benched Pokémon (excluding active)
    - Bench order follows action mask correspondence (sorted by species for determinism)
    - Empty slots are zero-padded
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

        self.my_bench: list[Pokemon] = [p for p in battle.team.values()
                        if p.species != active_species]
        self.opp_bench: list[Pokemon] = [p for p in battle.opponent_team.values()
                         if p.species != opp_active_species]

        # Sort bench for deterministic ordering (required for action correspondence)
        self.my_bench.sort(key=lambda p: p.species)
        self.opp_bench.sort(key=lambda p: p.species)

        self.my_available_moves: list[Move] = battle.available_moves
        self.opp_moves: list[Move] = list(battle.opponent_active_pokemon.moves.values())

        #-------- States --------
        self.arena_state    : ArenaStateGen1 = ArenaStateGen1(self.battle)
        # Active Pokémon separately
        self.my_active_state    : MyPokemonStateGen1      = MyPokemonStateGen1(self.my_active)
        self.opp_active_state   : OpponentPokemonStateGen1 = OpponentPokemonStateGen1(self.opp_active)
        # Bench: 5 slots (excluding active)
        self.my_bench_state  : TeamState = TeamState(self.my_bench , MyPokemonStateGen1      , 5)
        self.opp_bench_state : TeamState = TeamState(self.opp_bench, OpponentPokemonStateGen1, 5)
        # For backward compatibility, also create full team states
        self.my_team_state  : TeamState = TeamState(self.my_bench  + [self.my_active] , MyPokemonStateGen1      , self.MAX_TEAM_SIZE)
        self.opp_team_state : TeamState = TeamState(self.opp_bench + [self.opp_active], OpponentPokemonStateGen1, self.MAX_TEAM_SIZE)
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
        arr = np.concatenate([
            self.arena_state.to_array(),
            self.my_active_state.to_array(),
            self.opp_active_state.to_array(),
            self.my_bench_state.to_array(),
            self.opp_bench_state.to_array(),
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
        from env.states.gen1.my_pokemon_state_gen_1 import MyPokemonStateGen1
        from env.states.gen1.opponent_pokemon_state_gen_1 import OpponentPokemonStateGen1
        
        return (
                ArenaStateGen1.array_len()
                + MyPokemonStateGen1.array_len()              # my_active
                + OpponentPokemonStateGen1.array_len()       # opp_active
                + TeamState.compute_array_len(MyPokemonStateGen1, 5)        # my_bench (5 slots)
                + TeamState.compute_array_len(OpponentPokemonStateGen1, 5)  # opp_bench (5 slots)
                + MoveState.array_len() * MAX_MOVES          # opp_moves
                + MoveState.array_len() * MAX_MOVES          # my_moves
        )

    @classmethod
    def battle_context_len(cls) -> int:
        """Length of context (everything except my_moves) for policy slicing."""
        from env.states.gen1.my_pokemon_state_gen_1 import MyPokemonStateGen1
        from env.states.gen1.opponent_pokemon_state_gen_1 import OpponentPokemonStateGen1
        
        return (
                ArenaStateGen1.array_len()
                + MyPokemonStateGen1.array_len()              # my_active
                + OpponentPokemonStateGen1.array_len()       # opp_active
                + TeamState.compute_array_len(MyPokemonStateGen1, 5)        # my_bench (5 slots)
                + TeamState.compute_array_len(OpponentPokemonStateGen1, 5)  # opp_bench (5 slots)
                + MoveState.array_len() * MAX_MOVES          # opp_moves
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
