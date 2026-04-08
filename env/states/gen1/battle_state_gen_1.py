from __future__ import annotations

import threading
import numpy as np
from poke_env.battle import AbstractBattle, Move, Pokemon

from env.states.gen1.arena_state_gen1 import ArenaStateGen1
from env.states.gen1.my_pokemon_state_gen_1 import MyPokemonStateGen1
from env.states.gen1.opponent_pokemon_state_gen_1 import OpponentPokemonStateGen1
from env.states.move_state import MoveState
from env.states.team_state import TeamState
from env.states.state_utils import MAX_TEAM_SIZE, MAX_MOVES
from env.states.pokemon_state import _get_cached_move_state


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

    # Lazy-initialised class-level offsets (constant after first call, never mutated).
    _offsets_ready: bool = False
    _o_arena: int = 0
    _o_opp_moves: int = 0
    _o_opp_bench: int = 0
    _o_my_bench: int = 0
    _move_len: int = 0
    _buf_len: int = 0
    # Each thread gets its own scratch buffer — allocated once per thread, never per instance.
    _thread_local: threading.local = threading.local()

    @classmethod
    def _init_buffer(cls) -> None:
        a  = ArenaStateGen1.array_len()
        m  = MoveState.array_len()
        om = m * MAX_MOVES
        ob = TeamState.compute_array_len(OpponentPokemonStateGen1, 6)
        mb = TeamState.compute_array_len(MyPokemonStateGen1, 6)
        cls._o_arena     = 0
        cls._o_opp_moves = a
        cls._o_opp_bench = a + om
        cls._o_my_bench  = a + om + ob
        cls._move_len    = m
        cls._buf_len     = a + om + ob + mb
        cls._offsets_ready = True

    @classmethod
    def _get_thread_buf(cls) -> np.ndarray:
        """Return the calling thread's scratch buffer, allocating it on first use."""
        tl = cls._thread_local
        if not hasattr(tl, "buf"):
            tl.buf = np.zeros(cls._buf_len, dtype=np.float32)
        return tl.buf

    def __init__(self, battle: AbstractBattle) -> None:
        self.battle: AbstractBattle = battle

        # --- Active Pokémon ---
        self.my_active : Pokemon = battle.active_pokemon
        self.opp_active: Pokemon = battle.opponent_active_pokemon

        # --- Bench (including active) ---
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

        if not self.__class__._offsets_ready:
            self.__class__._init_buffer()

    def _encode_moves(self, available_moves: list[Move], attacking_pokemon: Pokemon, defending_pokemon: Pokemon) -> list[MoveState]:
        """Build up to MAX_MOVES MoveState objects, zero-padded."""
        all_moves = list(attacking_pokemon.moves.values()) + [None] * MAX_MOVES
        moves_list = [m if m in available_moves else None for m in all_moves[:MAX_MOVES]]
        attacking_types = tuple(attacking_pokemon.types)
        defending_types = tuple(defending_pokemon.types)

        return [_get_cached_move_state(m, defending_types, attacking_types, self.GEN) for m in moves_list]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Return the full flat float32 feature vector for this turn."""
        self.my_bench_state.encode_moves(self.opp_active, gen=self.GEN, available_moves=self.my_available_moves)

        buf = self.__class__._get_thread_buf()
        m = self._move_len

        buf[self._o_arena    : self._o_opp_moves] = self.arena_state.to_array()
        off = self._o_opp_moves
        for ms in self.opp_moves_state:
            buf[off : off + m] = ms.to_array()
            off += m
        buf[self._o_opp_bench : self._o_my_bench] = self.opp_bench_state.to_array()
        buf[self._o_my_bench  :                  ] = self.my_bench_state.to_array()

        return buf.copy()

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
