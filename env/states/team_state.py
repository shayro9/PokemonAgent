from __future__ import annotations

import numpy as np
from poke_env.battle.pokemon import Pokemon

from env.states.pokemon_state import PokemonState
from env.states.state_utils import MAX_TEAM_SIZE

class TeamState:
    """
    Holds up to MAX_TEAM_SIZE Pokémon state objects for one side of the field.

    Slots are filled in the order the team dict is iterated (active Pokémon
    excluded — it is encoded separately in BattleState).  Empty slots are
    padded with zero-initialised state objects so the output array length is
    always fixed.

    Usage
    -----
        my_team  = Team(bench_pokemons, MyPokemonState)
        opp_team = Team(bench_pokemons, OpponentPokemonState)
        vec      = my_team.to_array()          # shape (MAX_TEAM_SIZE * slot_len,)
    """

    def __init__(
            self,
            pokemons: list[Pokemon],
            state_cls: type[PokemonState],
            max_size: int = MAX_TEAM_SIZE,
    ) -> None:
        """
        :param pokemons:  Raw poke-env ``Pokemon`` objects (active excluded).
        :param state_cls: ``MyPokemonState`` or ``OpponentPokemonState``.
        :param max_size:  Maximum number of slots (default 6).
        """
        self.max_size = max_size
        pokemons.sort(key=lambda p: p.species)
        self.members: list[PokemonState] = ([state_cls(p) for p in pokemons[:max_size]]
                                          + [state_cls(None) for _ in range(max_size - len(pokemons[:max_size]))])
        self.alive_vector = self.encode_active_and_faint()
        self.active = self.get_active()

        # slot length is fixed by the state class
        self._slot_len: int = self.members[0].array_len()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Concatenated flat array for all slots.

        :returns: float32 array of shape ``(max_size * slot_len,)``.
        """
        return np.concatenate([
            np.concatenate([m.to_array() for m in self.members]),
            self.alive_vector, # vector the size of max_size with 1 if active, -1 if fainted and 0 else
        ]).astype(np.float32)

    def array_len(self) -> int:
        """Expected total flat vector length."""
        return self._slot_len * self.max_size + self.max_size

    @classmethod
    def compute_array_len(cls, state_cls: type[PokemonState], max_size: int = MAX_TEAM_SIZE) -> int:
        return state_cls.array_len() * max_size + max_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def alive_count(self) -> int:
        """Number of non-fainted, non-placeholder members."""
        return sum(1 for m in self.members if m.species != "none" and not m.fainted)

    def encode_active_and_faint(self) -> np.ndarray:
        def status(m) -> float:
            return 1.0 if m.active else -1.0 if m.fainted else 0.0

        vec = np.zeros(self.max_size, dtype=np.float32)
        vec[:len(self.members)] = [status(p) for p in self.members]
        return vec

    def get_active(self):
        return next((m for m in self.members if m.active), None)

    def describe(self) -> str:
        lines = [f"Team  ({self.alive_count()} alive / {self.max_size} slots):",
                 f"Alive vector: {self.alive_vector}",
                 self.active.describe() if self.active else "",]
        for i, member in enumerate(self.members):
            tag = "(empty)" if member.species == "none" else ""
            lines.append(
                f"  [{i}] {member.species:<12}  hp={member.hp:.2f}  {tag}"
            )
        lines.append(f"  Array length : {self.array_len()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()