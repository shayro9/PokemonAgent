"""
BattleConfig — generation-specific configuration for the battle state and policy pipeline.

Threading the right state classes through the policy extractor, env wrapper, and policy
player so that adding Gen 2 requires only:
  1. New state classes (BattleStateGen2, MyPokemonStateGen2)
  2. A new factory:  BattleConfig.gen2()

No changes to core policy or env code are needed.

Usage
-----
    from env.battle_config import BattleConfig

    config = BattleConfig.gen1()
    env = PokemonRLWrapper(battle_config=config, ...)
    policy = AttentionPointerPolicy(obs_space, act_space, lr, battle_config=config)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BattleConfig:
    """
    Immutable configuration for a specific Pokémon generation.

    Holds the state class references and derives all observation-layout dimensions
    from them.  All properties are computed lazily from the class references, so
    construction is cheap and there is no import-time coupling to Gen-specific modules.

    Parameters
    ----------
    gen                  : generation number (1, 2, ...)
    battle_state_cls     : BattleState class for this gen.
                           Must implement ``array_len()`` and ``battle_before_me_len()``.
    my_pokemon_state_cls : MyPokemonState class for this gen.
                           Must implement ``array_len()``.
    action_space_size    : total number of discrete actions (moves + switches).
                           Gen 1 = 10 (6 switches + 4 moves).
    """

    gen: int
    battle_state_cls: type
    my_pokemon_state_cls: type
    action_space_size: int = 10

    # ── Derived dimensions ────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        """Total observation vector length."""
        return self.battle_state_cls.array_len()

    @property
    def arena_opponent_len(self) -> int:
        """Length of the context slice (arena + opponent state) before my_pokemon."""
        return self.battle_state_cls.battle_before_me_len()

    @property
    def my_pokemon_len(self) -> int:
        """Length of a single my-team pokemon embedding (pokemon state + moves)."""
        return self.my_pokemon_state_cls.array_len()

    @property
    def context_len(self) -> int:
        """Full context length: arena + opponent + my active pokemon."""
        return self.arena_opponent_len + self.my_pokemon_len

    @property
    def move_len(self) -> int:
        """Length of a single MoveState feature vector."""
        from env.states.move_state import MoveState
        return MoveState.array_len()

    @property
    def my_moves_start(self) -> int:
        """Offset within the my_pokemon block where move features begin."""
        from env.states.state_utils import MAX_MOVES
        return self.my_pokemon_len - self.move_len * MAX_MOVES

    @property
    def n_switch_actions(self) -> int:
        """Number of team slots available for switching (= MAX_TEAM_SIZE = 6)."""
        from env.states.state_utils import MAX_TEAM_SIZE
        return MAX_TEAM_SIZE

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def gen1(cls) -> "BattleConfig":
        """Return the canonical configuration for Gen 1."""
        from env.states.gen1.battle_state_gen_1 import BattleStateGen1
        from env.states.gen1.my_pokemon_state_gen_1 import MyPokemonStateGen1
        return cls(
            gen=1,
            battle_state_cls=BattleStateGen1,
            my_pokemon_state_cls=MyPokemonStateGen1,
            action_space_size=10,
        )
