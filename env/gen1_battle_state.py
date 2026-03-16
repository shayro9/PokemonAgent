from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from env.pokemon_stats import PokemonStats

# ---------------------------------------------------------------------------
# Gen 1 OU – what does NOT exist compared to modern gens:
#   - No weather
#   - No Special Attack / Special Defense split (only "Special" stat)
#   - No held items / abilities / Tera / Mega / Z-moves
#   - Only 5 stats : HP, Atk, Def, Spc, Spe
#   - Boosts track  : atk, def, spc, spe, accuracy, evasion
# ---------------------------------------------------------------------------

MAX_MOVES = 4

# Observation size placeholder – updated once to_array() is finalised
GEN1_OBS_SIZE = 0   # TODO: update after move embedding is designed


@dataclass
class Gen1BattleState:
    """
    Structured observation for a Gen 1 OU battle.

    Fields intentionally stripped of anything that does not exist in Gen 1:
      - No weather
      - No abilities / held items
      - No Tera / Mega / Z-moves
      - No Sp.Atk / Sp.Def split  (uses single 'spc' stat)

    my_side and opp_side are PokemonStats(gen1=True) instances which hold
    hp, stats (5), boosts (6), status and effects all in one place.

    Call .to_array() to get the flat np.ndarray passed to the model.
    """

    # --- global ---
    turn: float                 # (1)  turn / 30.0

    # --- my pokemon ---
    my_side:    PokemonStats    # hp + stats + boosts + status + effects

    # --- opponent pokemon ---
    opp_side:       PokemonStats    # hp + stats (belief) + boosts + status + effects
    opp_preparing:  float           # (1)  0 or 1  (e.g. Hyper Beam recharge)

    # --- moves (4 × move_embed each side) ---
    my_moves:  np.ndarray       # (MAX_MOVES * move_embed_len,)
    opp_moves: np.ndarray       # (MAX_MOVES * move_embed_len,)

    # --- beliefs ---
    opp_protect_belief: float   # (1)

    # ------------------------------------------------------------------
    # factories
    # ------------------------------------------------------------------
    @classmethod
    def empty(cls) -> "Gen1BattleState":
        """Return an all-zero Gen1BattleState (useful for testing)."""
        return cls(
            turn=0.0,
            my_side=PokemonStats(gen1=True),
            opp_side=PokemonStats(gen1=True),
            opp_preparing=0.0,
            my_moves=np.array([],  dtype=np.float32),
            opp_moves=np.array([], dtype=np.float32),
            opp_protect_belief=1.0,
        )

    # ------------------------------------------------------------------
    # to_array  –  NOT implemented yet, waiting for move embed design
    # ------------------------------------------------------------------
    def to_array(self) -> np.ndarray:
        raise NotImplementedError(
            "Gen1BattleState.to_array() is not implemented yet. "
            "Define the Gen 1 move embedding first, then fill this in."
        )

    # ------------------------------------------------------------------
    # from_battle  –  NOT implemented yet
    # ------------------------------------------------------------------
    @classmethod
    def from_battle(
        cls,
        battle,
        opp_protect_belief: float = 1.0,
    ) -> "Gen1BattleState":
        raise NotImplementedError(
            "Gen1BattleState.from_battle() is not implemented yet."
        )

    # ------------------------------------------------------------------
    # describe  –  NOT implemented yet
    # ------------------------------------------------------------------
    def describe(self) -> str:
        raise NotImplementedError(
            "Gen1BattleState.describe() is not implemented yet."
        )
