from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from combat.beliefs.stats_belief import StatBelief


@dataclass
class BattleTracker:
    """All mutable per-battle state, in one place.

    One instance per battle_tag. Call commit() at the end of each turn
    (after reward has already read the old values).
    """

    # HP
    last_my_hp: float = 1.0
    last_opp_hp: float = 1.0

    # Status — snapshot from END of previous turn, not a live query
    last_my_status: Any = None   # poke_env Status | None
    last_opp_status: Any = None  # poke_env Status | None

    # Move / PP bookkeeping (was: my_last_move, last_opp_pp)
    my_last_move: Any = None
    last_opp_pp: dict = field(default_factory=dict)

    # Protect belief (was: last_protect_chance, protect_belief)
    last_protect_chance: float = 1.0
    protect_belief: float = 1.0

    stat_belief: Optional[StatBelief] = None

    @property
    def last_hp(self) -> tuple[float, float]:
        return self.last_my_hp, self.last_opp_hp

    @property
    def last_status(self) -> tuple[Any, Any]:
        return self.last_my_status, self.last_opp_status

    def commit(self, battle) -> None:
        """Snapshot current HP + status. Call AFTER reward has read old values."""
        self.last_my_hp = battle.active_pokemon.current_hp_fraction
        self.last_opp_hp = battle.opponent_active_pokemon.current_hp_fraction
        self.last_my_status = battle.active_pokemon.status
        self.last_opp_status = battle.opponent_active_pokemon.status