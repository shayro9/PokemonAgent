from __future__ import annotations
from typing import Optional, Tuple, Union, List

import numpy as np
from poke_env.battle import Move, MoveCategory, Status, PokemonType

from combat.combat_utils import type_chart_for_gen
from env.states.state_utils import (
    normalize, normalize_vector, encode_enum,
    BOOST_NORM, GEN1_BOOST_KEYS, ALL_STATUSES,
)

MOVE_CATEGORIES = tuple(MoveCategory)


class MoveState:
    """Snapshot of a move's relevant battle state, ready for embedding."""
    BOOST_KEYS: list[str]   = GEN1_BOOST_KEYS

    def __init__(
        self,
        move: Move,
        opp_types,
        my_types,
        gen: int,
        # damage_fraction: float = 0.0,
    ):
        # Multi-hit bounds
        if isinstance(move.n_hit, tuple):
            self.min_hits, self.max_hits = move.n_hit
        elif isinstance(move.n_hit, int):
            self.min_hits = self.max_hits = move.n_hit
        else:
            self.min_hits = self.max_hits = 1

        # Type multiplier
        type1, type2 = (list(opp_types) + [None])[:2]
        self.type_multiplier = move.type.damage_multiplier(
            type1, type2, type_chart=type_chart_for_gen(gen)
        )

        acc = getattr(move, "accuracy", None)
        self.id              = getattr(move, "id", None)
        self.base_power      = getattr(move, "base_power", 0.0) or 0.0
        self.accuracy        = 1.0 if acc is True else (acc or 0.0)
        self.max_pp          = getattr(move, "max_pp", 0.0) or 0.0
        self.priority        = getattr(move, "priority", 0)
        self.heal            = getattr(move, "heal", 0.0) or 0.0
        self.crit_ratio      = getattr(move, "crit_ratio", 0.0) or 0.0
        self.category        = getattr(move, "category", None)
        self.is_protect_move = float(getattr(move, "is_protect_move", False) or False)
        self.breaks_protect  = float(getattr(move, "breaks_protect", False) or False)
        self.is_stab         = float(getattr(move, "type", None) in my_types)
        self.status          = getattr(move, "status", None)
        self.opp_boosts      = dict(getattr(move, "boosts", {}) or {})
        self.self_boost      = dict(getattr(move, "self_boost", {}) or {})
        self.recoil          = getattr(move, "recoil", 0.0) or 0.0
        self.drain           = getattr(move, "drain", 0.0) or 0.0
        # self.damage_fraction = damage_fraction

    def to_array(self) -> np.ndarray:
        """Return the fixed-length feature vector for this move."""
        if getattr(self, "_is_zero", False):
            return np.zeros(self.array_len(), dtype=np.float32)

        arr = np.concatenate([
            [normalize(self.base_power, 200.0)],
            [self.accuracy],
            [normalize(self.priority, 7.0)],
            [normalize(self.heal, 1.0)],
            [normalize(self.crit_ratio, 6.0)],
            encode_enum(self.category, MOVE_CATEGORIES),
            [self.is_protect_move],
            [self.breaks_protect],
            [self.is_stab],
            encode_enum(self.status, ALL_STATUSES),
            normalize_vector(self.opp_boosts.values(), BOOST_NORM, symmetric=True),
            normalize_vector(self.self_boost.values(), BOOST_NORM, symmetric=True),
            self._encode_type_multiplier(self.type_multiplier),
            [self.recoil],
            [self.drain],
            [normalize(self.min_hits, 5.0)],
            [normalize(self.max_hits, 5.0)],
        ]).astype(np.float32)

        # vec.append(_scale_01(self.damage_fraction, 1.0))
        assert len(arr) == self.array_len(), f"embed: expected {self.array_len()}, got {len(arr)}"
        return arr

    def array_len(self) -> int:
        return 13 + len(MOVE_CATEGORIES) + len(ALL_STATUSES) + 2 * len(self.BOOST_KEYS)

    def describe(self) -> str:
        """Human-readable breakdown of the move state. Useful for debugging."""
        opp_boost_str = " | ".join(
            f"{k}={int(v):+d}" for k, v in self.opp_boosts.items() if v != 0
        )
        self_boost_str = " | ".join(
            f"{k}={int(v):+d}" for k, v in self.self_boost.items() if v != 0
        )
        lines = [
            f"Move           : {self.id}  "    
            f"Base power     : {self.base_power}",
            f"Category       : {self.category.name if self.category else 'none'}",
            f"Accuracy       : {self.accuracy}",
            f"Max PP         : {self.max_pp}",
            f"Priority       : {self.priority:+d}",
            f"Type multiplier: {self.type_multiplier}x",
            f"STAB           : {bool(self.is_stab)}",
            f"Hits           : {self.min_hits}–{self.max_hits}",
            f"Crit ratio     : {self.crit_ratio}",
            f"Heal           : {self.heal}",
            f"Recoil         : {self.recoil}",
            f"Drain          : {self.drain}",
            f"Status inflict : {self.status.name if self.status else 'none'}",
            f"Opp boosts     : {opp_boost_str if opp_boost_str else 'none'}",
            f"Self boosts    : {self_boost_str if self_boost_str else 'none'}",
            f"Protect move   : {bool(self.is_protect_move)}",
            f"Breaks protect : {bool(self.breaks_protect)}",
            f"Array length   : {self.array_len()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MoveState(base_power={self.base_power}, category={self.category}, "
            f"type_multiplier={self.type_multiplier})"
            # f"damage_fraction={self.damage_fraction})"
        )

    # ------------------------------------------------------------------
    # Default (null) move
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls) -> "MoveState":
        """return an empty move_state."""
        obj = cls.__new__(cls)  # bypass __init__
        obj._is_zero = True
        return obj

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_type_multiplier(type_multiplier: float) -> np.ndarray:
        val = -1.0 if type_multiplier == 0.0 else float(np.log2(type_multiplier) / 2.0)
        return np.array([val], dtype=np.float32)