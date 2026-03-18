from __future__ import annotations

import numpy as np
from poke_env.battle import Move, MoveCategory

from combat.combat_utils import type_chart_for_gen
from env.states.state_utils import (
    normalize, normalize_vector, encode_enum, encode_dicts, pull_attribute,
    BOOST_NORM, GEN1_BOOST_KEYS, ALL_STATUSES,
)

MOVE_CATEGORIES = tuple(MoveCategory)


class MoveState:
    """Snapshot of a move's relevant battle state, ready for embedding."""
    BOOST_KEYS: list[str]   = GEN1_BOOST_KEYS

    def __init__(
        self,
        move: Move | None,
        opp_types,
        my_types,
        gen: int,
        # damage_fraction: float = 0.0,
    ):
        # Multi-hit bounds — 0 when no move, 1 as fallback for unknown n_hit on a real move
        if move is None:
            self.min_hits = self.max_hits = 0
        else:
            n_hit = pull_attribute(move, "n_hit", 1, lambda x: x)
            if isinstance(n_hit, tuple):
                self.min_hits, self.max_hits = n_hit
            elif isinstance(n_hit, int):
                self.min_hits = self.max_hits = n_hit
            else:
                self.min_hits = self.max_hits = 1

        # Type multiplier
        if move is not None:
            type1, type2 = (list(opp_types) + [None])[:2]
            self.type_multiplier = move.type.damage_multiplier(
                type1, type2, type_chart=type_chart_for_gen(gen)
            )
        else:
            self.type_multiplier = 1.0

        acc = pull_attribute(move, "accuracy", None, lambda x: x)
        self.id              = pull_attribute(move, "id",              None,  lambda x: x)
        self.base_power      = pull_attribute(move, "base_power",      0.0,   float)
        self.accuracy        = 1.0 if acc is True else pull_attribute(move, "accuracy", 0.0, float)
        self.max_pp          = pull_attribute(move, "max_pp",          0.0,   float)
        self.priority        = pull_attribute(move, "priority",        0,     int)
        self.heal            = pull_attribute(move, "heal",            0.0,   float)
        self.crit_ratio      = pull_attribute(move, "crit_ratio",      0.0,   float)
        self.category        = pull_attribute(move, "category",        None,  lambda x: x)
        self.is_protect_move = pull_attribute(move, "is_protect_move", False, float)
        self.breaks_protect  = pull_attribute(move, "breaks_protect",  False, float)
        self.is_stab         = float(pull_attribute(move, "type",      None,  lambda x: x) in my_types)
        self.status          = pull_attribute(move, "status",          None,  lambda x: x)
        self.opp_boosts      = dict(pull_attribute(move, "boosts",     {},    lambda x: x) or {})
        self.self_boost      = dict(pull_attribute(move, "self_boost", {},    lambda x: x) or {})
        self.recoil          = pull_attribute(move, "recoil",          0.0,   float)
        self.drain           = pull_attribute(move, "drain",           0.0,   float)

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
            normalize_vector(encode_dicts(self.opp_boosts, self.BOOST_KEYS), BOOST_NORM, symmetric=True),
            normalize_vector(encode_dicts(self.self_boost, self.BOOST_KEYS), BOOST_NORM, symmetric=True),
            self._encode_type_multiplier(self.type_multiplier),
            [self.recoil],
            [self.drain],
            [normalize(self.min_hits, 5.0)],
            [normalize(self.max_hits, 5.0)],
        ]).astype(np.float32)

        # vec.append(_scale_01(self.damage_fraction, 1.0))
        assert len(arr) == self.array_len(), f"embed: expected {self.array_len()}, got {len(arr)}"
        return arr

    @classmethod
    def array_len(cls) -> int:
        return 13 + len(MOVE_CATEGORIES) + len(ALL_STATUSES) + 2 * len(cls.BOOST_KEYS)

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