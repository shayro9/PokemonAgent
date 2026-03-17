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
        return 14 + len(MOVE_CATEGORIES) + len(ALL_STATUSES) + 2 * len(self.BOOST_KEYS)

    def describe(self) -> str:
        """Human-readable breakdown of the pokemon state. Useful for debugging."""
        active_status  = [ALL_STATUSES[i].name for i, v in enumerate(self.status)  if v == 1.0]
        boost_lines = " | ".join(
            f"{k}={int(v):+d}" for k, v in zip(self.BOOST_KEYS, self.boosts) if v != 0
        )

        lines = [
            f"Species       : {self.species}",
            f"HP            : {self.hp:.2f}",
            f"Stats         : {stat_lines}",
            f"Boosts        : {boost_lines if boost_lines else 'none'}",
            f"Status        : {active_status  if active_status  else 'none'}",
            f"Effects       : {active_effects if active_effects else 'none'}",
            f"STAB          : {self.stab}",
            f"Array length  : {self.array_len()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()
    def __repr__(self) -> str:
        return (
            f"MoveState(base_power={self.base_power}, category={self.category}, "
            f"type_multiplier={self.type_multiplier}, "
            # f"damage_fraction={self.damage_fraction})"
        )

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_type_multiplier(type_multiplier: float) -> np.ndarray:
        return np.array(-1.0 if type_multiplier == 0.0 else float(np.log2(type_multiplier) / 2.0))