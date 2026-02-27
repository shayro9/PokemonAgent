import numpy as np
from functools import lru_cache
from typing import List

from poke_env.data import GenData
from poke_env.battle import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.status import Status

BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
MOVE_CATEGORIES = tuple(MoveCategory)
MOVE_STATUSES = tuple(Status)


def embed_status(status) -> np.ndarray:
    vec = [1.0 if status == s else 0.0 for s in MOVE_STATUSES]
    return np.array(vec, dtype=np.float32)


@lru_cache(maxsize=None)
def _type_chart_for_gen(gen: int):
    """Cache generation type charts to avoid repeated GenData lookups per move."""
    return GenData.from_gen(gen).type_chart


def _iter_scaled_boosts(boosts: dict | None):
    if not boosts:
        for _ in BOOST_KEYS:
            yield 0.0
        return

    for key in BOOST_KEYS:
        yield _scale_m11(boosts.get(key, 0), 6.0)


def _scale_01(x: float, max_x: float = 1) -> float:
    return min(1.0, max(0.0, x / max_x)) if max_x > 0 else 0.0


def _scale_m11(x: float, max_abs: float) -> float:
    return max(-1.0, min(1.0, x / max_abs)) if max_abs > 0 else 0.0


def _safe_int(move, key, default=0):
    entry = getattr(move, "entry", None)
    if isinstance(entry, dict):
        return int(entry.get(key, 0) or 0)
    return default


def embed_move(move: Move, opp_types, gen) -> np.ndarray:
    """
    Embeds a poke_env Move object into a fixed-length vector.
    Fully based on poke-env enums (no hardcoded strings).
    """
    vec: List[float] = []
    # ---------------------------------------------------
    # 1) Scalars
    # ---------------------------------------------------
    vec.append(_scale_01(move.base_power or 0, 200.0))

    if move.accuracy is True:
        vec.append(1.0)
    else:
        vec.append(_scale_01(move.accuracy or 0))

    vec.append(_scale_01(move.max_pp or 0, 40.0))
    vec.append(_scale_m11(_safe_int(move, "priority", 0), 7.0))

    # ---------------------------------------------------
    # 2) Category one-hot
    # ---------------------------------------------------
    for cat in MOVE_CATEGORIES:
        vec.append(1.0 if move.category == cat else 0.0)

    # ---------------------------------------------------
    # 3) Type multies
    # ---------------------------------------------------
    type1, type2 = (opp_types + [None])[:2]
    mult = move.type.damage_multiplier(type1, type2, type_chart=_type_chart_for_gen(gen))
    vec.append(-1.0 if mult == 0.0 else float(np.log2(mult) / 2.0))
    # ---------------------------------------------------
    # 4) Flags
    # ---------------------------------------------------

    # ---------------------------------------------------
    # 5) Status inflicted
    # ---------------------------------------------------
    for s in MOVE_STATUSES:
        vec.append(1.0 if move.status == s else 0.0)

    # ---------------------------------------------------
    # 6) Boosts (target)
    # ---------------------------------------------------
    vec.extend(_iter_scaled_boosts(move.boosts))

    # ---------------------------------------------------
    # 7) Self boosts
    # ---------------------------------------------------
    vec.extend(_iter_scaled_boosts(getattr(move, "self_boost", None)))

    # ---------------------------------------------------
    # 8) Recoil / Drain
    # ---------------------------------------------------
    vec.append(0 if not move.recoil else -move.recoil)
    vec.append(move.drain or 0.0)
    # ---------------------------------------------------
    # 9) Multihit
    # ---------------------------------------------------
    if isinstance(move.n_hit, tuple):
        min_hits, max_hits = move.n_hit
    elif isinstance(move.n_hit, int):
        min_hits = max_hits = move.n_hit
    else:
        min_hits = max_hits = 1

    vec.append(_scale_01(min_hits, 5.0))
    vec.append(_scale_01(max_hits, 5.0))

    return np.array(vec, dtype=np.float32)


def calc_types_vector(my_types: list[PokemonType], opp_types: list[PokemonType], gen: int):
    vec = []

    my_types = list(my_types) + [None]
    opp_types = list(opp_types) + [None]

    my_types = my_types[:2]
    opp_types = opp_types[:2]

    type_chart = _type_chart_for_gen(gen)

    for my_t in my_types:
        for opp_t in opp_types:
            if my_t is None or opp_t is None:
                vec.append(0.0)  # neutral
            else:
                mult = my_t.damage_multiplier(opp_t, type_chart=type_chart)
                vec.append(-1.0 if mult == 0.0 else float(np.log2(mult)))

    return np.array(vec, dtype=np.float32)
