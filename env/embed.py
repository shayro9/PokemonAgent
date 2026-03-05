import numpy as np

from typing import List

from poke_env.battle import Move, Weather
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect

from combat.combat_utils import type_chart_for_gen

BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
MOVE_CATEGORIES = tuple(MoveCategory)
MOVE_STATUSES = tuple(Status)
TRACKED_EFFECTS = [Effect.CONFUSION, Effect.MUST_RECHARGE, Effect.ENCORE]
WEATHERS = tuple(Weather)

MAX_MOVES = 4
MOVE_EMBED_LEN = 39


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


def embed_move(
    move: Move,
    opp_types,
    my_types,
    gen: int,
    damage_fraction: float = 0.0,
) -> np.ndarray:
    """Encode a move into a fixed-length numeric feature vector.

    :param move: Move instance to embed.
    :param opp_types: Defender typing iterable.
    :param my_types: Attacker typing iterable.
    :param gen: Battle generation number.
    :param damage_fraction: Pre-computed expected damage as a fraction of
        the opponent's max HP in ``[0, 1]``.  Pass ``0.0`` (default) when
        the estimate is unavailable (e.g. for opponent moves).
    :returns: A NumPy vector with ``MOVE_EMBED_LEN`` float features.
    """
    vec: List[float] = list()

    # Scalars
    vec.append(_scale_01(move.base_power or 0, 200.0))
    vec.append(1.0 if move.accuracy is True else _scale_01(move.accuracy or 0))
    vec.append(_scale_01(move.max_pp or 0, 40.0))
    vec.append(_scale_m11(_safe_int(move, "priority", 0), 7.0))
    vec.append(_scale_01(move.heal or 0, 1))
    vec.append(_scale_01(move.crit_ratio or 0, 6))

    # Category one-hot
    for cat in MOVE_CATEGORIES:
        vec.append(1.0 if move.category == cat else 0.0)

    # Protect moves
    vec.append(move.is_protect_move)
    vec.append(move.breaks_protect)

    # Type multiplier
    type1, type2 = (list(opp_types) + [None])[:2]
    mult = move.type.damage_multiplier(type1, type2, type_chart=type_chart_for_gen(gen))
    vec.append(-1.0 if mult == 0.0 else float(np.log2(mult) / 2.0))
    vec.append(1.0 if move.type in my_types else 0.0)

    # Status inflicted
    for s in MOVE_STATUSES:
        vec.append(1.0 if move.status == s else 0.0)

    # Boosts target
    vec.extend(_iter_scaled_boosts(move.boosts))

    # Self boosts
    vec.extend(_iter_scaled_boosts(getattr(move, "self_boost", None)))

    # Recoil / Drain
    vec.append(0 if not move.recoil else -move.recoil)
    vec.append(move.drain or 0.0)

    # Multihit
    if isinstance(move.n_hit, tuple):
        min_hits, max_hits = move.n_hit
    elif isinstance(move.n_hit, int):
        min_hits = max_hits = move.n_hit
    else:
        min_hits = max_hits = 1

    vec.append(_scale_01(min_hits, 5.0))
    vec.append(_scale_01(max_hits, 5.0))

    # Expected damage fraction — pre-computed by caller from StatBelief
    vec.append(_scale_01(damage_fraction, 1.0))

    result = np.array(vec, dtype=np.float32)
    assert len(result) == MOVE_EMBED_LEN, f"embed_move: expected {MOVE_EMBED_LEN}, got {len(result)}"
    return result


def embed_status(status) -> np.ndarray:
    return np.array([1.0 if status == s else 0.0 for s in MOVE_STATUSES], dtype=np.float32)


def embed_effects(effects) -> np.ndarray:
    return np.array([int(e in effects) for e in TRACKED_EFFECTS])


def embed_weather(weather) -> np.ndarray:
    return np.array([1.0 if weather == w else 0.0 for w in WEATHERS], dtype=np.float32)


def calc_types_vector(
    my_types: list[PokemonType],
    opp_types: list[PokemonType],
    gen: int,
    opp_tera_mode: bool = False,
) -> np.ndarray:
    my_types = (list(my_types) + [None])[:2]
    type_chart = type_chart_for_gen(gen)

    def _mult(my_t, opp_t) -> float:
        if my_t is None or opp_t is None:
            return 0.0
        mult = my_t.damage_multiplier(opp_t, type_chart=type_chart)
        return -1.0 if mult == 0.0 else float(np.log2(mult))

    if opp_tera_mode:
        return np.array([_mult(t, opp_types[0]) for t in my_types], dtype=np.float32)

    opp_types = (list(opp_types) + [None])[:2]
    vec = [_mult(my_t, opp_t) for my_t in my_types for opp_t in opp_types]
    return np.array(vec, dtype=np.float32)