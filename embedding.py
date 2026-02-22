import numpy as np
from typing import Any, List

from poke_env.data import GenData
from poke_env.battle import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.status import Status


def _scale_01(x: float, max_x: float = 1) -> float:
    return min(1.0, max(0.0, x / max_x)) if max_x > 0 else 0.0


def _scale_m11(x: float, max_abs: float) -> float:
    return max(-1.0, min(1.0, x / max_abs)) if max_abs > 0 else 0.0


def embed_move(move: Move) -> np.ndarray:
    """
    Embeds a poke_env Move object into a fixed-length vector.
    Fully based on poke-env enums (no hardcoded strings).
    """
    print(move.boosts)
    vec: List[float] = []
    # ---------------------------------------------------
    # 2) Scalars
    # ---------------------------------------------------
    vec.append(_scale_01(move.base_power or 0, 200.0))

    if move.accuracy is True:
        vec.append(1.0)
    else:
        vec.append(_scale_01(move.accuracy or 0))

    vec.append(_scale_01(move.max_pp or 0, 40.0))
    vec.append(_scale_m11(move.priority or 0, 7.0))

    # ---------------------------------------------------
    # 2) Category one-hot
    # ---------------------------------------------------
    for cat in MoveCategory:
        vec.append(1.0 if move.category == cat else 0.0)

    # ---------------------------------------------------
    # 3) Type multies
    # ---------------------------------------------------

    # ---------------------------------------------------
    # 4) Flags
    # ---------------------------------------------------

    # ---------------------------------------------------
    # 5) Status inflicted
    # ---------------------------------------------------
    for s in Status:
        vec.append(1.0 if move.status == s else 0.0)

    # ---------------------------------------------------
    # 6) Boosts (target)
    # ---------------------------------------------------
    boost_keys = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

    for key in boost_keys:
        val = move.boosts.get(key, 0) if move.boosts else 0
        vec.append(_scale_m11(val, 6.0))

    # ---------------------------------------------------
    # 7) Self boosts
    # ---------------------------------------------------
    self_boosts = move.self_boost if hasattr(move, "self_boost") else None
    if self_boosts:
        for key in boost_keys:
            val = self_boosts.get(key, 0)
            vec.append(_scale_m11(val, 6.0))
    else:
        vec.extend([0.0] * len(boost_keys))

    # ---------------------------------------------------
    # 8) Recoil / Drain
    # ---------------------------------------------------

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

    for my_t in my_types:
        for opp_t in opp_types:
            if my_t is None or opp_t is None:
                vec.append(0.0)  # neutral
            else:
                mult = my_t.damage_multiplier(opp_t, type_chart=GenData.from_gen(gen).type_chart)
                vec.append(-1.0 if mult == 0.0 else float(np.log2(mult)))

    return np.array(vec, dtype=np.float32)