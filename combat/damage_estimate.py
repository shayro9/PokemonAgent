"""
combat/damage_estimate.py
==========================
Estimate the expected damage fraction a move will deal against the
opponent, using the current ``StatBelief`` posterior mean for their
defense stat and ``opp.max_hp`` read directly from the battle.

The result is a single float in ``[0, 1]`` representing the fraction
of the opponent's max HP the move is expected to remove.  This is
appended as one extra scalar to each of our move embeddings so the
attention-pointer head has a directly actionable "how much does this
hurt" signal per move.

Formula (simplified Gen-9, same as belief inference):
    D_raw = (level_factor * BP * A_eff / D_eff / 50 + 2) * modifier
    damage_fraction = D_raw / opp_max_hp

Where:
    A_eff  = my base attack * my boost multiplier   (known exactly)
    D_eff  = belief mean for opp def/spd * opp boost multiplier
    modifier from calc_modifier (STAB, type, weather, screens, crit)
"""

from __future__ import annotations

from poke_env.battle import MoveCategory

from combat.stats_belief import StatBelief, DEF_IDX, SPD_IDX, level_factor, HP_IDX
from combat.combat_utils import calc_modifier, boost_multiplier


def estimate_move_damage_fraction(
    move,
    battle,
    stat_belief: StatBelief,
) -> float:
    """Estimate the fraction of the opponent's max HP this move will deal.

    Returns ``0.0`` for status moves, moves with no base power, or when
    ``opp.max_hp`` is not yet known (early in battle before any damage).

    :param move: The move to evaluate.
    :param battle: poke-env battle object (provides me, opp, gen).
    :param stat_belief: Current posterior belief over opponent stats.
    :returns: Expected damage as a fraction of opp max HP, clipped to ``[0, 1]``.
    """
    if move.category == MoveCategory.STATUS:
        return 0.0

    bp = float(move.base_power or 0)
    if bp <= 0:
        return 0.0

    me  = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    opp_max_hp = stat_belief.mean[HP_IDX]
    if opp_max_hp <= 0:
        return 0.0

    is_special = (move.category == MoveCategory.SPECIAL)
    atk_key = "spa" if is_special else "atk"
    def_idx = SPD_IDX if is_special else DEF_IDX
    def_key = "spd" if is_special else "def"

    # Our effective attack (known exactly)
    my_atk_eff = float(me.stats[atk_key]) * boost_multiplier(me.boosts.get(atk_key, 0))

    # Opponent effective defense (belief mean * their boost)
    opp_def_base = float(stat_belief.mean[def_idx])
    opp_def_eff  = opp_def_base * boost_multiplier(opp.boosts.get(def_key, 0))
    if opp_def_eff <= 0:
        return 0.0

    lf = level_factor(me.level)
    modifier, _ = calc_modifier(move, me, opp, battle, attacker_is_us=True)

    d_raw = (lf * bp * my_atk_eff / opp_def_eff / 50.0 + 2.0) * modifier
    damage_fraction = d_raw / opp_max_hp
    return float(min(1.0, max(0.0, damage_fraction)))
