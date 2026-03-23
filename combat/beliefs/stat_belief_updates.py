"""
combat/stat_belief_updates.py
==============================
Poke-env-aware wrappers that translate raw battle state into
``StatBelief`` updates.

Public API
----------
``update_stat_belief(belief, battle, tracker, opp_last_move)``
    Single entry point.  Returns a new ``StatBelief`` with all
    available evidence for this turn applied.
"""

from __future__ import annotations

from poke_env.battle import MoveCategory

from combat.beliefs.stats_belief import StatBelief, build_stat_belief, level_factor
from combat.combat_utils import calc_modifier, boost_multiplier
from combat.event_parser import moved_first


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def update_stat_belief(
    belief: StatBelief | None,
    battle,
    tracker,
    opp_last_move,
) -> StatBelief:
    """Apply all available turn evidence to the stat belief.

    :param belief: Current ``StatBelief``, or ``None`` on the first turn.
    :param battle: poke-env battle object.
    :param tracker: ``BattleTracker`` holding last-turn HP snapshots and
        the move we played.
    :param opp_last_move: The opponent's last move as detected by
        ``detect_opponent_move_from_events``, or ``None`` if unknown.
    :returns: Updated ``StatBelief``.
    """
    opp = battle.opponent_active_pokemon
    me  = battle.active_pokemon

    if belief is None:
        return build_stat_belief(opp, battle.gen)

    opp_lf = level_factor(opp.level)
    my_lf  = level_factor(me.level)
    belief = _update_from_damage_dealt(belief, battle, tracker, my_lf)
    belief = _update_from_damage_received(belief, battle, tracker, opp_last_move, opp_lf)
    belief = _update_from_speed_order(belief, battle)
    return belief


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _update_from_damage_dealt(
    belief: StatBelief,
    battle,
    tracker,
    lf: float,
) -> StatBelief:
    """Update opp Def or SpD from damage we dealt last turn.

    Boost adjustments applied here:
    - Our effective Atk/SpA  = base stat × our boost multiplier
    - Inferred opp effective Def/SpD is divided by opp's boost multiplier
      to recover the unboosted base stat the belief tracks.

    :param belief: Current belief.
    :param battle: poke-env battle object.
    :param tracker: Tracker holding ``my_last_move`` and ``last_opp_hp``.
    :param lf: Pre-computed level factor for the opponent.
    :returns: Updated belief, or the original if no valid evidence.
    """
    my_move = tracker.my_last_move
    if my_move is None:
        return belief

    bp = float(my_move.base_power or 0)
    if bp <= 0:
        return belief

    opp_hp_delta = tracker.last_opp_hp - battle.opponent_active_pokemon.current_hp_fraction
    if opp_hp_delta <= 0.01:
        return belief

    me  = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    is_special = (my_move.category == MoveCategory.SPECIAL)
    atk_key = "spa" if is_special else "atk"
    def_key = "spd" if is_special else "def"

    # Our effective attack includes our boost
    my_atk_base    = float(me.stats[atk_key])
    my_atk_boost   = boost_multiplier(me.boosts.get(atk_key, 0))
    my_atk_eff     = my_atk_base * my_atk_boost

    # Opp's boost — used to convert effective def back to base
    opp_def_boost  = boost_multiplier(opp.boosts.get(def_key, 0))

    modifier, extra_noise_frac = calc_modifier(my_move, me, opp, battle, True)

    return belief.update_from_damage_dealt(
        damage_fraction=opp_hp_delta,
        my_attack=my_atk_eff,           # boosted effective attack
        opp_def_boost=opp_def_boost,    # passed through so belief can un-boost
        base_power=bp,
        move_is_special=is_special,
        level_factor=lf,
        modifier=modifier,
        extra_noise_frac=extra_noise_frac,
    )


def _update_from_damage_received(
    belief: StatBelief,
    battle,
    tracker,
    opp_last_move,
    lf: float,
) -> StatBelief:
    """Update opp Atk or SpA from damage we received last turn.

    Boost adjustments applied here:
    - Our effective Def/SpD  = base stat × our boost multiplier
    - Inferred opp effective Atk/SpA is divided by opp's boost multiplier
      to recover the unboosted base stat the belief tracks.

    :param belief: Current belief.
    :param battle: poke-env battle object.
    :param tracker: Tracker holding ``last_my_hp``.
    :param opp_last_move: Opponent's last move from event parser, or ``None``.
    :param lf: Pre-computed level factor for the opponent.
    :returns: Updated belief, or the original if no valid evidence.
    """
    if opp_last_move is None:
        return belief

    my_hp_delta = tracker.last_my_hp - battle.active_pokemon.current_hp_fraction
    if my_hp_delta <= 0.01:
        return belief

    bp = float(opp_last_move.base_power or 0)
    if bp <= 0:
        return belief

    me  = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    is_special = (opp_last_move.category == MoveCategory.SPECIAL)
    def_key = "spd" if is_special else "def"
    atk_key = "spa" if is_special else "atk"

    # Our effective defense includes our boost
    my_def_base   = float(me.stats[def_key])
    my_def_boost  = boost_multiplier(me.boosts.get(def_key, 0))
    my_def_eff    = my_def_base * my_def_boost

    my_max_hp     = float(me.stats["hp"])

    # Opp's boost — used to convert effective atk back to base
    opp_atk_boost = boost_multiplier(opp.boosts.get(atk_key, 0))

    modifier, extra_noise_frac = calc_modifier(opp_last_move, opp, me, battle, False)

    return belief.update_from_damage_received(
        damage_fraction=my_hp_delta,
        my_max_hp=my_max_hp,
        my_defense=my_def_eff,          # boosted effective defense
        opp_atk_boost=opp_atk_boost,    # passed through so belief can un-boost
        base_power=bp,
        move_is_special=is_special,
        level_factor=lf,
        modifier=modifier,
        extra_noise_frac=extra_noise_frac,
    )


def _update_from_speed_order(
    belief: StatBelief,
    battle,
) -> StatBelief:
    """Update opp Spe from turn-order evidence parsed directly from events."""
    # Speed boosts affect turn order — apply our boost to the observation too
    our_spe_base  = float(battle.active_pokemon.stats["spe"])
    our_spe_boost = boost_multiplier(battle.active_pokemon.boosts.get("spe", 0))
    our_spe_eff   = our_spe_base * our_spe_boost

    we_moved_first = moved_first(battle)
    if we_moved_first is None:
        return belief

    return belief.update_from_speed_order(
        our_spe=our_spe_eff,
        we_moved_first=we_moved_first,
    )