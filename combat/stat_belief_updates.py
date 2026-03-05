"""
combat/stat_belief_updates.py
==============================
Poke-env-aware wrappers that translate raw battle state into
``StatBelief`` updates.

This module is intentionally the *only* place that imports both
poke-env types and ``StatBelief``.  ``stat_belief.py`` stays pure
(no poke-env dependency) and ``singles_env_wrapper.py`` stays thin
(no stat-inference logic).

Public API
----------
``update_stat_belief(belief, battle, tracker, opp_last_move)``
    Single entry point.  Returns a new ``StatBelief`` with all
    available evidence for this turn applied.
"""

from __future__ import annotations

from poke_env.battle import MoveCategory

from combat.stats_belief import StatBelief, build_stat_belief, level_factor
from combat.combat_utils import calc_modifier


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

    Initialises the belief from the prior if this is the first turn.
    Otherwise runs up to three independent Bayesian updates — one per
    evidence source — and returns the final posterior.

    :param belief: Current ``StatBelief``, or ``None`` on the first turn.
    :param battle: poke-env battle object.
    :param tracker: ``BattleTracker`` holding last-turn HP snapshots and
        the move we played.
    :param opp_last_move: The opponent's last move as detected by
        ``detect_opponent_move``, or ``None`` if unknown.
    :returns: Updated ``StatBelief``.
    """
    opp = battle.opponent_active_pokemon
    me = battle.active_pokemon

    if belief is None:
        return build_stat_belief(opp, battle.gen)

    opp_lf = level_factor(opp.level)
    my_lf = level_factor(me.level)
    belief = _update_from_damage_dealt(belief, battle, tracker, my_lf)
    belief = _update_from_damage_received(belief, battle, tracker, opp_last_move, opp_lf)
    belief = _update_from_speed_order(belief, battle)
    return belief


# ---------------------------------------------------------------------------
# Private helpers — one per evidence source
# ---------------------------------------------------------------------------

def _update_from_damage_dealt(
    belief: StatBelief,
    battle,
    tracker,
    lf: float,
) -> StatBelief:
    """Update opp Def or SpD from damage we dealt last turn.

    Uses the move we played (``tracker.my_last_move``) for exact BP
    and category, so no guessing is required.

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

    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    is_special = (my_move.category == MoveCategory.SPECIAL)
    atk_key = "spa" if is_special else "atk"
    my_atk = float(me.stats[atk_key])

    modifier, extra_noise_frac = calc_modifier(my_move, me, opp, battle, True)

    return belief.update_from_damage_dealt(
        damage_fraction=opp_hp_delta,
        my_attack=my_atk,
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

    Uses the opponent's detected move for exact BP and category when
    available.  Skips the update entirely when the move is unknown
    (rather than guessing) to avoid injecting noisy evidence.

    :param belief: Current belief.
    :param battle: poke-env battle object.
    :param tracker: Tracker holding ``last_my_hp``.
    :param opp_last_move: Opponent's last move, or ``None`` if not detected.
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

    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    is_special = (opp_last_move.category == MoveCategory.SPECIAL)
    def_key = "spd" if is_special else "def"
    my_def    = float(me.stats[def_key])
    my_max_hp = float(me.stats["hp"])

    modifier, extra_noise_frac = calc_modifier(opp_last_move, opp, me, battle, False)

    return belief.update_from_damage_received(
        damage_fraction=my_hp_delta,
        my_max_hp=my_max_hp,
        my_defense=my_def,
        base_power=bp,
        move_is_special=is_special,
        level_factor=lf,
        modifier=modifier,
        extra_noise_frac=extra_noise_frac,  # BP is known exactly — low extra noise
    )


def _update_from_speed_order(
    belief: StatBelief,
    battle,
) -> StatBelief:
    """Update opp Spe from turn-order evidence.

    Uses poke-env's ``preparing`` and ``must_recharge`` flags as a
    reliable signal that the opponent could not have moved first.
    Returns the belief unchanged when order is ambiguous.

    :param belief: Current belief.
    :param battle: poke-env battle object.
    :returns: Updated belief, or the original if order is ambiguous.
    """
    opp = battle.opponent_active_pokemon
    me  = battle.active_pokemon
    our_spe = float(me.stats["spe"])

    if getattr(opp, "preparing", False) or getattr(opp, "must_recharge", False):
        return belief.update_from_speed_order(our_spe=our_spe, we_moved_first=True)

    return belief