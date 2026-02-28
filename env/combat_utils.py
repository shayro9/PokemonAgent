from poke_env.battle import MoveCategory, Move


def did_no_damage(battle, last_hp: dict, last_move, eps = 1e-6) -> bool:
    """
    Returns True if our last action did no damage to the opponent.
    Requires last_move to be a physical or special move (not status).
    """
    if last_move is None:
        return False
    if last_move.category == MoveCategory.STATUS:
        return False

    opp = battle.opponent_active_pokemon
    opp_key = f"opp_{opp.species}"

    current_hp = opp.current_hp_fraction
    previous_hp = last_hp.get(opp_key)
    if previous_hp is None or current_hp is None:
        return False

    # True if HP did not drop (within tolerance)
    return current_hp >= previous_hp - eps


def protect_chance(last_move: Move, last_chance: float = 1.0, no_damage: bool = False) -> float:
    if not no_damage:
        # We hit and they didn't protect → X resets
        return 1.0

    accuracy = (last_move.accuracy if isinstance(last_move.accuracy, float) else 1.0)
    p_hit = accuracy
    p_miss = 1.0 - p_hit

    p_scenario_1 = p_hit * last_chance  # X triples
    p_scenario_2 = p_miss * last_chance  # X triples
    p_scenario_3 = p_miss * (1.0 - last_chance)  # X resets

    next_if_protected = last_chance / 3
    next_if_reset = 1.0

    return (p_scenario_1 + p_scenario_2) * next_if_protected + p_scenario_3 * next_if_reset
