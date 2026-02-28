from poke_env.battle import MoveCategory


def did_no_damage(battle, last_hp: dict, last_move) -> bool:
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

    if previous_hp is None:
        return False

    return current_hp >= previous_hp
