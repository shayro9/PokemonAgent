from dataclasses import dataclass

from poke_env.battle import MoveCategory, Move

from env.embed import MAX_MOVES


def hp_history_key(battle) -> str:
    """Build a stable HP-history key scoped to battle and opponent species."""
    species = getattr(getattr(battle, "opponent_active_pokemon", None), "species", None) or "unknown"
    return f"{battle.battle_tag}|{species}"


def did_no_damage(battle, last_hp: dict, my_last_move, eps=1e-6) -> bool:
    """
    Returns True if our last action did no damage to the opponent.
    Requires my_last_move to be a physical or special move (not status).
    """
    if my_last_move is None:
        return False
    if my_last_move.category == MoveCategory.STATUS:
        return False

    opp = battle.opponent_active_pokemon

    battle_key = hp_history_key(battle)
    current_hp = opp.current_hp_fraction
    _, previous_hp = last_hp.get(battle_key, (1.0, 1.0))

    # True if HP did not drop (within tolerance)
    return current_hp >= previous_hp - eps


def clip_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def detect_opponent_move(battle, last_pp: dict) -> Move | None:
    """Detect which move the opponent used by comparing PP to last turn's snapshot.

    :param battle: The current battle state.
    :param last_pp: Dict mapping move_id -> pp from the previous turn.
    :returns: The Move that was used, or None if no change was detected.
    """
    moves = (battle.opponent_active_pokemon.moves or {}).values()
    for move in moves:
        if last_pp.get(move.id, move.current_pp) > move.current_pp or move.id not in last_pp.keys():
            return move

    return None


def snapshot_opponent_pp(battle) -> dict:
    """Snapshot the opponent's current PP for all revealed moves.

    :param battle: The current battle state.
    :returns: Dict mapping move_id -> current_pp.
    """
    return {
        move.id: move.current_pp
        for move in (battle.opponent_active_pokemon.moves or {}).values()
    }
