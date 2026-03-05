"""
combat/event_parser.py
======================
Parse poke-env battle observation events to replace fragile HP-diff and
PP-diff heuristics with direct event evidence.

Every public function accepts a ``battle`` object and reads
``battle.observations[battle.turn - 1].events`` — the list of events that
resolved on the turn *before* we must now act.

Event structure (inner list)
-----------------------------
    index 0 : protocol prefix (usually empty string)
    index 1 : event type, e.g. ``'move'``, ``'-damage'``, ``'-miss'``
    index 2+: event arguments (actor, target, move name, …)

Example turn events::

    ['', 'move', 'p2a: Regirock', 'Body Slam', 'p1a: Steelix']
    ['', '-resisted', 'p1a: Steelix']
    ['', '-damage', 'p1a: Steelix', '335/354']
    ['', 'move', 'p1a: Steelix', 'Toxic', 'p2a: Regirock']
    ['', '-status', 'p2a: Regirock', 'tox']
    ['', '-damage', 'p2a: Regirock', '342/364 tox', '[from] psn']
"""

from __future__ import annotations
from poke_env.battle import Move, MoveCategory


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_prev_turn_events(battle) -> list[list[str]]:
    """Return the event list for the turn that just resolved.

    :param battle: poke-env battle object.
    :returns: List of event rows, or an empty list when unavailable.
    """
    obs = battle.observations.get(battle.turn - 1)
    if obs is None:
        return []
    return getattr(obs, "events", [])


def _my_prefix(battle) -> str:
    """Return our player slot prefix, e.g. ``'p1a'``.

    :param battle: poke-env battle object.
    :returns: Slot prefix string.
    """
    role = getattr(battle, "player_role", None) or "p1"
    return f"{role}a"


def _opp_prefix(battle) -> str:
    """Return the opponent's player slot prefix, e.g. ``'p2a'``.

    :param battle: poke-env battle object.
    :returns: Slot prefix string.
    """
    role = getattr(battle, "player_role", None) or "p1"
    opp = "p2" if role == "p1" else "p1"
    return f"{opp}a"


def _to_move_id(name: str) -> str:
    """Normalise a display move name to a poke-env move ID.

    :param name: Display name such as ``'Body Slam'``.
    :returns: Lowercase, punctuation-stripped ID such as ``'bodyslam'``.
    """
    return name.lower().replace(" ", "").replace("-", "").replace("'", "").replace(".", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_opponent_move_from_events(battle) -> Move | None:
    """Detect the move the opponent used last turn from the event log.

    Looks for the first ``move`` event where the actor belongs to the
    opponent's slot, then resolves the move name against the opponent's
    already-revealed move dict.

    :param battle: poke-env battle object.
    :returns: The opponent's ``Move`` object, or ``None`` if not found.
    """
    events = _get_prev_turn_events(battle)
    opp_pfx = _opp_prefix(battle)

    for event in events:
        if len(event) >= 4 and event[1] == "move" and event[2].startswith(opp_pfx):
            move_id = _to_move_id(event[3])
            return (battle.opponent_active_pokemon.moves or {}).get(move_id)

    return None


def did_no_damage_from_events(battle, my_last_move: Move | None) -> bool:
    """Check whether our last move dealt no HP damage to the opponent.

    Returns ``False`` immediately for status moves or when no move was used.
    Otherwise scans the events after our ``move`` marker and looks for a
    ``-damage`` event on the opponent that is *not* attributed to a
    secondary source (``[from] psn``, etc.).

    :param battle: poke-env battle object.
    :param my_last_move: The move we played last turn, or ``None``.
    :returns: ``True`` if no direct damage was dealt, ``False`` otherwise.
    """
    if my_last_move is None:
        return False
    if my_last_move.category == MoveCategory.STATUS:
        return False

    events = _get_prev_turn_events(battle)
    my_pfx = _my_prefix(battle)
    opp_pfx = _opp_prefix(battle)

    # Locate our move event
    our_move_idx: int | None = None
    for i, event in enumerate(events):
        if len(event) >= 3 and event[1] == "move" and event[2].startswith(my_pfx):
            our_move_idx = i
            break

    if our_move_idx is None:
        # Our move wasn't found (e.g. we switched) — treat as no damage
        return False

    # Scan events following our move until the next 'move' event
    for event in events[our_move_idx + 1:]:
        if len(event) < 2:
            continue
        if event[1] == "move":
            break  # reached a different actor's move — stop
        if (
            len(event) >= 3
            and event[1] == "-damage"
            and event[2].startswith(opp_pfx)
            and not any("[from]" in part for part in event)
        ):
            return False  # direct damage was dealt

    return True


def we_moved_first_from_events(battle) -> bool | None:
    """Determine turn order by checking which ``move`` event appeared first.

    :param battle: poke-env battle object.
    :returns: ``True`` if we acted first, ``False`` if the opponent did,
        ``None`` if no move events were found (e.g. both switched).
    """
    events = _get_prev_turn_events(battle)
    my_pfx = _my_prefix(battle)
    opp_pfx = _opp_prefix(battle)

    for event in events:
        if len(event) >= 3 and event[1] == "move":
            actor = event[2]
            if actor.startswith(my_pfx):
                return True
            if actor.startswith(opp_pfx):
                return False

    return None
