from .embed import *

MAX_MOVES = 4
MOVE_EMBED_LEN = 4 + len(MoveCategory) + 1 + len(Status) + 7 + 7 + 2 + 2

ACTION_DEFAULT = -2
ACTION_FORFEIT = -1
ACTION_SWITCH_RANGE = range(0, 6)
ACTION_MOVE_RANGE = range(6, 10)
ACTION_MEGA_RANGE = range(10, 14)
ACTION_ZMOVE_RANGE = range(14, 18)
ACTION_DYNAMAX_RANGE = range(18, 22)
ACTION_TERASTALLIZE_RANGE = range(22, 26)


def _slot_is_available(sequence, slot: int) -> bool:
    """Check whether a slot index exists in a sequence.
    
    :param sequence: Sequence of available actions.
    :param slot: Zero-based slot index.
    :returns: ``True`` when the slot exists, otherwise ``False``."""
    return 0 <= slot < len(sequence)


def get_valid_action_mask(
        battle,
        *,
        allow_switches: bool = False,
        allow_moves: bool = True,
        allow_mega: bool = False,
        allow_zmove: bool = False,
        allow_dynamax: bool = False,
        allow_terastallize: bool = False,
) -> np.ndarray:
    """Build a boolean mask over the canonical action space ``[0..25]``.
    
    :param battle: Battle instance containing currently available actions.
    :param allow_switches: Whether switch actions should be considered valid.
    :param allow_moves: Whether move actions should be considered valid.
    :param allow_mega: Whether mega-evolution move variants are allowed.
    :param allow_zmove: Whether Z-move variants are allowed.
    :param allow_dynamax: Whether dynamax move variants are allowed.
    :param allow_terastallize: Whether terastallized move variants are allowed.
    :returns: A NumPy boolean vector where ``True`` entries indicate valid actions."""
    mask = np.zeros(26, dtype=bool)

    if allow_switches:
        available_switches = getattr(battle, "available_switches", [])
        for slot, action in enumerate(ACTION_SWITCH_RANGE):
            mask[action] = _slot_is_available(available_switches, slot)

    if allow_moves:
        available_moves = getattr(battle, "available_moves", [])
        all_moves = getattr(battle.active_pokemon, "moves", []).values()
        move_mask = [move in available_moves for move in all_moves]
        if not any(move_mask):
            move_mask = [1, 0, 0, 0]
        while len(move_mask) < MAX_MOVES:
            move_mask.append(False)
        for slot, action in enumerate(ACTION_MOVE_RANGE):
            mask[action] = move_mask[slot]

        if allow_mega and getattr(battle, "can_mega_evolve", False):
            for slot, action in enumerate(ACTION_MEGA_RANGE):
                mask[action] = _slot_is_available(available_moves, slot)

        if allow_zmove and getattr(battle, "can_z_move", False):
            available_z_moves = getattr(battle, "available_z_moves", [])
            for slot, action in enumerate(ACTION_ZMOVE_RANGE):
                mask[action] = _slot_is_available(available_z_moves, slot)

        if allow_dynamax and getattr(battle, "can_dynamax", False):
            for slot, action in enumerate(ACTION_DYNAMAX_RANGE):
                mask[action] = _slot_is_available(available_moves, slot)

        if allow_terastallize and getattr(battle, "can_tera", False):
            for slot, action in enumerate(ACTION_TERASTALLIZE_RANGE):
                mask[action] = _slot_is_available(available_moves, slot)

    return mask
