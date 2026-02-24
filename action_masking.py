from embedding import *

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
    return 0 <= slot < len(sequence)


def get_valid_action_mask(
        battle,
        *,
        allow_switches: bool = True,
        allow_moves: bool = True,
        allow_mega: bool = False,
        allow_zmove: bool = False,
        allow_dynamax: bool = False,
        allow_terastallize: bool = False,
) -> np.ndarray:
    """Returns a mask over the canonical action space [0..25]."""
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


def get_move_only_action_mask(self, battle) -> np.ndarray:
    """Returns a local [0..3] mask for the current 1v1 move-only setup."""
    canonical_mask = self.get_valid_action_mask(
        battle,
        allow_switches=False,
        allow_moves=True,
        allow_mega=False,
        allow_zmove=False,
        allow_dynamax=False,
        allow_terastallize=False,
    )
    return canonical_mask[ACTION_MOVE_RANGE.start:ACTION_MOVE_RANGE.stop]

