import numpy as np

from env.states.state_utils import MAX_MOVES, MAX_TEAM_SIZE


class ActionMaskGen1:

    ACTION_SPACE = 26
    ACTION_MOVE_RANGE = range(6, 10)
    ACTION_SWITCH_RANGE = range(0, 6)

    def __init__(self, allow_switches=False):

        self.allow_switches = allow_switches
        self.mask = np.zeros(self.ACTION_SPACE, dtype=bool)

    def get_mask(self) -> np.ndarray:
        return self.mask

    def set_mask(self, battle):
        if self.allow_switches:
            available_switches = getattr(battle, "available_switches", [])
            all_pokemons = getattr(battle, "team", []).values()
            self._set_mask_range(available_switches, all_pokemons, self.ACTION_SWITCH_RANGE)

        available_moves = getattr(battle, "available_moves", [])
        all_moves = getattr(battle.active_pokemon, "moves", []).values()
        self._set_mask_range(available_moves, all_moves, self.ACTION_MOVE_RANGE)

    def reset(self):
        self.mask = np.zeros(self.ACTION_SPACE, dtype=bool)

    def describe(self, battle=None) -> str:
        lines = []

        # --- Switches ---
        switch_lines = []
        if self.allow_switches:
            pokemons = []
            if battle is not None:
                pokemons = list(getattr(battle, "team", {}).values())

            for slot, action in enumerate(self.ACTION_SWITCH_RANGE):
                if not self.mask[action]:
                    continue

                if battle is not None and slot < len(pokemons):
                    name = getattr(pokemons[slot], "species", f"Pokemon_{slot + 1}")
                else:
                    name = f"Pokemon_{slot + 1}"

                switch_lines.append(f"[{action:02}] SWITCH {slot}: ({name})")

        # --- Moves ---
        move_lines = []
        moves = []
        if battle is not None:
            moves = list(getattr(battle.active_pokemon, "moves", {}).values())

        for slot, action in enumerate(self.ACTION_MOVE_RANGE):
            if not self.mask[action]:
                continue

            if battle is not None and slot < len(moves):
                name = getattr(moves[slot], "id", f"move_{slot + 1}")
            else:
                name = f"move_{slot + 1}"

            move_lines.append(f"[{action:02}] MOVE   {slot}: ({name})")

        # --- Build output ---
        lines.append("=== ACTION MASK ===")

        if self.allow_switches:
            lines.append("--- Switches ---")
            lines.extend(switch_lines if switch_lines else ["(none)"])

        lines.append("--- Moves ---")
        lines.extend(move_lines if move_lines else ["(none)"])

        # --- Summary ---
        valid_actions = np.where(self.mask)[0]
        lines.append(f"\nValid actions: {valid_actions.tolist()}")
        return "\n".join(lines)

    def _set_mask_range(self, available_items, all_items, action_range, move_action=False):
        _mask = [item in available_items for item in all_items]

        if not any(_mask) and move_action:
            _mask = [1, 0, 0, 0]  # Struggle
        while len(_mask) < len(action_range):
            _mask.append(False)

        for slot, action in enumerate(action_range):
            self.mask[action] = _mask[slot]