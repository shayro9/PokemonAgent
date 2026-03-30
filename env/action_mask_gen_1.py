import numpy as np
from sympy.codegen.abstract_nodes import List

from env.states.state_utils import MAX_MOVES, MAX_TEAM_SIZE


class ActionMaskGen1:

    ACTION_DEFAULT = -2
    ACTION_SPACE = 10
    ACTION_MOVE_RANGE = range(6, 10)
    ACTION_SWITCH_RANGE = range(0, 6)

    def __init__(self):
        self.mask = np.zeros(self.ACTION_SPACE, dtype=bool)

    def get_mask(self) -> np.ndarray:
        return self.mask

    def set(self, mask: list[int]):
        self.mask = np.array(mask, dtype=bool)

    def reset(self):
        self.mask = np.zeros(self.ACTION_SPACE, dtype=bool)

    def describe(self, battle=None) -> str:
        lines = []

        # --- Switches ---
        switch_lines = []
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

        lines.append("--- Switches ---")
        lines.extend(switch_lines if switch_lines else ["(none)"])

        lines.append("--- Moves ---")
        lines.extend(move_lines if move_lines else ["(none)"])

        # --- Summary ---
        valid_actions = np.where(self.mask)[0]
        lines.append(f"\nValid actions: {valid_actions.tolist()}")
        return "\n".join(lines)
