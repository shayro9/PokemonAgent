from __future__ import annotations

import numpy as np
from poke_env.battle.side_condition import SideCondition

# Side conditions tracked in ArenaState
# In Gen 1, Reflect and Light Screen exist but work differently (no turn limit)
GEN1_TRACKED_SCREENS = [
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
]

TURN_NORM = 30.0  # normalize turn number


class ArenaState:
    """
    Encodes the global arena state shared by both sides of the field.

    Contains:
        - turn number (normalized)
        - Reflect active on my side
        - Light Screen active on my side
        - Reflect active on opponent's side
        - Light Screen active on opponent's side

    Usage:
        ArenaState(battle)   →  from a poke-env battle object
        ArenaState()         →  empty / all zeros

    Subclass and override SIDE_CONDITIONS to track different conditions per gen.
    """

    TRACKED_SCREENS = GEN1_TRACKED_SCREENS

    def __init__(self, battle=None):
        if battle is not None:
            self._load_from_battle(battle)
        else:
            self._load_empty()

    def _load_from_battle(self, battle):
        self.turn        = battle.turn
        self.my_screens  = self._encode_screens(battle.side_conditions)
        self.opp_screens = self._encode_screens(battle.opponent_side_conditions)

    def _load_empty(self):
        self.turn        = 0
        self.my_screens  = np.zeros(len(self.TRACKED_SCREENS), dtype=np.float32)
        self.opp_screens = np.zeros(len(self.TRACKED_SCREENS), dtype=np.float32)

    # ------------------------------------------------------------------
    # encoding helpers
    # ------------------------------------------------------------------
    def _encode_screens(self, side_conditions: dict) -> np.ndarray:
        """Binary flags for each tracked side condition."""
        return np.array(
            [1.0 if sc in side_conditions else 0.0 for sc in self.TRACKED_SCREENS],
            dtype=np.float32,
        )

    def turn_encoded(self) -> float:
        """Normalize raw turn number → [0, 1]."""
        return min(self.turn / TURN_NORM, 1.0)

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def to_array(self) -> np.ndarray:
        """
        Flat encoded array: [turn_encoded, *my_screens, *opp_screens]
        """
        return np.concatenate([
            [self.turn_encoded()],
            self.my_screens,
            self.opp_screens,
        ]).astype(np.float32)

    @classmethod
    def array_len(cls) -> int:
        turn = 1
        return turn + len(cls.TRACKED_SCREENS) * 2

    def describe(self) -> str:
        """Human-readable breakdown of the arena state. Useful for debugging."""
        screen_names = [sc.name for sc in self.TRACKED_SCREENS]
        my_active    = [screen_names[i] for i, v in enumerate(self.my_screens)  if v == 1.0]
        opp_active   = [screen_names[i] for i, v in enumerate(self.opp_screens) if v == 1.0]

        lines = [
            f"Turn          : {self.turn} (encoded: {self.turn_encoded():.3f})",
            f"My  screens   : {my_active  if my_active  else 'none'}",
            f"Opp screens   : {opp_active if opp_active else 'none'}",
            f"Array length  : {self.array_len()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()




