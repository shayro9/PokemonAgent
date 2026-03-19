"""
Observation and action-space layout constants shared across the policy package.

Observation layout (flat vector from BattleState.to_array()):
  [ctx_before | my_moves (4 x MOVE_EMBED_LEN) | opp_moves (4 x MOVE_EMBED_LEN) | ctx_after]

Action space layout (matches SinglesEnvWrapper):
  0-5   : non-move actions (switches, etc.)
  6-9   : move slots 0-3   ← pointer head covers these
  10-25 : remaining non-move actions
"""

from env.embed import MAX_MOVES, MOVE_EMBED_LEN
from env.battle_state import CONTEXT_BEFORE_MY_MOVES, CONTEXT_AFTER_OPP_MOVES

# ── Observation slicing ────────────────────────────────────────────────────
MY_MOVES_LEN: int = MAX_MOVES * MOVE_EMBED_LEN          # 152
OPP_MOVES_START: int = CONTEXT_BEFORE_MY_MOVES + MY_MOVES_LEN
OPP_MOVES_LEN: int = MAX_MOVES * MOVE_EMBED_LEN          # 152

# Everything in the observation except my_moves
CONTEXT_DIM: int = CONTEXT_BEFORE_MY_MOVES + OPP_MOVES_LEN + CONTEXT_AFTER_OPP_MOVES

# ── Action space ───────────────────────────────────────────────────────────
MOVE_ACTION_START: int = 6
N_MOVE_ACTIONS: int = 4
TOTAL_ACTIONS: int = 26

