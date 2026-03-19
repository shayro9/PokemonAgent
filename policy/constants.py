"""
Observation and action-space layout constants shared across the policy package.

Observation layout (flat vector from BattleState.to_array()):
  [ctx_before | my_moves (4 x MOVE_EMBED_LEN) | opp_moves (4 x MOVE_EMBED_LEN) | ctx_after]

Action space layout (matches SinglesEnvWrapper):
  0-5   : non-move actions (switches, etc.)
  6-9   : move slots 0-3   ← pointer head covers these
  10-25 : remaining non-move actions
"""

from env.states.state_utils import MAX_MOVES
from env.states.move_state import MoveState
from env.states.gen1.battle_state_gen_1 import BattleStateGen1

# ── Observation slicing ────────────────────────────────────────────────────
MOVE_LEN: int = MoveState.array_len()

# Everything in the observation except my_moves
CONTEXT_LEN: int = BattleStateGen1.battle_context_len()

# ── Action space ───────────────────────────────────────────────────────────
MOVE_ACTION_START: int = 6
N_MOVE_ACTIONS: int = MAX_MOVES
TOTAL_ACTIONS: int = 26

