"""
Observation and action-space layout constants shared across the policy package.

Observation layout (flat vector from BattleState.to_array()):
  [arena (5) | my_active (22+140) | opp_active (25) | my_bench (5×162) | opp_bench (5×25) | opp_moves (140) | my_moves (140)]
  
  where:
  - my_active = pokemon(22) + 4 moves(140)
  - my_bench entry = pokemon(22) + 4 moves(140) = 162 dims per slot
  - CONTEXT = arena + my_active + opp_active + my_bench + opp_bench + opp_moves
  - MY_MOVES = my_moves (active pokemon's available moves)

Action space layout (matches SinglesEnvWrapper):
  0-5   : switch actions (select bench slot 0-5)
  6-9   : move slots 0-3   ← move pointer head scores these
  10-25 : remaining non-move actions
"""

from env.states.state_utils import MAX_MOVES
from env.states.move_state import MoveState
from env.states.gen1.battle_state_gen_1 import BattleStateGen1
from env.states.gen1.my_pokemon_state_gen_1 import MyPokemonStateGen1
from env.states.gen1.opponent_pokemon_state_gen_1 import OpponentPokemonStateGen1

# ── Observation slicing ────────────────────────────────────────────────────
MOVE_LEN: int = MoveState.array_len()

# Everything in the observation except my_moves
CONTEXT_LEN: int = BattleStateGen1.battle_context_len()

# Slicing constants for observation layout
ARENA_LEN: int = 5
ARENA_START: int = 0

MY_ACTIVE_LEN: int = MyPokemonStateGen1.array_len() + MAX_MOVES * MOVE_LEN  # 22 + 140 = 162
MY_ACTIVE_START: int = ARENA_START + ARENA_LEN

OPP_ACTIVE_LEN: int = OpponentPokemonStateGen1.array_len()
OPP_ACTIVE_START: int = MY_ACTIVE_START + MY_ACTIVE_LEN

MY_BENCH_LEN: int = MY_ACTIVE_LEN * 5  # 162 * 5 = 810 (5 bench slots × pokemon+moves)
MY_BENCH_START: int = OPP_ACTIVE_START + OPP_ACTIVE_LEN

OPP_BENCH_LEN: int = OPP_ACTIVE_LEN * 5 + 5  # 5 bench slots + alive_vector (no moves for opponent)
OPP_BENCH_START: int = MY_BENCH_START + MY_BENCH_LEN

# ── Action space ───────────────────────────────────────────────────────────
MOVE_ACTION_START: int = 6
N_MOVE_ACTIONS: int = MAX_MOVES
SWITCH_ACTION_START: int = 0
N_SWITCH_ACTIONS: int = 6
TOTAL_ACTIONS: int = 26

