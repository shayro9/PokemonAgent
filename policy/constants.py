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

# ── Observation slicing ────────────────────────────────────────────────────
MOVE_LEN: int = MoveState.array_len()

# Everything in the observation except my_moves
ARENA_OPPONENT_LEN: int = BattleStateGen1.battle_before_me_len()

# Slicing constants for observation layout
MY_POKEMON_LEN: int  = MyPokemonStateGen1.array_len() # 22 + 140 = 162
CONTEXT_LEN: int = ARENA_OPPONENT_LEN + MY_POKEMON_LEN
MY_MOVES_START = MY_POKEMON_LEN - MOVE_LEN * MAX_MOVES
# ── Action space ───────────────────────────────────────────────────────────
N_SWITCH_ACTIONS: int = 6

