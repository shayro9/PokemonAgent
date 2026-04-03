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

To support additional generations, pass a BattleConfig to AttentionPointerExtractor and
PokemonRLWrapper directly.  The module-level constants below remain as Gen 1 defaults
for code that has not yet been updated to accept a BattleConfig.
"""

from env.states.state_utils import MAX_MOVES  # re-exported for callers that import it here
from env.battle_config import BattleConfig

# ── Canonical Gen 1 config ─────────────────────────────────────────────────
GEN1_BATTLE_CONFIG: BattleConfig = BattleConfig.gen1()

# ── Gen 1 observation slicing constants (derived from GEN1_BATTLE_CONFIG) ──
# Kept for backward compatibility — prefer BattleConfig properties in new code.
MOVE_LEN: int          = GEN1_BATTLE_CONFIG.move_len
ARENA_OPPONENT_LEN: int = GEN1_BATTLE_CONFIG.arena_opponent_len
MY_POKEMON_LEN: int    = GEN1_BATTLE_CONFIG.my_pokemon_len
CONTEXT_LEN: int       = GEN1_BATTLE_CONFIG.context_len
MY_MOVES_START: int    = GEN1_BATTLE_CONFIG.my_moves_start
# ── Action space ───────────────────────────────────────────────────────────
N_SWITCH_ACTIONS: int  = GEN1_BATTLE_CONFIG.n_switch_actions

