# Code Review — PokemonAgent
> Reviewed: 2026-03-30 | All 642 existing tests pass baseline.

---

## Executive Summary

Well-structured Gen 1 Pokemon Showdown RL training system using `AttentionPointerPolicy` (custom MaskablePPO with dual pointer heads for moves and switches). The state-layer architecture and test coverage for state classes are strong. Three blockers need attention before this is production-safe, particularly the mutable extractor state and the dead `other_action_head` layer.

Focus areas requested: **clean code / structure**, **core functionality tested**, **GPU utilization**, **scalability to future gens**.

---

## 🔴 Blockers

---

### 1. `other_action_head` produces 0 outputs — dead layer + misleading constants

**Files:** `policy/constants.py:41`, `policy/policy.py:89,94`

```python
# constants.py
TOTAL_ACTIONS: int = 10        # correct for current env (0-9)

# policy.py
n_other_action = TOTAL_ACTIONS - N_MOVE_ACTIONS - N_SWITCH_ACTIONS  # 16 (10-25)
# → 10 - 4 - 6 = 0  ← comment is completely wrong

self.other_action_head = nn.Linear(trunk_hidden, n_other_action)  # nn.Linear(256, 0)
```

**Why:** `nn.Linear(trunk_hidden, 0)` produces a `(B, 0)` tensor that is silently concatenated and contributes nothing. The comment `# 16 (10-25)` and the entire header block in `constants.py` describe a *26-action* space that doesn't exist in the env (`Discrete(10)`). This confusion bleeds into `tests/env/test_singles_env_wrapper.py:29` which hardcodes `Discrete(26)` with the comment `# Real action space` — the opposite of true. Tests are validating against a ghost action space.

**Suggested fix (pick one):**
- If the intent is 10 actions: remove `other_action_head`, remove it from `_build_logits`, and fix all comments in `constants.py`.
- If the intent is 26 actions: update `singles_env_wrapper.py` to `Discrete(26)`, `ActionMaskGen1.ACTION_SPACE = 26`, and `TOTAL_ACTIONS = 26`.

---

### 2. Wrong import from `sympy` in `action_mask_gen_1.py`

**File:** `env/action_mask_gen_1.py:2`

```python
from sympy.codegen.abstract_nodes import List   # sympy = Computer Algebra System
```

`sympy` is not in `requirements.txt` and `List` is never referenced anywhere in the file. This will raise `ImportError` in any standard install (Colab, CI, fresh venv). The type hint already uses lowercase `list[int]` correctly.

**Suggested fix:** Delete the import entirely.

---

### 3. `AttentionPointerExtractor` stores forward-pass tensors as mutable instance state

**File:** `policy/extractor.py:72-75`, `:150-153`

```python
# Declared as instance attributes:
self.move_hidden: Optional[torch.Tensor] = None   # (B, 4, move_hidden)
self.team_hidden: Optional[torch.Tensor] = None   # (B, 6, team_hidden)

# Written during forward(), read by policy._build_logits() immediately after:
self.move_hidden = move_hidden
self.team_hidden = team_hidden
```

**Why this is a problem:**
- **`nn.DataParallel` breaks silently**: each GPU replica writes its own `self.move_hidden`; the policy reads from the *main* module replica — stale data from the wrong batch.
- **Re-entrant calls corrupt state**: two calls to `forward()` before `_build_logits()` reads will silently overwrite the first batch's tensors.
- **Impure API contract**: `nn.Module.forward()` should be a pure function — no side effects on `self`.

**Suggested fix:** Return hidden states directly from `forward()` via a dataclass:

```python
from dataclasses import dataclass

@dataclass
class ExtractorOutput:
    features: torch.Tensor      # (B, trunk_hidden)
    move_hidden: torch.Tensor   # (B, 4, move_hidden)
    team_hidden: torch.Tensor   # (B, 6, team_hidden)

def forward(self, obs: torch.Tensor) -> ExtractorOutput:
    ...
    return ExtractorOutput(features=features, move_hidden=move_hidden, team_hidden=team_hidden)
```

Then `_build_logits(self, features, move_hidden, team_hidden)` takes them as arguments instead of reading `self.mlp_extractor.move_hidden`.

---

## 🟡 Suggestions

---

### 4. `BattleStateGen1._buf` is a class-level shared mutable array (thread-unsafe)

**File:** `env/states/gen1/battle_state_gen_1.py:32-50`, `:93-107`

```python
_buf: np.ndarray | None = None   # shared across ALL instances of the class
```

Two concurrent `to_array()` calls race to write the same buffer. The `.copy()` at the end guards the *reader*, but the write phase is unguarded. Safe in single-threaded training but will silently corrupt observations in any vectorised-env setup.

**Suggested fix:** Move buffer creation to `__init__` (instance-level), or document clearly that this class is not thread-safe and add a `threading.Lock`.

---

### 5. `alive_vector` position assumption in extractor is silently fragile

**File:** `policy/extractor.py:92-93`

```python
my_pokemon_vector = obs[:, ARENA_OPPONENT_LEN:-MAX_TEAM_SIZE]   # assumes alive_vector is last
alive_vector      = obs[:, -MAX_TEAM_SIZE:]
```

This only works because `TeamState.to_array()` appends `alive_vector` last. There is no assertion, no contract, and no test that enforces this ordering. Adding any new field after `alive_vector` in `TeamState` would silently corrupt every observation.

**Suggested fix:** Add a layout assertion in `BattleStateGen1` or in the extractor's `__init__`:

```python
# In AttentionPointerExtractor.__init__
expected = ARENA_OPPONENT_LEN + MY_POKEMON_LEN * MAX_TEAM_SIZE + MAX_TEAM_SIZE
assert expected == obs_dim, (
    f"Observation layout mismatch: extractor expects {expected}, got {obs_dim}. "
    "Update slicing constants if TeamState layout changed."
)
```

---

### 6. `pull_attribute` silently swallows all exceptions

**File:** `env/states/state_utils.py:88-95`

```python
except Exception as e:
    return type_value(default_value)   # 'e' is captured but never logged
```

Any bug surfacing through `pull_attribute` (wrong attribute name, type mismatch, etc.) returns a silent default. This makes training regressions caused by missing data very hard to diagnose.

**Suggested fix:** At minimum, log a warning:

```python
except (AttributeError, TypeError, ValueError) as e:
    import logging
    logging.warning("pull_attribute(%s, %s): %s — returning default", obj, key, e)
    return type_value(default_value)
```

---

### 7. Wildcard imports in `training/train.py`

**File:** `training/train.py:14-15`

```python
from config.config import *
from training.battle_metrics_log import *
```

These make it impossible to know what names are in scope without reading both files, break IDE navigation/refactoring, and increase the chance of silent name shadowing. Especially risky near `LR`, `GAMMA`, `LOG_FREQ` constants.

**Suggested fix:** Make all imports explicit.

---

### 8. Missing tests for critical RL components

The state-layer tests are excellent. These components have **no dedicated coverage**:

| Component | Risk if broken |
|---|---|
| `env/reward.py` — `get_state_value()` | Wrong sign silently degrades training; asymmetric WIN/LOSS bonuses; boost weights are tuned magic numbers |
| `env/action_mask_gen_1.py` — `ActionMaskGen1` | Invalid mask = illegal actions sent to env |
| `policy/attention.py` — `CrossAttention` | Core architecture + NaN guard path untested |
| `policy/policy.py` — `AttentionPointerPolicy.forward()` | Policy's own logit assembly untested end-to-end |
| `policy/device_manager.py` — CPU fallback | `get_device("auto")` when CUDA absent |

The reward function deserves priority: `WIN_BONUS=15`, `LOSS_PENALTY=-10` (asymmetric), `FRZ/SLP=1.2`, `OPP_BOOST_PENALTIES['spe']=1.5` — a sign flip in `_calculate_boost_value` would reward crippling your own Pokémon.

---

### 9. Scalability: gen-specific classes hardcoded throughout the policy pipeline

**Files:** `policy/extractor.py:23-24`, `policy/constants.py:20-24`, `env/singles_env_wrapper.py:9,47`

All three files directly import `BattleStateGen1`, `MyPokemonStateGen1`, etc. Adding Gen 2 requires forking or patching multiple files.

The subclassing infrastructure is already good (`STAT_KEYS`, `BOOST_KEYS`, `TRACKED_EFFECTS` as class vars, abstract `to_array()`/`array_len()`). The missing piece is a top-level config that threads the right classes through:

```python
@dataclass
class BattleConfig:
    battle_state_cls: type     # BattleStateGen1, BattleStateGen2, ...
    arena_opponent_len: int    # derived from battle_state_cls.battle_before_me_len()
    my_pokemon_len: int        # derived from my_pokemon_state_cls.array_len()
    my_moves_start: int
    gen: int
```

Pass this into `PokemonRLWrapper`, `AttentionPointerExtractor`, and `build_env`. The constants in `policy/constants.py` would then be derived from the config rather than from hardcoded Gen 1 class imports.

---

### 10. `print()` statements in library utility code

**File:** `policy/device_manager.py:33,36`

```python
print(f"✓ CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
print("⊘ CUDA not available - using CPU")
```

Side effects in a utility function make it impossible to suppress output (tests, notebooks, multi-env instantiation). 

**Suggested fix:** Use `logging.info()` so callers can configure verbosity with standard Python logging.

---

## 💭 Nits

---

### 11. `MY_BENCH_LEN` has a wrong comment and is unused

**File:** `policy/constants.py:33`

```python
MY_BENCH_LEN: int = MY_POKEMON_LEN * 6 + 6  # 162 * 5 = 810 (5 bench slots × pokemon+moves)
```

Comment says `×5 = 810`, code does `×6 + 6`. Constant is never referenced elsewhere. Consider removing or correcting.

---

### 12. `DeviceConfig` stores both the raw input and resolved device string redundantly

**File:** `training/device_config.py:17-18`

```python
self.device_str = get_device(device)   # resolved: "cuda" or "cpu"
self.device = device                   # raw input: "auto", "cuda", or "cpu"
```

`self.device` (the raw input) is never read after construction. `__str__` returns `device_str`. Drop `self.device` or rename for clarity.

---

### 13. `algo` / `use_action_masking` hardcoded in two places with stale TODO

**Files:** `training/train.py:75-76`, `training/evaluation.py:111-112`

```python
# TODO: add from args
algo = "maskable_ppo"
use_action_masking = (algo == "maskable_ppo")
```

Same TODO duplicated in two files. Since the codebase is PPO-only, either promote `use_action_masking` to a `train_model()` parameter or just inline `True` and remove the indirection.

---

### 14. `_MOVE_STATE_CACHE` is an unbounded module-level dict

**File:** `env/states/pokemon_state.py:16`

```python
_MOVE_STATE_CACHE: dict[tuple, MoveState] = {}
```

For long training runs across diverse team pools, this grows without bound. The cache also persists across test runs, which can cause test pollution if a test relies on a freshly-constructed `MoveState`.

**Suggested fix:** Use `functools.lru_cache` with a bounded `maxsize` on `_get_cached_move_state`, or document the unbounded growth as an accepted trade-off.

---

## ✅ What's Working Well

- **Pointer head architecture**: clean and correct — `bias=True` fix applied, equivariance preserved via shared-weight encoders, `nan_to_num` guards the all-masked softmax path.
- **State class hierarchy**: well-designed for subclassing (`STAT_KEYS`, `BOOST_KEYS`, `TRACKED_EFFECTS` as class vars, abstract `to_array()`/`array_len()` with runtime assertions).
- **`_slice_observation`**: vectorised `gather` + `masked_fill` — efficient, no Python loops.
- **Test coverage on state classes**: excellent — shape assertions, dtype checks, NaN/Inf guards, boundary cases for fainted/empty slots across all state files.
- **`EvalResult.win_rate`**: correctly includes draws as `draws / 2`.
- **`CrossAttention`**: clean einsum-based implementation, proper `key_padding_mask` support.
- **`build_mlp`**: `LayerNorm → ReLU` is a solid default for RL feature extraction.
- **Lazy `_init_buffer`** in `BattleStateGen1`: good pattern for deferring expensive setup without requiring explicit initialization by callers.

---

## GPU Utilization Assessment

**Current state: Good for single-GPU SB3 training.**

- ✅ `get_device("auto")` with CUDA detection in `device_manager.py`
- ✅ `device=str(device_config)` threaded through to `MaskablePPO` — SB3 handles `.to(device)` for all model parameters
- ✅ `torch.as_tensor(obs_tensor, device=self.device)` in `extract_features` — avoids unnecessary CPU round-trips
- ✅ All `CrossAttention` and `build_mlp` operations are pure PyTorch — GPU-compatible
- ⚠️ Blocker #3 above (mutable extractor state) would break `nn.DataParallel` multi-GPU if ever needed
- ℹ️ Env-side observation construction (`BattleStateGen1`, `TeamState`) is CPU/NumPy — expected and correct for gym-style envs

---

## Priority Order for Fixes

1. 🔴 **Fix or remove `sympy` import** (2 min, prevents ImportError)
2. 🔴 **Resolve TOTAL_ACTIONS=10 vs 26 confusion** (clarifies intent for everything else)
3. 🟡 **Add reward function tests** (highest training-quality risk)
4. 🔴 **Refactor extractor to return tensors rather than storing on `self`** (correctness for any multi-GPU future)
5. 🟡 **Add layout assertion for alive_vector position** (prevents silent corruption)
6. 🟡 **Fix wildcard imports** (maintainability)
7. 🟡 **Add `BattleConfig` abstraction** (unlocks future gens cleanly)
