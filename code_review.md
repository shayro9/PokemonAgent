# Code Review — PokemonAgent
> Reviewed: 2026-03-30 | All 642 existing tests pass baseline.

---

## Executive Summary

Well-structured Gen 1 Pokemon Showdown RL training system using `AttentionPointerPolicy` (custom MaskablePPO with dual pointer heads for moves and switches). The state-layer architecture and test coverage for state classes are strong. Three blockers need attention before this is production-safe, particularly the mutable extractor state and the dead `other_action_head` layer.

Focus areas requested: **clean code / structure**, **core functionality tested**, **GPU utilization**, **scalability to future gens**.

---

## 🔴 Blockers

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

## 💭 Nits

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

3. 🟡 **Add reward function tests** (highest training-quality risk)
4. 🔴 **Refactor extractor to return tensors rather than storing on `self`** (correctness for any multi-GPU future)
7. 🟡 **Add `BattleConfig` abstraction** (unlocks future gens cleanly)
