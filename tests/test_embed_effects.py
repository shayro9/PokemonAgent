import numpy as np
import pytest
from poke_env.battle.effect import Effect

from env.embed import TRACKED_EFFECTS, embed_effects


def test_output_shape():
    result = embed_effects({})
    assert result.shape == (len(TRACKED_EFFECTS),)


def test_empty_effects():
    result = embed_effects({})
    assert np.all(result == 0)


def test_single_effect_present():
    effect = TRACKED_EFFECTS[0]
    result = embed_effects({effect})
    assert result[0] == 1
    assert np.sum(result) == 1


def test_all_effects_present():
    result = embed_effects(set(TRACKED_EFFECTS))
    assert np.all(result == 1)


def test_untracked_effect_ignored():
    # Find an Effect not in TRACKED_EFFECTS
    untracked = next(e for e in Effect if e not in TRACKED_EFFECTS)
    result = embed_effects({untracked})
    assert np.all(result == 0)


def test_multiple_tracked_effects():
    n = min(2, len(TRACKED_EFFECTS))  # or just hardcode 2
    tracked_subset = set(TRACKED_EFFECTS[:n])
    result = embed_effects(tracked_subset)
    assert np.sum(result) == n
    assert all(result[i] == 1 for i in range(n))


def test_output_dtype():
    result = embed_effects({})
    assert result.dtype in [np.int32, np.int64, np.float64]


def test_mixed_tracked_and_untracked():
    untracked = next(e for e in Effect if e not in TRACKED_EFFECTS)
    effects = {TRACKED_EFFECTS[0], untracked}
    result = embed_effects(effects)
    assert np.sum(result) == 1
    assert result[0] == 1
