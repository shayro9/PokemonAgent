"""Unit tests for ``env.embed.embed_effects``.

This file documents and verifies that effect embeddings:
- produce the expected vector shape,
- flag tracked effects as ``1``,
- ignore untracked effects,
- and return an integer-encoded vector.
"""

import numpy as np
from poke_env.battle.effect import Effect

from env.embed import TRACKED_EFFECTS, embed_effects


def test_embed_effects_output_shape_matches_tracked_effect_count():
    """It should return one slot per configured tracked effect."""
    result = embed_effects(set())
    assert result.shape == (len(TRACKED_EFFECTS),)


def test_embed_effects_empty_input_returns_all_zeros():
    """It should encode no active effects as an all-zero vector."""
    result = embed_effects(set())
    assert np.all(result == 0)


def test_embed_effects_single_tracked_effect_sets_only_its_slot():
    """It should set exactly one entry when one tracked effect is present."""
    selected = TRACKED_EFFECTS[0]
    result = embed_effects({selected})
    assert result[0] == 1
    assert np.sum(result) == 1


def test_embed_effects_all_tracked_effects_set_all_slots():
    """It should set all entries to one when all tracked effects are active."""
    result = embed_effects(set(TRACKED_EFFECTS))
    assert np.all(result == 1)


def test_embed_effects_untracked_effect_is_ignored():
    """It should ignore effects that are not in ``TRACKED_EFFECTS``."""
    untracked = next(effect for effect in Effect if effect not in TRACKED_EFFECTS)
    result = embed_effects({untracked})
    assert np.all(result == 0)


def test_embed_effects_mixed_input_sets_only_tracked_members():
    """It should set entries for tracked effects even if untracked effects are present too."""
    untracked = next(effect for effect in Effect if effect not in TRACKED_EFFECTS)
    result = embed_effects({TRACKED_EFFECTS[0], untracked})
    assert np.sum(result) == 1
    assert result[0] == 1


def test_embed_effects_returns_integer_vector():
    """It should return an integer dtype vector to represent 0/1 flags."""
    result = embed_effects(set())
    assert np.issubdtype(result.dtype, np.integer)
