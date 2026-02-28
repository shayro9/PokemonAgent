"""Unit tests for ``env.embed.embed_status``.

This file documents and verifies that status embeddings are valid one-hot vectors
over the configured move statuses.
"""

import numpy as np

from env.embed import MOVE_STATUSES, embed_status


def test_embed_status_output_length_matches_known_statuses():
    """It should allocate one position per known status value."""
    result = embed_status(None)
    assert result.shape == (len(MOVE_STATUSES),)


def test_embed_status_known_status_is_one_hot():
    """It should set exactly one index when the input status is known."""
    status = MOVE_STATUSES[0]
    result = embed_status(status)
    assert np.sum(result) == 1
    assert result[0] == 1


def test_embed_status_unknown_status_is_all_zeros():
    """It should produce all zeros when the status does not match any known value."""
    result = embed_status(object())
    assert np.all(result == 0)
