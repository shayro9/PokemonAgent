"""Unit tests for ``env.embed.calc_types_vector``.

This file documents and verifies that type interaction vectors:
- always contain 4 values (2 attacker slots x 2 defender slots),
- map immunities to ``-1.0``,
- and use zeros for missing type slots.
"""

import numpy as np
from poke_env.battle.pokemon_type import PokemonType

from env.embed import calc_types_vector


def test_calc_types_vector_has_fixed_four_element_shape():
    """It should always return 4 matchup features regardless of single typing."""
    result = calc_types_vector([PokemonType.FIRE], [PokemonType.GRASS], gen=9)
    assert result.shape == (4,)


def test_calc_types_vector_encodes_immunity_as_negative_one():
    """It should map damage multiplier 0 to -1.0 in log2 space."""
    result = calc_types_vector([PokemonType.GROUND], [PokemonType.FLYING], gen=9)
    assert float(result[0]) == -1.0


def test_calc_types_vector_uses_zero_for_missing_type_slots():
    """It should pad missing attacker/defender secondary slots with 0.0."""
    result = calc_types_vector([PokemonType.WATER], [PokemonType.FIRE], gen=9)
    assert np.allclose(result[1:], 0.0)
