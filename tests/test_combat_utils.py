"""Unit tests for ``env.combat_utils.did_no_damage``.

This file documents and verifies that damage detection:
- returns ``False`` when no actionable damaging move is available,
- handles missing historical HP values safely,
- and reports no-damage outcomes when opponent HP does not decrease.
"""

from types import SimpleNamespace

from poke_env.battle import MoveCategory

from env.combat_utils import did_no_damage


def _make_battle(species: str, current_hp_fraction: float):
    """Create a minimal battle stub with the opponent active pokemon."""
    opponent = SimpleNamespace(species=species, current_hp_fraction=current_hp_fraction)
    return SimpleNamespace(opponent_active_pokemon=opponent)


def _make_move(category: MoveCategory):
    """Create a minimal move stub exposing only the category field."""
    return SimpleNamespace(category=category)


def test_did_no_damage_returns_false_when_last_move_is_none():
    """It should return False if there is no previous move to evaluate."""
    battle = _make_battle("pikachu", 1.0)
    assert did_no_damage(battle, {"opp_pikachu": 1.0}, None) is False


def test_did_no_damage_returns_false_for_status_moves():
    """It should ignore status-category moves for damage checks."""
    battle = _make_battle("pikachu", 1.0)
    status_move = _make_move(MoveCategory.STATUS)
    assert did_no_damage(battle, {"opp_pikachu": 1.0}, status_move) is False


def test_did_no_damage_returns_false_when_previous_hp_is_missing():
    """It should return False when there is no tracked previous HP for the target."""
    battle = _make_battle("pikachu", 1.0)
    damaging_move = _make_move(MoveCategory.SPECIAL)
    assert did_no_damage(battle, {}, damaging_move) is False


def test_did_no_damage_returns_false_when_opponent_hp_decreases():
    """It should return False when current HP is lower than the previous value."""
    battle = _make_battle("pikachu", 0.4)
    damaging_move = _make_move(MoveCategory.PHYSICAL)
    assert did_no_damage(battle, {"opp_pikachu": 0.7}, damaging_move) is False


def test_did_no_damage_returns_true_when_opponent_hp_is_unchanged():
    """It should return True when current HP is equal to the previous value."""
    battle = _make_battle("pikachu", 0.7)
    damaging_move = _make_move(MoveCategory.PHYSICAL)
    assert did_no_damage(battle, {"opp_pikachu": 0.7}, damaging_move) is True


def test_did_no_damage_returns_true_when_opponent_hp_increases():
    """It should return True when current HP is greater than the previous value."""
    battle = _make_battle("pikachu", 0.8)
    damaging_move = _make_move(MoveCategory.SPECIAL)
    assert did_no_damage(battle, {"opp_pikachu": 0.5}, damaging_move) is True
