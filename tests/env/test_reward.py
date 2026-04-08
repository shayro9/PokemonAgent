"""
Tests for env/reward.py — get_state_value() and _calculate_boost_value().

Priority: highest (a sign flip in _calculate_boost_value would silently reward
crippling your own Pokémon and penalise crippling the opponent).

Covers:
  1. _calculate_boost_value — sign correctness for own vs opponent boosts
  2. get_state_value — HP contributions (own positive, opponent negative)
  3. get_state_value — faint bonus/penalty
  4. get_state_value — status condition weights
  5. get_state_value — boost rewards (own positive boost → reward)
  6. get_state_value — opponent boost penalties (opponent -spe → reward)
  7. get_state_value — WIN_BONUS / LOSS_PENALTY asymmetry
  8. get_state_value — missing team members padding
  9. get_state_value — additive combination of all components
"""

import pytest
from unittest.mock import MagicMock
from poke_env.battle import Status

from env.reward import (
    get_state_value,
    _calculate_boost_value,
    HP_VALUE,
    FAINTED_VALUE,
    WIN_BONUS,
    LOSS_PENALTY,
    OWN_BOOST_VALUES,
    OPP_BOOST_PENALTIES,
    _STATUS_WEIGHTS,
)
from env.states.gen1.battle_state_gen_1 import MAX_TEAM_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pokemon(
    hp_fraction: float = 1.0,
    fainted: bool = False,
    status=None,
    boosts: dict | None = None,
) -> MagicMock:
    mon = MagicMock()
    mon.current_hp_fraction = hp_fraction
    mon.fainted = fainted
    mon.status = status
    mon.boosts = boosts or {}
    return mon


def make_battle(
    own_team: list | None = None,
    opp_team: list | None = None,
    won: bool = False,
    lost: bool = False,
) -> MagicMock:
    battle = MagicMock()
    battle.won = won
    battle.lost = lost
    battle.team = {f"mon_{i}": m for i, m in enumerate(own_team or [])}
    battle.opponent_team = {f"opp_{i}": m for i, m in enumerate(opp_team or [])}
    return battle


# ---------------------------------------------------------------------------
# 1. _calculate_boost_value — sign correctness
# ---------------------------------------------------------------------------

class TestCalculateBoostValue:

    def test_own_positive_boost_is_positive(self):
        """Positive boost on own mon (+2 atk) → positive reward."""
        value = _calculate_boost_value({'atk': 2}, OWN_BOOST_VALUES, negate=False)
        assert value == pytest.approx(2 * OWN_BOOST_VALUES['atk'])
        assert value > 0

    def test_own_negative_boost_is_negative(self):
        """Negative boost on own mon (-2 atk) → negative reward (bad for us)."""
        value = _calculate_boost_value({'atk': -2}, OWN_BOOST_VALUES, negate=False)
        assert value == pytest.approx(-2 * OWN_BOOST_VALUES['atk'])
        assert value < 0

    def test_opp_negative_boost_is_positive(self):
        """Negative boost on opponent (-2 spe) → positive reward (we crippled them)."""
        value = _calculate_boost_value({'spe': -2}, OPP_BOOST_PENALTIES, negate=True)
        # negate=True: value += -(-2) * weight = +2 * weight
        assert value == pytest.approx(2 * OPP_BOOST_PENALTIES['spe'])
        assert value > 0

    def test_opp_positive_boost_is_negative(self):
        """Positive boost on opponent (+2 spe) → negative reward (they boosted)."""
        value = _calculate_boost_value({'spe': 2}, OPP_BOOST_PENALTIES, negate=True)
        # negate=True: value += -(+2) * weight = -2 * weight
        assert value == pytest.approx(-2 * OPP_BOOST_PENALTIES['spe'])
        assert value < 0

    def test_zero_boost_is_zero(self):
        """Zero boost contributes nothing."""
        value = _calculate_boost_value({'atk': 0, 'spe': 0}, OWN_BOOST_VALUES, negate=False)
        assert value == pytest.approx(0.0)

    def test_empty_boosts_is_zero(self):
        assert _calculate_boost_value({}, OWN_BOOST_VALUES, negate=False) == pytest.approx(0.0)

    def test_unknown_stat_ignored(self):
        """Stats not in boost_values dict are silently ignored."""
        value = _calculate_boost_value({'nonexistent': 5}, OWN_BOOST_VALUES, negate=False)
        assert value == pytest.approx(0.0)

    def test_multiple_stats_sum(self):
        """Multiple stats are summed independently."""
        boosts = {'atk': 1, 'spe': 2}
        expected = 1 * OWN_BOOST_VALUES['atk'] + 2 * OWN_BOOST_VALUES['spe']
        assert _calculate_boost_value(boosts, OWN_BOOST_VALUES, negate=False) == pytest.approx(expected)

    def test_spe_has_highest_opp_penalty_weight(self):
        """-spe on opponent is valued higher than -atk (1.5 vs 1.2)."""
        spe_val = _calculate_boost_value({'spe': -1}, OPP_BOOST_PENALTIES, negate=True)
        atk_val = _calculate_boost_value({'atk': -1}, OPP_BOOST_PENALTIES, negate=True)
        assert spe_val > atk_val


# ---------------------------------------------------------------------------
# 2. HP contributions
# ---------------------------------------------------------------------------

class TestHPContributions:

    def test_full_hp_own_mon_adds_hp_value(self):
        mon = make_pokemon(hp_fraction=1.0)
        battle = make_battle(own_team=[mon], opp_team=[])
        # own: +1.0 * HP_VALUE; padding for missing own members; opp: padding for missing opp
        value = get_state_value(battle)
        own_hp = 1.0 * HP_VALUE
        own_padding = (MAX_TEAM_SIZE - 1) * HP_VALUE  # 5 missing own slots
        opp_padding = -(MAX_TEAM_SIZE - 0) * HP_VALUE  # 6 missing opp slots
        assert value == pytest.approx(own_hp + own_padding + opp_padding)

    def test_damaged_own_mon_reduces_value(self):
        """Half-HP own mon is worth less than full-HP own mon."""
        full = make_battle(own_team=[make_pokemon(0.5)], opp_team=[])
        val_half = get_state_value(full)
        full2 = make_battle(own_team=[make_pokemon(1.0)], opp_team=[])
        val_full = get_state_value(full2)
        assert val_full > val_half

    def test_opponent_hp_subtracts(self):
        """Opponent with more HP reduces our state value."""
        low_opp = make_battle(own_team=[], opp_team=[make_pokemon(0.1)])
        high_opp = make_battle(own_team=[], opp_team=[make_pokemon(1.0)])
        assert get_state_value(low_opp) > get_state_value(high_opp)

    def test_symmetric_full_teams_cancel(self):
        """Full-HP teams on both sides with no other factors: own positives and opp negatives."""
        own = [make_pokemon(1.0)] * MAX_TEAM_SIZE
        opp = [make_pokemon(1.0)] * MAX_TEAM_SIZE
        battle = make_battle(own_team=own, opp_team=opp)
        # own: 6 * 1.0 * HP_VALUE = 6.0
        # opp: -6 * 1.0 * HP_VALUE = -6.0
        # padding: (6-6)*HP + -(6-6)*HP = 0
        assert get_state_value(battle) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Faint bonus/penalty
# ---------------------------------------------------------------------------

class TestFaintRewards:

    def test_own_fainted_mon_penalises(self):
        """Fainted own mon → -FAINTED_VALUE instead of hp fraction."""
        fainted = make_pokemon(hp_fraction=0.0, fainted=True)
        alive = make_pokemon(hp_fraction=0.0, fainted=False)
        b_fainted = make_battle(own_team=[fainted], opp_team=[])
        b_alive = make_battle(own_team=[alive], opp_team=[])
        # Fainted adds -FAINTED_VALUE; alive with 0 hp adds 0
        assert get_state_value(b_fainted) < get_state_value(b_alive)
        diff = get_state_value(b_alive) - get_state_value(b_fainted)
        assert diff == pytest.approx(FAINTED_VALUE)

    def test_opp_fainted_mon_rewards(self):
        """Fainted opponent mon → +FAINTED_VALUE."""
        fainted = make_pokemon(hp_fraction=0.0, fainted=True)
        alive = make_pokemon(hp_fraction=0.0, fainted=False)
        b_fainted = make_battle(own_team=[], opp_team=[fainted])
        b_alive = make_battle(own_team=[], opp_team=[alive])
        assert get_state_value(b_fainted) > get_state_value(b_alive)
        diff = get_state_value(b_fainted) - get_state_value(b_alive)
        assert diff == pytest.approx(FAINTED_VALUE)

    def test_fainted_mon_contributes_hp_and_faint_penalty(self):
        """HP fraction and faint penalty are additive (not mutually exclusive)."""
        fainted_full_hp = make_pokemon(hp_fraction=1.0, fainted=True)
        battle = make_battle(own_team=[fainted_full_hp], opp_team=[])
        value = get_state_value(battle)
        # own: +1.0*HP_VALUE (hp) - FAINTED_VALUE (faint penalty)
        # own padding: (6-1)*HP = +5; opp padding: -(6-0)*HP = -6
        padding = (MAX_TEAM_SIZE - 1) * HP_VALUE - MAX_TEAM_SIZE * HP_VALUE  # -1.0
        assert value == pytest.approx(HP_VALUE - FAINTED_VALUE + padding)


# ---------------------------------------------------------------------------
# 4. Status condition weights
# ---------------------------------------------------------------------------

class TestStatusWeights:

    @pytest.mark.parametrize("status,weight", list(_STATUS_WEIGHTS.items()))
    def test_own_status_penalises(self, status, weight):
        """Any status on own mon reduces value vs. no status."""
        with_status = make_battle(own_team=[make_pokemon(1.0, status=status)], opp_team=[])
        without = make_battle(own_team=[make_pokemon(1.0)], opp_team=[])
        assert get_state_value(with_status) < get_state_value(without)
        diff = get_state_value(without) - get_state_value(with_status)
        assert diff == pytest.approx(weight)

    @pytest.mark.parametrize("status,weight", list(_STATUS_WEIGHTS.items()))
    def test_opp_status_rewards(self, status, weight):
        """Any status on opponent mon increases value vs. no status."""
        with_status = make_battle(own_team=[], opp_team=[make_pokemon(1.0, status=status)])
        without = make_battle(own_team=[], opp_team=[make_pokemon(1.0)])
        assert get_state_value(with_status) > get_state_value(without)
        diff = get_state_value(with_status) - get_state_value(without)
        assert diff == pytest.approx(weight)

    def test_slp_frz_highest_weight(self):
        """SLP and FRZ have the highest status weight (1.2)."""
        assert _STATUS_WEIGHTS[Status.SLP] == pytest.approx(1.2)
        assert _STATUS_WEIGHTS[Status.FRZ] == pytest.approx(1.2)
        for s, w in _STATUS_WEIGHTS.items():
            assert w <= 1.2

    def test_fainted_mon_has_no_status_reward(self):
        """Fainted branch takes priority — status on fainted mon is not double-counted."""
        fainted_slp = make_pokemon(fainted=True, status=Status.SLP)
        fainted_no_status = make_pokemon(fainted=True, status=None)
        b1 = make_battle(own_team=[fainted_slp], opp_team=[])
        b2 = make_battle(own_team=[fainted_no_status], opp_team=[])
        # Both should hit the fainted branch, not the status branch
        assert get_state_value(b1) == pytest.approx(get_state_value(b2))


# ---------------------------------------------------------------------------
# 5 & 6. Boost rewards
# ---------------------------------------------------------------------------

class TestBoostRewards:

    def test_own_positive_spe_boost_increases_value(self):
        """Agent's own +1 spe boost increases state value."""
        boosted = make_battle(own_team=[make_pokemon(boosts={'spe': 1})], opp_team=[])
        unboosted = make_battle(own_team=[make_pokemon()], opp_team=[])
        assert get_state_value(boosted) > get_state_value(unboosted)
        diff = get_state_value(boosted) - get_state_value(unboosted)
        assert diff == pytest.approx(OWN_BOOST_VALUES['spe'])

    def test_own_negative_atk_reduces_value(self):
        """Agent's own -1 atk reduces state value (we were debuffed)."""
        debuffed = make_battle(own_team=[make_pokemon(boosts={'atk': -1})], opp_team=[])
        normal = make_battle(own_team=[make_pokemon()], opp_team=[])
        assert get_state_value(debuffed) < get_state_value(normal)

    def test_opp_negative_spe_increases_value(self):
        """Opponent -1 spe increases state value (we crippled their speed)."""
        crippled = make_battle(own_team=[], opp_team=[make_pokemon(boosts={'spe': -1})])
        normal = make_battle(own_team=[], opp_team=[make_pokemon()])
        assert get_state_value(crippled) > get_state_value(normal)
        diff = get_state_value(crippled) - get_state_value(normal)
        assert diff == pytest.approx(OPP_BOOST_PENALTIES['spe'])

    def test_opp_positive_atk_reduces_value(self):
        """Opponent +1 atk reduces state value (they set up on us)."""
        set_up = make_battle(own_team=[], opp_team=[make_pokemon(boosts={'atk': 1})])
        normal = make_battle(own_team=[], opp_team=[make_pokemon()])
        assert get_state_value(set_up) < get_state_value(normal)


# ---------------------------------------------------------------------------
# 7. WIN_BONUS / LOSS_PENALTY asymmetry
# ---------------------------------------------------------------------------

class TestGameOutcome:

    def test_win_adds_win_bonus(self):
        not_won = make_battle(won=False, lost=False)
        won = make_battle(won=True, lost=False)
        assert get_state_value(won) - get_state_value(not_won) == pytest.approx(WIN_BONUS)

    def test_loss_adds_loss_penalty(self):
        not_lost = make_battle(won=False, lost=False)
        lost = make_battle(won=False, lost=True)
        assert get_state_value(lost) - get_state_value(not_lost) == pytest.approx(LOSS_PENALTY)

    def test_win_bonus_larger_than_loss_penalty_magnitude(self):
        """WIN_BONUS > |LOSS_PENALTY| — asymmetric by design."""
        assert WIN_BONUS > abs(LOSS_PENALTY)

    def test_win_and_loss_flags_independent(self):
        """Only won or lost flag active at a time (poke-env guarantee)."""
        only_win = make_battle(won=True, lost=False)
        only_loss = make_battle(won=False, lost=True)
        assert get_state_value(only_win) > get_state_value(only_loss)


# ---------------------------------------------------------------------------
# 8. Missing team member padding
# ---------------------------------------------------------------------------

class TestTeamPadding:

    def test_own_padding_assumes_full_hp(self):
        """Missing own slot is treated as a full-HP mon; a damaged mon scores lower."""
        damaged_mon = make_pokemon(hp_fraction=0.5)
        with_damaged = make_battle(own_team=[damaged_mon], opp_team=[])
        empty_own = make_battle(own_team=[], opp_team=[])
        # empty: 0 + 6 (padding) - 6 (opp padding) = 0.0
        # with_damaged: 0.5 + 5 (padding) - 6 (opp padding) = -0.5
        assert get_state_value(empty_own) > get_state_value(with_damaged)

    def test_opp_padding_assumes_full_hp(self):
        """Missing opp slot is treated as a full-HP opp; a damaged opp scores higher."""
        damaged_opp = make_pokemon(hp_fraction=0.5)
        with_damaged = make_battle(own_team=[], opp_team=[damaged_opp])
        empty_opp = make_battle(own_team=[], opp_team=[])
        # empty: 6 (own padding) - 6 (opp padding) = 0.0
        # with_damaged: 6 (own padding) - 0.5 - 5 (opp padding) = 0.5
        assert get_state_value(with_damaged) > get_state_value(empty_opp)

    def test_full_teams_no_padding(self):
        """With 6 mons each, padding terms are zero."""
        own = [make_pokemon(1.0)] * MAX_TEAM_SIZE
        opp = [make_pokemon(1.0)] * MAX_TEAM_SIZE
        battle = make_battle(own_team=own, opp_team=opp)
        # own: 6*1.0 - 0 padding; opp: -6*1.0 - 0 padding → sum = 0
        assert get_state_value(battle) == pytest.approx(0.0)
