"""
tests/test_tracker.py
=====================
Unit tests for BattleSnapshot and Tracker (combat/tracker.py).

Run with:
    pytest tests/test_tracker.py -v

Mocking strategy
----------------
- ``detect_my_move_from_events`` and ``detect_opponent_move_from_events`` are
  patched at the module level so tests never touch the real event parser.
- The poke-env ``AbstractBattle`` object is faked with a lightweight MagicMock.
- ``Status`` values are imported directly from poke_env so tests stay faithful
  to the real enum.

Notable: commit() reads ``current_hp`` (raw int), NOT ``current_hp_fraction``.
This differs from BattleTracker.commit() — tests document that behaviour
explicitly so a future unification is easy to spot.
"""

import unittest
from unittest.mock import MagicMock, patch

from poke_env.battle.status import Status

from env.tracker import BattleSnapshot, Tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_battle(
    my_hp: float = 1.0,
    opp_hp: float = 1.0,
    my_status: Status | None = None,
    opp_status: Status | None = None,
) -> MagicMock:
    """Build a minimal mock AbstractBattle for commit() calls.

    commit() reads ``current_hp`` (not ``current_hp_fraction``).
    """
    battle = MagicMock()
    battle.active_pokemon.current_hp_fraction = my_hp
    battle.active_pokemon.status = my_status
    battle.opponent_active_pokemon.current_hp_fraction = opp_hp
    battle.opponent_active_pokemon.status = opp_status
    return battle


_PATCH_MY_MOVE  = "env.tracker.detect_my_move_from_events"
_PATCH_OPP_MOVE = "env.tracker.detect_opponent_move_from_events"


# ===========================================================================
# BattleSnapshot
# ===========================================================================

class TestBattleSnapshotDefaults(unittest.TestCase):
    """A default BattleSnapshot must have safe, sensible zero values."""

    def setUp(self):
        self.snap = BattleSnapshot()

    def test_my_hp_default(self):
        self.assertEqual(self.snap.my_hp, 1.0)

    def test_opp_hp_default(self):
        self.assertEqual(self.snap.opp_hp, 1.0)

    def test_my_status_default_none(self):
        self.assertIsNone(self.snap.my_status)

    def test_opp_status_default_none(self):
        self.assertIsNone(self.snap.opp_status)

    def test_my_move_default_none(self):
        self.assertIsNone(self.snap.my_move)

    def test_opp_move_default_none(self):
        self.assertIsNone(self.snap.opp_move)


class TestBattleSnapshotCustomValues(unittest.TestCase):
    """BattleSnapshot correctly stores explicitly provided values."""

    def setUp(self):
        self.snap = BattleSnapshot(
            my_hp=0.5,
            opp_hp=0.25,
            my_status=Status.BRN,
            opp_status=Status.PAR,
            my_move="tackle",
            opp_move="surf",
        )

    def test_my_hp(self):
        self.assertAlmostEqual(self.snap.my_hp, 0.5)

    def test_opp_hp(self):
        self.assertAlmostEqual(self.snap.opp_hp, 0.25)

    def test_my_status(self):
        self.assertEqual(self.snap.my_status, Status.BRN)

    def test_opp_status(self):
        self.assertEqual(self.snap.opp_status, Status.PAR)

    def test_my_move(self):
        self.assertEqual(self.snap.my_move, "tackle")

    def test_opp_move(self):
        self.assertEqual(self.snap.opp_move, "surf")


class TestBattleSnapshotIsDataclass(unittest.TestCase):
    """BattleSnapshot must be a mutable dataclass (not frozen)."""

    def test_supports_field_mutation(self):
        snap = BattleSnapshot()
        snap.my_hp = 0.3
        self.assertAlmostEqual(snap.my_hp, 0.3)

    def test_equality_by_value(self):
        a = BattleSnapshot(my_hp=0.5, opp_hp=0.5)
        b = BattleSnapshot(my_hp=0.5, opp_hp=0.5)
        self.assertEqual(a, b)

    def test_inequality_on_differing_field(self):
        a = BattleSnapshot(my_hp=0.5)
        b = BattleSnapshot(my_hp=0.9)
        self.assertNotEqual(a, b)


# ===========================================================================
# Tracker — empty state
# ===========================================================================

class TestTrackerEmpty(unittest.TestCase):
    """Properties must return safe defaults when history is empty."""

    def setUp(self):
        self.tracker = Tracker()

    def test_history_is_empty_list(self):
        self.assertEqual(self.tracker.history, [])

    def test_last_my_hp_default(self):
        self.assertEqual(self.tracker.last_my_hp, 1.0)

    def test_last_opp_hp_default(self):
        self.assertEqual(self.tracker.last_opp_hp, 1.0)

    def test_last_my_status_default(self):
        self.assertIsNone(self.tracker.last_my_status)

    def test_last_opp_status_default(self):
        self.assertIsNone(self.tracker.last_opp_status)

    def test_opp_last_move_raises_on_empty(self):
        """opp_last_move has no guard — it raises IndexError on empty history."""
        with self.assertRaises(IndexError):
            _ = self.tracker.opp_last_move

    def test_my_last_move_raises_on_empty(self):
        """my_last_move has no guard — it raises IndexError on empty history."""
        with self.assertRaises(IndexError):
            _ = self.tracker.my_last_move


# ===========================================================================
# Tracker — commit()
# ===========================================================================

class TestTrackerCommitSingleTurn(unittest.TestCase):
    """commit() appends exactly one snapshot with values read from the battle."""

    def setUp(self):
        self.tracker = Tracker()
        self.battle = _make_battle(my_hp=0.8, opp_hp=0.6, my_status=Status.BRN)

    @patch(_PATCH_MY_MOVE, return_value="tackle")
    @patch(_PATCH_OPP_MOVE, return_value="surf")
    def test_history_grows_by_one(self, _opp, _my):
        self.tracker.commit(self.battle)
        self.assertEqual(len(self.tracker.history), 1)

    @patch(_PATCH_MY_MOVE, return_value="tackle")
    @patch(_PATCH_OPP_MOVE, return_value="surf")
    def test_snapshot_my_hp(self, _opp, _my):
        self.tracker.commit(self.battle)
        self.assertAlmostEqual(self.tracker.history[0].my_hp, 0.8)

    @patch(_PATCH_MY_MOVE, return_value="tackle")
    @patch(_PATCH_OPP_MOVE, return_value="surf")
    def test_snapshot_opp_hp(self, _opp, _my):
        self.tracker.commit(self.battle)
        self.assertAlmostEqual(self.tracker.history[0].opp_hp, 0.6)

    @patch(_PATCH_MY_MOVE, return_value="tackle")
    @patch(_PATCH_OPP_MOVE, return_value="surf")
    def test_snapshot_my_status(self, _opp, _my):
        self.tracker.commit(self.battle)
        self.assertEqual(self.tracker.history[0].my_status, Status.BRN)

    @patch(_PATCH_MY_MOVE, return_value="tackle")
    @patch(_PATCH_OPP_MOVE, return_value="surf")
    def test_snapshot_opp_status_none(self, _opp, _my):
        self.tracker.commit(self.battle)
        self.assertIsNone(self.tracker.history[0].opp_status)

    @patch(_PATCH_MY_MOVE, return_value="tackle")
    @patch(_PATCH_OPP_MOVE, return_value="surf")
    def test_snapshot_my_move(self, _opp, _my):
        self.tracker.commit(self.battle)
        self.assertEqual(self.tracker.history[0].my_move, "tackle")

    @patch(_PATCH_MY_MOVE, return_value="tackle")
    @patch(_PATCH_OPP_MOVE, return_value="surf")
    def test_snapshot_opp_move(self, _opp, _my):
        self.tracker.commit(self.battle)
        self.assertEqual(self.tracker.history[0].opp_move, "surf")

    @patch(_PATCH_MY_MOVE, return_value=None)
    @patch(_PATCH_OPP_MOVE, return_value=None)
    def test_snapshot_moves_none_when_parser_returns_none(self, _opp, _my):
        """If the event parser can't detect a move it returns None — snapshot must reflect that."""
        self.tracker.commit(self.battle)
        self.assertIsNone(self.tracker.history[0].my_move)
        self.assertIsNone(self.tracker.history[0].opp_move)


# ===========================================================================
# Tracker — properties after one commit
# ===========================================================================

class TestTrackerPropertiesAfterOneCommit(unittest.TestCase):

    def setUp(self):
        self.tracker = Tracker()
        battle = _make_battle(my_hp=0.7, opp_hp=0.4,
                              my_status=Status.PAR, opp_status=Status.SLP)
        with patch(_PATCH_MY_MOVE, return_value="earthquake"), \
             patch(_PATCH_OPP_MOVE, return_value="blizzard"):
            self.tracker.commit(battle)

    def test_last_my_hp(self):
        self.assertAlmostEqual(self.tracker.last_my_hp, 0.7)

    def test_last_opp_hp(self):
        self.assertAlmostEqual(self.tracker.last_opp_hp, 0.4)

    def test_last_my_status(self):
        self.assertEqual(self.tracker.last_my_status, Status.PAR)

    def test_last_opp_status(self):
        self.assertEqual(self.tracker.last_opp_status, Status.SLP)

    def test_my_last_move(self):
        self.assertEqual(self.tracker.my_last_move, "earthquake")

    def test_opp_last_move(self):
        self.assertEqual(self.tracker.opp_last_move, "blizzard")


# ===========================================================================
# Tracker — multiple commits (properties always reflect the LATEST snapshot)
# ===========================================================================

class TestTrackerMultipleCommits(unittest.TestCase):
    """After several commits, properties must return the most recent snapshot."""

    def _commit(self, tracker, my_hp, opp_hp, my_move, opp_move,
                my_status=None, opp_status=None):
        battle = _make_battle(my_hp=my_hp, opp_hp=opp_hp,
                              my_status=my_status, opp_status=opp_status)
        with patch(_PATCH_MY_MOVE, return_value=my_move), \
             patch(_PATCH_OPP_MOVE, return_value=opp_move):
            tracker.commit(battle)

    def setUp(self):
        self.tracker = Tracker()
        self._commit(self.tracker, 1.0, 1.0, "tackle", "growl")
        self._commit(self.tracker, 0.8, 0.9, "earthquake", "surf", opp_status=Status.BRN)
        self._commit(self.tracker, 0.5, 0.3, "flamethrower", None, my_status=Status.TOX)

    def test_history_length(self):
        self.assertEqual(len(self.tracker.history), 3)

    def test_last_my_hp_is_latest(self):
        self.assertAlmostEqual(self.tracker.last_my_hp, 0.5)

    def test_last_opp_hp_is_latest(self):
        self.assertAlmostEqual(self.tracker.last_opp_hp, 0.3)

    def test_my_last_move_is_latest(self):
        self.assertEqual(self.tracker.my_last_move, "flamethrower")

    def test_opp_last_move_is_latest_none(self):
        self.assertIsNone(self.tracker.opp_last_move)

    def test_last_my_status_is_latest(self):
        self.assertEqual(self.tracker.last_my_status, Status.TOX)

    def test_last_opp_status_is_latest(self):
        """opp_status cleared on turn 3 — last snapshot reflects that."""
        self.assertIsNone(self.tracker.last_opp_status)

    def test_earlier_snapshots_preserved(self):
        self.assertEqual(self.tracker.history[0].my_move, "tackle")
        self.assertEqual(self.tracker.history[1].opp_move, "surf")

    def test_each_snapshot_is_independent(self):
        """Mutating a snapshot must not affect others."""
        self.tracker.history[0].my_hp = 99.0
        self.assertAlmostEqual(self.tracker.history[1].my_hp, 0.8)
        self.assertAlmostEqual(self.tracker.last_my_hp, 0.5)


# ===========================================================================
# Tracker — event parser is called with the battle object
# ===========================================================================

class TestTrackerDelegatesEventParsing(unittest.TestCase):
    """commit() must forward the battle object to both event-parser functions."""

    def test_my_move_parser_receives_battle(self):
        tracker = Tracker()
        battle = _make_battle()
        with patch(_PATCH_MY_MOVE, return_value=None) as mock_my, \
             patch(_PATCH_OPP_MOVE, return_value=None):
            tracker.commit(battle)
            mock_my.assert_called_once_with(battle)

    def test_opp_move_parser_receives_battle(self):
        tracker = Tracker()
        battle = _make_battle()
        with patch(_PATCH_MY_MOVE, return_value=None), \
             patch(_PATCH_OPP_MOVE, return_value=None) as mock_opp:
            tracker.commit(battle)
            mock_opp.assert_called_once_with(battle)

    def test_both_parsers_called_once_per_commit(self):
        tracker = Tracker()
        battle = _make_battle()
        with patch(_PATCH_MY_MOVE, return_value=None) as mock_my, \
             patch(_PATCH_OPP_MOVE, return_value=None) as mock_opp:
            tracker.commit(battle)
            tracker.commit(battle)
            self.assertEqual(mock_my.call_count, 2)
            self.assertEqual(mock_opp.call_count, 2)


# ===========================================================================
# Tracker — independent instances share no state
# ===========================================================================

class TestTrackerIsolation(unittest.TestCase):
    """Two Tracker instances must not share history."""

    def test_separate_histories(self):
        t1 = Tracker()
        t2 = Tracker()
        battle = _make_battle(my_hp=0.5)
        with patch(_PATCH_MY_MOVE, return_value=None), \
             patch(_PATCH_OPP_MOVE, return_value=None):
            t1.commit(battle)

        self.assertEqual(len(t1.history), 1)
        self.assertEqual(len(t2.history), 0)


if __name__ == "__main__":
    unittest.main()