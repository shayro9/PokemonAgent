"""Unit tests for teams/team_generators.py."""

import pytest
from pathlib import Path

from teams.team_generators import (
    format_stats,
    mon_to_kwargs,
    build_slot,
    load_pool,
    split_pool,
    sample_team,
    iter_teams,
    iter_matchups,
    # backward-compat aliases
    format_stats_dict,
    _mon_kwargs,
    generate_team,
    load_pokemon_pool,
    split_pokemon_pool,
    sample_team_of_n,
)

# Absolute path to the repo root — works regardless of where pytest is invoked
ROOT = Path(__file__).parent.parent
GEN1OU_DB = str(ROOT / "data" / "matchups_gen1ou_db.json")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

GEN1_MON = {
    "name": "Tauros",
    "species": "Tauros",
    "ability": "No Ability",
    "item": "",
    "moves": ["bodyslam", "earthquake", "blizzard", "hyperbeam"],
    "nature": "Serious",
    "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
    "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
    "gender": "N",
    "level": 68,
    "shiny": False,
}

MATCHUP_ENTRY = {
    "agent":    GEN1_MON,
    "opponent": {**GEN1_MON, "name": "Starmie", "species": "Starmie",
                 "moves": ["surf", "blizzard", "thunderbolt", "recover"]},
}

SMALL_POOL = [MATCHUP_ENTRY] * 10   # 10 identical entries — enough for split/sample tests


# ---------------------------------------------------------------------------
# format_stats
# ---------------------------------------------------------------------------

class TestFormatStats:
    def test_all_values(self):
        stats = {"hp": 255, "atk": 200, "def": 100, "spa": 50, "spd": 25, "spe": 10}
        assert format_stats(stats) == "255,200,100,50,25,10"

    def test_missing_keys_default_to_zero(self):
        assert format_stats({"hp": 100}) == "100,0,0,0,0,0"

    def test_none_returns_empty(self):
        assert format_stats(None) == ""

    def test_empty_dict_returns_empty(self):
        assert format_stats({}) == ""

    def test_alias_matches(self):
        stats = {"hp": 1, "atk": 2, "def": 3, "spa": 4, "spd": 5, "spe": 6}
        assert format_stats_dict(stats) == format_stats(stats)


# ---------------------------------------------------------------------------
# mon_to_kwargs
# ---------------------------------------------------------------------------

class TestMonToKwargs:
    def test_basic_fields(self):
        kwargs = mon_to_kwargs(GEN1_MON)
        assert kwargs["species"] == "Tauros"
        assert kwargs["nickname"] == "Tauros"
        assert kwargs["moves"] == ["bodyslam", "earthquake", "blizzard", "hyperbeam"]
        assert kwargs["level"] == 68

    def test_no_ability_stripped(self):
        kwargs = mon_to_kwargs(GEN1_MON)
        assert kwargs["ability"] == ""

    def test_noability_stripped(self):
        mon = {**GEN1_MON, "ability": "noability"}
        assert mon_to_kwargs(mon)["ability"] == ""

    def test_real_ability_kept(self):
        mon = {**GEN1_MON, "ability": "Intimidate"}
        assert mon_to_kwargs(mon)["ability"] == "Intimidate"

    def test_gender_N_passed_through(self):
        kwargs = mon_to_kwargs(GEN1_MON)
        assert kwargs["gender"] == "N"

    def test_evs_formatted(self):
        kwargs = mon_to_kwargs(GEN1_MON)
        assert kwargs["evs"] == "255,255,255,255,255,255"

    def test_ivs_formatted(self):
        kwargs = mon_to_kwargs(GEN1_MON)
        assert kwargs["ivs"] == "30,30,30,30,30,30"

    def test_alias_matches(self):
        assert _mon_kwargs(GEN1_MON) == mon_to_kwargs(GEN1_MON)


# ---------------------------------------------------------------------------
# build_slot
# ---------------------------------------------------------------------------

class TestBuildSlot:
    def _slot(self, **overrides):
        kwargs = mon_to_kwargs(GEN1_MON)
        kwargs.update(overrides)
        return build_slot(**kwargs)

    def test_produces_11_pipes(self):
        slot = self._slot()
        assert slot.count("|") == 11

    def test_nickname_is_field_0(self):
        slot = self._slot()
        assert slot.split("|")[0] == "Tauros"

    def test_ability_field_empty_for_gen1(self):
        slot = self._slot()
        assert slot.split("|")[3] == ""

    def test_moves_field(self):
        slot = self._slot()
        assert slot.split("|")[4] == "bodyslam,earthquake,blizzard,hyperbeam"

    def test_level_field(self):
        slot = self._slot()
        assert slot.split("|")[10] == "68"

    def test_level_100_is_blank(self):
        slot = self._slot(level=100)
        assert slot.split("|")[10] == ""

    def test_gender_N_is_blank(self):
        # gender "N" should be stored as empty in packed format
        slot = self._slot()
        assert slot.split("|")[7] == ""

    def test_shiny(self):
        slot = self._slot(shiny=True)
        assert slot.split("|")[9] == "S"

    def test_not_shiny(self):
        slot = self._slot(shiny=False)
        assert slot.split("|")[9] == ""

    def test_alias_matches(self):
        kwargs = mon_to_kwargs(GEN1_MON)
        assert generate_team(**kwargs) == build_slot(**kwargs)


# ---------------------------------------------------------------------------
# load_pool / split_pool
# ---------------------------------------------------------------------------

class TestLoadPool:
    def test_loads_real_file(self):
        pool = load_pool(GEN1OU_DB)
        assert isinstance(pool, list)
        assert len(pool) > 0

    def test_entries_have_agent_and_opponent(self):
        pool = load_pool(GEN1OU_DB)
        assert "agent" in pool[0]
        assert "opponent" in pool[0]

    def test_alias_matches(self):
        assert load_pokemon_pool(GEN1OU_DB) == load_pool(GEN1OU_DB)


class TestSplitPool:
    def test_sizes_sum_to_original(self):
        pool = list(range(100))          # any list works
        train, eval_ = split_pool(pool, 0.8, seed=0)  # type: ignore[arg-type]
        assert len(train) + len(eval_) == 100

    def test_train_fraction_respected(self):
        pool = list(range(100))
        train, _ = split_pool(pool, 0.8, seed=0)  # type: ignore[arg-type]
        assert len(train) == 80

    def test_reproducible_with_same_seed(self):
        pool = list(range(50))
        a, _ = split_pool(pool, 0.8, seed=42)  # type: ignore[arg-type]
        b, _ = split_pool(pool, 0.8, seed=42)  # type: ignore[arg-type]
        assert a == b

    def test_different_seeds_differ(self):
        pool = list(range(50))
        a, _ = split_pool(pool, 0.8, seed=1)   # type: ignore[arg-type]
        b, _ = split_pool(pool, 0.8, seed=2)   # type: ignore[arg-type]
        assert a != b

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            split_pool(SMALL_POOL, 0.0, seed=0)
        with pytest.raises(ValueError):
            split_pool(SMALL_POOL, 1.0, seed=0)

    def test_alias_matches(self):
        a = split_pool(SMALL_POOL, 0.8, seed=0)
        b = split_pokemon_pool(SMALL_POOL, 0.8, seed=0)
        assert a == b


# ---------------------------------------------------------------------------
# sample_team
# ---------------------------------------------------------------------------

class TestSampleTeam:
    def test_returns_string(self):
        assert isinstance(sample_team(SMALL_POOL), str)

    def test_single_pokemon_no_bracket(self):
        packed = sample_team(SMALL_POOL, n=1)
        assert "]" not in packed

    def test_multi_pokemon_joined_with_bracket(self):
        packed = sample_team(SMALL_POOL, n=3)
        assert packed.count("]") == 2      # 3 slots → 2 separators

    def test_n_equals_pool_size(self):
        small = SMALL_POOL[:4]
        packed = sample_team(small, n=4)
        assert packed.count("]") == 3

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            sample_team(SMALL_POOL, n=0)
        with pytest.raises(ValueError):
            sample_team(SMALL_POOL, n=7)

    def test_opponent_side(self):
        packed = sample_team(SMALL_POOL, n=1, side="opponent")
        assert packed.split("|")[0] == "Starmie"

    def test_agent_side(self):
        packed = sample_team(SMALL_POOL, n=1, side="agent")
        assert packed.split("|")[0] == "Tauros"

    def test_alias_matches(self):
        import random as _random
        _random.seed(99)
        a = sample_team(SMALL_POOL, n=2)
        _random.seed(99)
        b = sample_team_of_n(SMALL_POOL, n=2)
        assert a == b


# ---------------------------------------------------------------------------
# iter_teams
# ---------------------------------------------------------------------------

class TestIterTeams:
    def test_yields_strings(self):
        gen = iter_teams(SMALL_POOL, seed=0)
        for _ in range(5):
            assert isinstance(next(gen), str)

    def test_seeded_is_reproducible(self):
        gen1 = iter_teams(SMALL_POOL, seed=7)
        gen2 = iter_teams(SMALL_POOL, seed=7)
        results1 = [next(gen1) for _ in range(10)]
        results2 = [next(gen2) for _ in range(10)]
        assert results1 == results2

    def test_different_seeds_differ(self):
        pool = load_pool(GEN1OU_DB)
        gen1 = iter_teams(pool, seed=1)
        gen2 = iter_teams(pool, seed=2)
        results1 = [next(gen1) for _ in range(20)]
        results2 = [next(gen2) for _ in range(20)]
        assert results1 != results2

    def test_n_pokemon_per_team(self):
        gen = iter_teams(SMALL_POOL, n=3, seed=0)
        packed = next(gen)
        assert packed.count("]") == 2


# ---------------------------------------------------------------------------
# iter_matchups
# ---------------------------------------------------------------------------

class TestIterMatchups:
    def test_yields_tuples(self):
        gen = iter_matchups(SMALL_POOL, seed=0)
        pair = next(gen)
        assert isinstance(pair, tuple)
        assert len(pair) == 2

    def test_both_sides_are_strings(self):
        gen = iter_matchups(SMALL_POOL, seed=0)
        agent, opp = next(gen)
        assert isinstance(agent, str)
        assert isinstance(opp, str)

    def test_agent_and_opponent_differ(self):
        gen = iter_matchups(SMALL_POOL, seed=0)
        agent, opp = next(gen)
        assert agent != opp

    def test_seeded_is_reproducible(self):
        gen1 = iter_matchups(SMALL_POOL, seed=3)
        gen2 = iter_matchups(SMALL_POOL, seed=3)
        pairs1 = [next(gen1) for _ in range(5)]
        pairs2 = [next(gen2) for _ in range(5)]
        assert pairs1 == pairs2

    def test_is_infinite(self):
        # pool has 10 entries; pulling 30 should cycle without StopIteration
        gen = iter_matchups(SMALL_POOL, seed=0)
        for _ in range(30):
            next(gen)


