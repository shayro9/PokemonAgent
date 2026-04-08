import json
import tempfile
import unittest

from teams.generators import matchup_generator, team_generator
from teams.showdown_pokemon import MonArgs, team_from_dict
from teams.teams_util import format_stats_dict


SAMPLE_MON = {
    "name": "Sparky",
    "species": "Pikachu",
    "item": "Light Ball",
    "ability": "Static",
    "moves": ["Thunderbolt", "Volt Tackle"],
    "nature": "Timid",
    "evs": {"spe": 252, "spa": 252, "hp": 4},
    "ivs": {"atk": 0, "hp": 31},
    "gender": "F",
    "level": 50,
    "shiny": True,
    "teraType": "Electric",
}

BULBASAUR = {
    "name": "Bulbasaur",
    "species": "Bulbasaur",
    "ability": "Overgrow",
    "moves": ["Sleep Powder"],
}

SQUIRTLE = {
    "name": "Shell",
    "species": "Squirtle",
    "ability": "Torrent",
    "moves": ["Surf"],
}


class TestFormatStatsDict(unittest.TestCase):
    def test_returns_canonical_order_and_fills_missing_stats(self):
        stats = {"spe": 252, "hp": 4, "spa": 252}

        self.assertEqual(format_stats_dict(stats), "4,0,0,252,0,252")

    def test_returns_empty_string_for_missing_stats(self):
        self.assertEqual(format_stats_dict(None), "")
        self.assertEqual(format_stats_dict({}), "")


class TestShowdownPokemon(unittest.TestCase):
    def test_mon_to_showdown_omits_duplicate_species_and_default_trailing_values(self):
        mon = MonArgs(
            nickname="Bulbasaur",
            species="Bulbasaur",
            ability="Overgrow",
            moves=["Sleep Powder"],
            gender="N",
            level=100,
            happiness=255,
            dynamaxlevel=10,
        )

        self.assertEqual(
            mon.to_showdown(),
            "Bulbasaur|||Overgrow|Sleep Powder|||||||",
        )

    def test_team_from_dict_packs_each_mon_using_showdown_format(self):
        packed_team = team_from_dict([SAMPLE_MON, BULBASAUR]).to_showdown()

        self.assertEqual(
            packed_team,
            "Sparky|Pikachu|Light Ball|Static|Thunderbolt,Volt Tackle|Timid|4,0,0,252,0,252|F|31,0,0,0,0,0|S|50|,,,,,Electric]"
            "Bulbasaur|||Overgrow|Sleep Powder|||||||",
        )


class TestTeamGenerator(unittest.TestCase):
    def test_generator_cycles_pool_without_shuffling_when_disabled(self):
        pool = [[BULBASAUR], [SQUIRTLE]]
        generator = team_generator(pool=pool, shuffle_each_epoch=False)

        self.assertEqual(next(generator), "Bulbasaur|||Overgrow|Sleep Powder|||||||")
        self.assertEqual(next(generator), "Shell|Squirtle||Torrent|Surf|||||||")
        self.assertEqual(next(generator), "Bulbasaur|||Overgrow|Sleep Powder|||||||")

    def test_generator_can_load_pool_from_data_path(self):
        payload = {"pool": [[BULBASAUR]]}

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = handle.name

        generator = team_generator(data_path=path, shuffle_each_epoch=False)

        self.assertEqual(next(generator), "Bulbasaur|||Overgrow|Sleep Powder|||||||")


class TestMatchupGenerator(unittest.TestCase):
    def test_generator_returns_agent_and_opponent_showdown_teams(self):
        pool = [
            {
                "agent": [BULBASAUR],
                "opponent": [SQUIRTLE],
            }
        ]
        generator = matchup_generator(pool=pool, shuffle_each_epoch=False)

        self.assertEqual(
            next(generator),
            (
                "Bulbasaur|||Overgrow|Sleep Powder|||||||",
                "Shell|Squirtle||Torrent|Surf|||||||",
            ),
        )

    def test_empty_pool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Pool is empty"):
            team_generator(pool=[])


class TestTeamSize(unittest.TestCase):
    """Test team_size property across different generator types and pool formats."""

    def test_team_size_with_single_pokemon_team_generator(self):
        """team_size returns 1 for single-Pokémon teams."""
        pool = [[BULBASAUR]]
        generator = team_generator(pool=pool, shuffle_each_epoch=False)

        self.assertEqual(generator.team_size, 1)

    def test_team_size_with_multiple_pokemon_team_generator(self):
        """team_size returns correct count for multi-Pokémon teams."""
        pool = [[BULBASAUR, SQUIRTLE, SAMPLE_MON]]
        generator = team_generator(pool=pool, shuffle_each_epoch=False)

        self.assertEqual(generator.team_size, 3)

    def test_team_size_with_full_size_team(self):
        """team_size correctly counts a full 6-Pokémon team."""
        full_team = [BULBASAUR, SQUIRTLE, SAMPLE_MON] * 2
        pool = [full_team]
        generator = team_generator(pool=pool, shuffle_each_epoch=False)

        self.assertEqual(generator.team_size, 6)

    def test_team_size_with_empty_pool_returns_zero(self):
        """team_size returns 0 for empty pool."""
        generator = team_generator.__wrapped__(
            pool=[], shuffle_each_epoch=False
        ) if hasattr(team_generator, '__wrapped__') else None
        
        # Direct instantiation since team_generator validates pool
        from teams.generators import InfinitePoolGenerator
        from teams.showdown_pokemon import team_from_dict
        
        generator = InfinitePoolGenerator(
            [],
            transform_fn=lambda e: team_from_dict(e).to_showdown(),
            shuffle_each_epoch=False
        )

        self.assertEqual(generator.team_size, 0)

    def test_team_size_with_matchup_generator_counts_agent_team(self):
        """team_size with matchup generator returns agent team size."""
        pool = [
            {
                "agent": [BULBASAUR, SQUIRTLE],
                "opponent": [SAMPLE_MON],
            }
        ]
        generator = matchup_generator(pool=pool, shuffle_each_epoch=False)

        self.assertEqual(generator.team_size, 2)

    def test_team_size_with_multiple_matchups_uses_first_entry(self):
        """team_size with multiple matchups checks first pool entry."""
        pool = [
            {
                "agent": [BULBASAUR],
                "opponent": [SQUIRTLE],
            },
            {
                "agent": [BULBASAUR, SQUIRTLE, SAMPLE_MON],
                "opponent": [SAMPLE_MON],
            },
        ]
        generator = matchup_generator(pool=pool, shuffle_each_epoch=False)

        # Should use first entry's agent team
        self.assertEqual(generator.team_size, 1)

    def test_team_size_remains_consistent_after_next_calls(self):
        """team_size property remains unchanged after calling next()."""
        pool = [[BULBASAUR, SQUIRTLE]]
        generator = team_generator(pool=pool, shuffle_each_epoch=False)

        initial_size = generator.team_size
        next(generator)
        next(generator)

        self.assertEqual(generator.team_size, initial_size)

    def test_team_size_remains_consistent_after_reset(self):
        """team_size property remains unchanged after reset()."""
        pool = [[BULBASAUR, SQUIRTLE, SAMPLE_MON]]
        generator = team_generator(pool=pool, shuffle_each_epoch=False)

        initial_size = generator.team_size
        next(generator)
        generator.reset()

        self.assertEqual(generator.team_size, initial_size)

    def test_team_size_with_seeded_generator(self):
        """team_size works correctly with seeded generators."""
        pool = [[BULBASAUR, SQUIRTLE]]
        generator = team_generator(pool=pool, seed=42, shuffle_each_epoch=True)

        self.assertEqual(generator.team_size, 2)

    def test_team_size_with_shuffle_disabled(self):
        """team_size works correctly with shuffle_each_epoch=False."""
        pool = [[BULBASAUR, SQUIRTLE]]
        generator = team_generator(pool=pool, shuffle_each_epoch=False)

        self.assertEqual(generator.team_size, 2)

    def test_team_size_detects_non_dict_entries(self):
        """team_size handles non-dict pool entries (direct lists)."""
        pool = [[BULBASAUR, SQUIRTLE]]
        generator = team_generator(pool=pool, shuffle_each_epoch=False)

        # Non-dict entries (direct lists) use len() directly
        self.assertEqual(generator.team_size, 2)


class TestInfinitePoolGeneratorFork(unittest.TestCase):
    """Tests for InfinitePoolGenerator.fork()."""

    def _make_gen(self, seed=42, shuffle=True):
        from teams.generators import InfinitePoolGenerator
        pool = [[BULBASAUR], [SQUIRTLE], [SAMPLE_MON]]
        return InfinitePoolGenerator(
            pool,
            transform_fn=lambda e: team_from_dict(e).to_showdown(),
            seed=seed,
            shuffle_each_epoch=shuffle,
        )

    def test_fork_offsets_seed_by_worker_id(self):
        """fork(n) produces a generator with seed = original_seed + n."""
        gen = self._make_gen(seed=10)
        forked = gen.fork(5)
        self.assertEqual(forked._seed, 15)

    def test_fork_worker_0_replicates_original_sequence(self):
        """fork(0) produces the same sequence as an identically seeded generator."""
        gen = self._make_gen(seed=42)
        forked = gen.fork(0)
        original_seq = [next(gen) for _ in range(6)]
        forked_seq = [next(forked) for _ in range(6)]
        self.assertEqual(original_seq, forked_seq)

    def test_fork_unseed_generator_uses_worker_id_as_seed(self):
        """fork on a seeded=None generator uses worker_id as the new seed."""
        from teams.generators import InfinitePoolGenerator
        pool = [[BULBASAUR], [SQUIRTLE]]
        gen = InfinitePoolGenerator(
            pool,
            transform_fn=lambda e: team_from_dict(e).to_showdown(),
            seed=None,
            shuffle_each_epoch=True,
        )
        forked = gen.fork(7)
        self.assertEqual(forked._seed, 7)

    def test_fork_distinct_workers_produce_different_sequences(self):
        """Different worker_ids yield different team orderings."""
        pool = [[BULBASAUR], [SQUIRTLE], [SAMPLE_MON]]
        from teams.generators import InfinitePoolGenerator
        gen = InfinitePoolGenerator(
            pool,
            transform_fn=lambda e: team_from_dict(e).to_showdown(),
            seed=0,
            shuffle_each_epoch=True,
        )
        seq_w0 = [next(gen.fork(0)) for _ in range(3)]
        seq_w1 = [next(gen.fork(1)) for _ in range(3)]
        seq_w2 = [next(gen.fork(2)) for _ in range(3)]
        # At least two of the three should differ (seed offset must change order)
        sequences = [seq_w0, seq_w1, seq_w2]
        self.assertGreater(len(set(map(tuple, sequences))), 1)

    def test_fork_preserves_pool_and_config(self):
        """fork copies pool, transform_fn, and shuffle_each_epoch unchanged."""
        gen = self._make_gen(seed=99, shuffle=False)
        forked = gen.fork(3)
        self.assertEqual(forked._pool, gen._pool)
        self.assertEqual(forked._shuffle_each_epoch, False)

    def test_fork_does_not_mutate_original(self):
        """Calling fork does not change the original generator's seed or state."""
        gen = self._make_gen(seed=42)
        first_from_original = next(gen)
        gen.fork(10)
        gen.reset()
        self.assertEqual(next(gen), first_from_original)


if __name__ == "__main__":
    unittest.main()