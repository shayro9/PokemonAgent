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
                "agent": BULBASAUR,
                "opponent": SQUIRTLE,
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


if __name__ == "__main__":
    unittest.main()