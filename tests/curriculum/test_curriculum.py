"""Tests for curriculum models, registry, runtime, and YAML loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from poke_env.player.player import Player

from curriculum.models import CurriculumStage, OpponentPlayerSpec
from curriculum.registry import build_opponent_player, resolve_opponent_player_class
from curriculum.runtime import LinearCurriculum
from curriculum.yaml_loader import load_curriculum_from_yaml


class CustomCurriculumPlayer(Player):
    """Custom Player used to verify class_path loading."""

    def __init__(self, *args, custom_flag: bool = False, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.custom_flag = custom_flag

    def choose_move(self, battle):
        return None


class StubPlayer(Player):
    """Stub Player used to verify instantiation arguments."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def choose_move(self, battle):
        return None


class TestRegistry(unittest.TestCase):
    """Tests for opponent-player registry resolution."""

    def test_resolve_builtin_player_id(self):
        spec = OpponentPlayerSpec(id="random")

        player_cls = resolve_opponent_player_class(spec)

        self.assertEqual(player_cls.__name__, "RandomPlayer")

    def test_resolve_custom_class_path(self):
        spec = OpponentPlayerSpec(
            class_path="tests.curriculum.test_curriculum.CustomCurriculumPlayer"
        )

        player_cls = resolve_opponent_player_class(spec)

        self.assertIs(player_cls, CustomCurriculumPlayer)

    def test_build_opponent_player_passes_framework_and_custom_kwargs(self):
        spec = OpponentPlayerSpec(id="random", kwargs={"custom_flag": True})

        with patch("curriculum.registry.resolve_opponent_player_class", return_value=StubPlayer):
            player = build_opponent_player(
                spec,
                battle_format="gen1ou",
                server_configuration="server",
                account_configuration="account",
            )

        self.assertIsInstance(player, StubPlayer)
        self.assertEqual(player.kwargs["battle_format"], "gen1ou")
        self.assertEqual(player.kwargs["server_configuration"], "server")
        self.assertEqual(player.kwargs["account_configuration"], "account")
        self.assertTrue(player.kwargs["custom_flag"])

    def test_build_opponent_player_rejects_reserved_kwargs(self):
        spec = OpponentPlayerSpec(
            id="random",
            kwargs={"battle_format": "override"},
        )

        with self.assertRaises(ValueError):
            build_opponent_player(
                spec,
                battle_format="gen1ou",
                server_configuration="server",
                account_configuration="account",
            )


class TestLinearCurriculum(unittest.TestCase):
    """Tests for runtime stage resolution."""

    def test_maybe_advance_changes_stage_once_per_boundary(self):
        curriculum = LinearCurriculum(
            [
                CurriculumStage(
                    name="random",
                    start_timestep=0,
                    end_timestep=10,
                    opponent_player=OpponentPlayerSpec(id="random"),
                ),
                CurriculumStage(
                    name="heuristic",
                    start_timestep=10,
                    end_timestep=None,
                    opponent_player=OpponentPlayerSpec(id="heuristic"),
                ),
            ]
        )

        initial = curriculum.initialize()
        mid_stage = curriculum.maybe_advance(5)
        next_stage = curriculum.maybe_advance(10)
        repeated = curriculum.maybe_advance(15)

        self.assertEqual(initial.name, "random")
        self.assertIsNone(mid_stage)
        self.assertEqual(next_stage.name, "heuristic")
        self.assertIsNone(repeated)

    def test_stage_for_timesteps_returns_last_stage_after_final_boundary(self):
        curriculum = LinearCurriculum(
            [
                CurriculumStage(
                    name="random",
                    start_timestep=0,
                    end_timestep=10,
                    opponent_player=OpponentPlayerSpec(id="random"),
                ),
                CurriculumStage(
                    name="max-power",
                    start_timestep=10,
                    end_timestep=20,
                    opponent_player=OpponentPlayerSpec(id="max-power"),
                ),
            ]
        )

        stage = curriculum.stage_for_timesteps(25)

        self.assertEqual(stage.name, "max-power")


class TestYamlLoader(unittest.TestCase):
    """Tests for YAML curriculum parsing."""

    def test_load_curriculum_from_yaml_builds_linear_stages(self):
        yaml_text = """
stages:
  - name: warmup
    duration_timesteps: 10
    opponent_player:
      id: random
  - name: endgame
    opponent_player:
      id: heuristic
"""
        curriculum = self._load_from_text(yaml_text)

        stages = curriculum.stages

        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0].start_timestep, 0)
        self.assertEqual(stages[0].end_timestep, 10)
        self.assertEqual(stages[0].opponent_player.id, "random")
        self.assertEqual(stages[1].start_timestep, 10)
        self.assertIsNone(stages[1].end_timestep)
        self.assertEqual(stages[1].opponent_player.id, "heuristic")

    def test_load_curriculum_from_yaml_requires_boundary_for_non_final_stage(self):
        yaml_text = """
stages:
  - name: warmup
    opponent_player:
      id: random
  - name: endgame
    opponent_player:
      id: heuristic
"""
        with self.assertRaises(ValueError):
            self._load_from_text(yaml_text)

    def test_load_curriculum_from_yaml_supports_custom_class_path(self):
        yaml_text = """
stages:
  - name: custom-stage
    opponent_player:
      class_path: tests.curriculum.test_curriculum.CustomCurriculumPlayer
      kwargs:
        custom_flag: true
"""
        curriculum = self._load_from_text(yaml_text)

        stage = curriculum.stages[0]

        self.assertEqual(
            stage.opponent_player.class_path,
            "tests.curriculum.test_curriculum.CustomCurriculumPlayer",
        )
        self.assertTrue(stage.opponent_player.kwargs["custom_flag"])

    def _load_from_text(self, yaml_text: str) -> LinearCurriculum:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "curriculum.yaml"
            config_path.write_text(yaml_text.strip(), encoding="utf-8")
            return load_curriculum_from_yaml(config_path)
