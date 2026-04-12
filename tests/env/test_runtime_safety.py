"""Tests for env.runtime_safety helpers."""

from traceback import FrameSummary
from unittest.mock import patch

from env.runtime_safety import is_external_battle_exception


class TestIsExternalBattleException:
    def test_true_for_poke_env_only_traceback(self):
        exc = RuntimeError("boom")
        fake_tb = [
            FrameSummary(
                "/usr/local/lib/python3.11/site-packages/poke_env/environment.py",
                123,
                "step",
            )
        ]
        with patch("env.runtime_safety.traceback.extract_tb", return_value=fake_tb):
            assert is_external_battle_exception(exc)

    def test_false_when_repository_frame_exists(self):
        exc = RuntimeError("boom")
        fake_tb = [
            FrameSummary(
                "/usr/local/lib/python3.11/site-packages/poke_env/environment.py",
                123,
                "step",
            ),
            FrameSummary(
                "/workspace/PokemonAgent/env/singles_env_wrapper.py",
                88,
                "action_to_order",
            ),
        ]
        with patch("env.runtime_safety.traceback.extract_tb", return_value=fake_tb):
            assert not is_external_battle_exception(exc)
