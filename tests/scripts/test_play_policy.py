"""Unit tests for scripts.play_policy module."""
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO
import argparse

from scripts.play_policy import (
    build_arg_parser, _resolve_team, _print_packed_team,
    AGENT_CHOICES, DEFAULT_FORMAT, DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH,
)
from tests.conftest import MockTeamGenerator


class TestBuildArgParser(unittest.TestCase):
    """Test build_arg_parser function."""

    def test_parser_has_required_arguments(self):
        """Verify parser has all expected arguments."""
        parser = build_arg_parser()
        
        # Test that challenge-user is required
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_parser_default_values(self):
        """Verify parser sets correct defaults."""
        parser = build_arg_parser()
        args = parser.parse_args(['--challenge-user', 'TestUser'])
        
        self.assertEqual(args.agent, "policy")
        self.assertEqual(args.model_path, DEFAULT_MODEL_PATH)
        self.assertEqual(args.ai_team, "random")
        self.assertEqual(args.my_team, "random")
        self.assertEqual(args.data_path, DEFAULT_DATA_PATH)
        self.assertEqual(args.format, DEFAULT_FORMAT)
        self.assertEqual(args.n_challenges, 1)
        self.assertFalse(args.no_verbose)

    def test_parser_agent_choices(self):
        """Verify agent argument accepts valid choices."""
        parser = build_arg_parser()
        
        for agent in AGENT_CHOICES:
            args = parser.parse_args(['--challenge-user', 'User', '--agent', agent])
            self.assertEqual(args.agent, agent)

    def test_parser_agent_invalid_choice_fails(self):
        """Verify invalid agent choice is rejected."""
        parser = build_arg_parser()
        
        with self.assertRaises(SystemExit):
            parser.parse_args(['--challenge-user', 'User', '--agent', 'invalid'])

    def test_parser_custom_model_path(self):
        """Verify custom model path is accepted."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'User',
            '--model-path', 'custom/path/model'
        ])
        
        self.assertEqual(args.model_path, 'custom/path/model')

    def test_parser_custom_format(self):
        """Verify custom battle format is accepted."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'User',
            '--format', 'gen8ou'
        ])
        
        self.assertEqual(args.format, 'gen8ou')

    def test_parser_n_challenges(self):
        """Verify n_challenges integer argument."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'User',
            '--n-challenges', '5'
        ])
        
        self.assertEqual(args.n_challenges, 5)

    def test_parser_no_verbose_flag(self):
        """Verify no-verbose flag disables verbosity."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'User',
            '--no-verbose'
        ])
        
        self.assertTrue(args.no_verbose)

    def test_parser_team_size(self):
        """Verify team-size argument."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'User',
            '--team-size', '3'
        ])
        
        self.assertEqual(args.team_size, 3)

    def test_parser_player_name(self):
        """Verify player-name argument."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'User',
            '--player-name', 'MyBot'
        ])
        
        self.assertEqual(args.player_name, 'MyBot')


class TestResolveTeam(unittest.TestCase):
    """Test _resolve_team function."""

    @patch('scripts.play_policy.matchup_generator')
    def test_resolve_team_returns_team_string(self, mock_gen):
        """Verify team is resolved as a string."""
        mock_gen.return_value = iter([("agent_team", "opponent_team")])
        
        pool = [{"agent": [{}], "opponent": [{}]}]
        
        team = _resolve_team("random", pool, side="agent", team_size=1)
        
        self.assertEqual(team, "agent_team")

    @patch('scripts.play_policy.matchup_generator')
    def test_resolve_team_opponent_side(self, mock_gen):
        """Verify team is selected from opponent side."""
        mock_gen.return_value = iter([("agent_team", "opponent_team")])
        
        pool = [{"agent": [{}], "opponent": [{}]}]
        
        team = _resolve_team("random", pool, side="opponent", team_size=1)
        
        self.assertEqual(team, "opponent_team")


class TestPrintPackedTeam(unittest.TestCase):
    """Test _print_packed_team function."""

    def test_print_packed_team_simple_format(self):
        """Verify team printing works with simple format."""
        # Simple packed format (not newline-separated)
        packed = "Alakazam|Alakazam||Magic Guard|Psychic,Focus Blast"
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            _print_packed_team("Test Team", packed)
            output = mock_stdout.getvalue()
        
        self.assertIn("Test Team", output)
        self.assertIn("Alakazam", output)

    def test_print_packed_team_with_label(self):
        """Verify label is printed."""
        packed = "Alakazam|Alakazam||Magic Guard|Psychic"
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            _print_packed_team("My Custom Label", packed)
            output = mock_stdout.getvalue()
        
        self.assertIn("My Custom Label", output)

    def test_print_packed_team_extracts_pokemon_name(self):
        """Verify Pokemon species is extracted and displayed."""
        packed = "AlakazamNickname|Alakazam||Magic Guard|Psychic"
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            _print_packed_team("Team", packed)
            output = mock_stdout.getvalue()
        
        # Should show species info
        self.assertIn("Pokemon", output)

    def test_print_packed_team_shows_moves(self):
        """Verify moves are displayed."""
        packed = "AlakazamBot|Alakazam||Magic Guard|Psychic,Focus Blast,Dazzling Gleam"
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            _print_packed_team("Team", packed)
            output = mock_stdout.getvalue()
        
        self.assertIn("Moves", output)

    def test_print_packed_team_handles_newline_format(self):
        """Verify newline-separated format is handled."""
        # Export format with newlines
        packed = """Alakazam
Alakazam @ Light Ball
Ability: Magic Guard
Moves: Psychic, Focus Blast"""
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            _print_packed_team("Team", packed)
            output = mock_stdout.getvalue()
        
        self.assertIn("Team", output)
        # Should not crash, should print the format as-is

    def test_print_packed_team_multiple_pokemon(self):
        """Verify multiple Pokemon are displayed."""
        # Multiple Pokemon separated by ]
        packed = "Alakazam|Alakazam||Magic Guard|Psychic]]Chansey|Chansey||Serene Grace|Soft Boiled"
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            _print_packed_team("Team", packed)
            output = mock_stdout.getvalue()
        
        self.assertIn("Alakazam", output)
        self.assertIn("Chansey", output)


class TestPlayPolicyArgumentIntegration(unittest.TestCase):
    """Integration tests for argument parsing."""

    def test_parse_args_policy_agent_with_model(self):
        """Verify policy agent can be configured with model path."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'Player1',
            '--agent', 'policy',
            '--model-path', 'models/trained_model',
        ])
        
        self.assertEqual(args.agent, 'policy')
        self.assertEqual(args.model_path, 'models/trained_model')

    def test_parse_args_random_agent(self):
        """Verify random agent configuration."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'Player1',
            '--agent', 'random',
        ])
        
        self.assertEqual(args.agent, 'random')

    def test_parse_args_max_power_agent(self):
        """Verify max-power agent configuration."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'Player1',
            '--agent', 'max-power',
        ])
        
        self.assertEqual(args.agent, 'max-power')

    def test_parse_args_with_fixed_teams(self):
        """Verify fixed team arguments are accepted."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'Player1',
            '--ai-team', 'random',
            '--my-team', 'random',
        ])
        
        self.assertEqual(args.ai_team, 'random')
        self.assertEqual(args.my_team, 'random')

    def test_parse_args_custom_data_path(self):
        """Verify custom data path is accepted."""
        parser = build_arg_parser()
        args = parser.parse_args([
            '--challenge-user', 'Player1',
            '--data-path', 'custom/data/path.json',
        ])
        
        self.assertEqual(args.data_path, 'custom/data/path.json')


if __name__ == "__main__":
    unittest.main()
