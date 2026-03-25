"""Shared pytest configuration and fixtures for all tests."""
import json
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# ===========================================================================
# Mock Helpers
# ===========================================================================

def make_mock_battle(
    player_username: str = "test_player",
    opponent_username: str = "opponent",
    finished: bool = False,
    won: bool = False,
    lost: bool = False,
    turn: int = 1,
    active_pokemon_moves: dict | None = None,
    available_moves: list | None = None,
    available_switches: list | None = None,
    team: dict | None = None,
):
    """Create a mock battle object for testing.
    
    :param player_username: Username of the player (agent).
    :param opponent_username: Username of the opponent.
    :param finished: Whether the battle is finished.
    :param won: Whether the player won (only if finished=True).
    :param lost: Whether the player lost (only if finished=True).
    :param turn: Current turn number.
    :param active_pokemon_moves: Dict of move ID to mock move objects.
    :param available_moves: List of available move objects.
    :param available_switches: List of available Pokemon to switch to.
    :param team: Dict of Pokemon species name to mock Pokemon object.
    :returns: A mock Battle object."""
    battle = MagicMock()
    battle.player_username = player_username
    battle.opponent_username = opponent_username
    battle.finished = finished
    battle.won = won
    battle.lost = lost
    battle.turn = turn
    
    # Setup active pokemon and moves
    active_pkmn = MagicMock()
    active_pkmn.moves = active_pokemon_moves or {}
    battle.active_pokemon = active_pkmn
    battle.available_moves = available_moves or []
    battle.available_switches = available_switches or []
    battle.team = team or {}
    
    return battle


def make_mock_move(move_id: str, power: int = 100, accuracy: int = 100):
    """Create a mock Move object.
    
    :param move_id: Move identifier.
    :param power: Move power/base power.
    :param accuracy: Move accuracy percentage.
    :returns: A mock Move object."""
    move = MagicMock()
    move.id = move_id
    move.power = power
    move.accuracy = accuracy
    return move


def make_mock_pokemon(species: str, level: int = 100):
    """Create a mock Pokemon object.
    
    :param species: Pokemon species name.
    :param level: Pokemon level.
    :returns: A mock Pokemon object."""
    pkmn = MagicMock()
    pkmn.species = species
    pkmn.level = level
    return pkmn


def make_mock_model():
    """Create a mock MaskablePPO model.
    
    :returns: A mock MaskablePPO model object."""
    model = MagicMock()
    model.predict.return_value = (0, None)  # (action, _info)
    model.learn.return_value = None
    model.save.return_value = None
    return model


def make_mock_generator(items: list):
    """Create a simple mock generator that cycles through items.
    
    :param items: List of items to yield.
    :returns: A generator function."""
    def gen():
        while True:
            for item in items:
                yield item
    return gen()


# ===========================================================================
# Fixture Helpers
# ===========================================================================

@dataclass
class MockEnvironmentSpec:
    """Helper for creating mock environment specs."""
    battle_format: str = "gen1ou"
    rounds_per_opponent: int = 2000
    action_space_size: int = 26
    observation_space_shape: tuple = (768,)  # BattleStateGen1.array_len()


class MockTeamGenerator:
    """Mock team generator for testing."""
    
    def __init__(self, teams: list[str] | None = None):
        """Initialize with a list of packed team strings.
        
        :param teams: List of packed team strings to cycle through."""
        self.teams = teams or ["Alakazam|||Magic Guard|Psychic|||||||"]
        self.index = 0
        self.reset_count = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        team = self.teams[self.index % len(self.teams)]
        self.index += 1
        return team
    
    def reset(self):
        """Reset the generator index."""
        self.index = 0
        self.reset_count += 1
