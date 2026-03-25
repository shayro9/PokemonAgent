import time

from poke_env import RandomPlayer, LocalhostServerConfiguration, AccountConfiguration, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.environment import SingleAgentWrapper

from sb3_contrib.common.wrappers import ActionMasker

from env.singles_env_wrapper import PokemonRLWrapper


def _wrap_action_masker(env, *, enabled: bool):
    """Attach an action-mask wrapper when masking is enabled.
    
    :param env: Environment instance to potentially wrap.
    :param enabled: Whether action masking should be applied.
    :returns: The original environment or an ``ActionMasker``-wrapped environment."""
    if not enabled:
        return env

    def mask_fn(e):
        base = e
        if hasattr(base, "unwrapped"):
            base = base.unwrapped

        if hasattr(base, "env") and hasattr(base.env, "unwrapped"):
            base = base.env.unwrapped

        return base.action_masks()

    return ActionMasker(env, mask_fn)


def build_env(
        battle_format: str,
        opponent_generator,
        rounds_per_opponent: int,
        agent_team_generator=None,
        battle_team_generator=None,
        use_action_masking: bool = False,
        strict: bool = True,
) -> SingleAgentWrapper:
    """Construct the single-agent battle environment.
    
    :param strict: If true, action-order converters will throw an error if the move is
            illegal. Otherwise, it will return default. Defaults to True.
    :param battle_format: Showdown battle format name.
    :param opponent_generator: Optional generator for opponent teams.
    :param rounds_per_opponent: Battles played before rotating opponent teams.
    :param agent_team_generator: Optional generator for agent teams.
    :param battle_team_generator: Optional generator yielding both teams.
    :param use_action_masking: Whether to wrap the env with ``ActionMasker``.
    :returns: A configured ``SingleAgentWrapper`` environment."""

    unique_id = int(time.time() * 1000) % 100000

    opponent_policy = SimpleHeuristicsPlayer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    agent = PokemonRLWrapper(
        battle_format=battle_format,
        battle_team_generator=battle_team_generator,
        agent_team_generator=agent_team_generator,
        opponent_team_generator=opponent_generator,
        rounds_per_opponents=rounds_per_opponent,
        server_configuration=LocalhostServerConfiguration,
        account_configuration1=AccountConfiguration(f"Player_{unique_id}", None),
        account_configuration2=AccountConfiguration(f"Opponent_{unique_id}", None),
        strict=strict,
    )

    env = SingleAgentWrapper(agent, opponent_policy)
    env = _wrap_action_masker(env, enabled=use_action_masking)
    return env
