import time

from poke_env import RandomPlayer, LocalhostServerConfiguration, AccountConfiguration
from poke_env.environment import SingleAgentWrapper

from sb3_contrib.common.wrappers import ActionMasker

from config.config import TEAM_BY_NAME
from env.singles_env_wrapper import PokemonRLWrapper


def _wrap_action_masker(env, *, enabled: bool):
    """
    If enabled, attach ActionMasker so MaskablePPO (and others) can use masks.
    If disabled, return env unchanged.
    """
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
        agent_team: str,
        battle_format: str,
        opponent_names: list[str],
        opponent_generator,
        rounds_per_opponent: int,
        opponent_pool: list[str] = None,
        agent_team_generator=None,
        battle_team_generator=None,
        use_action_masking: bool = False,
) -> SingleAgentWrapper:
    if not opponent_pool:
        opponent_pool = [TEAM_BY_NAME[name] for name in opponent_names]

    unique_id = int(time.time() * 1000) % 100000

    opponent_policy = RandomPlayer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    agent = PokemonRLWrapper(
        battle_format=battle_format,
        team=agent_team,
        opponent_teams=opponent_pool,
        battle_team_generator=battle_team_generator,
        agent_team_generator=agent_team_generator,
        opponent_team_generator=opponent_generator,
        rounds_per_opponents=rounds_per_opponent,
        server_configuration=LocalhostServerConfiguration,
        account_configuration1=AccountConfiguration(f"Player_{unique_id}", None),
        account_configuration2=AccountConfiguration(f"Opponent_{unique_id}", None),
        strict=True,
    )

    env = SingleAgentWrapper(agent, opponent_policy)
    env = _wrap_action_masker(env, enabled=use_action_masking)
    return env
