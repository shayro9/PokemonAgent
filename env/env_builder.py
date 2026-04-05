import time

from poke_env import RandomPlayer, LocalhostServerConfiguration, AccountConfiguration, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.environment import SingleAgentWrapper

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from env.battle_config import BattleConfig
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
        battle_config: BattleConfig | None = None,
        worker_id: int = 0,
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
    :param battle_config: Generation config. Defaults to Gen 1.
    :param worker_id: Index used to ensure unique account names across parallel workers.
    :returns: A configured ``SingleAgentWrapper`` environment."""

    unique_id = f"{int(time.time() * 1000) % 100000}_{worker_id}"

    opponent_policy = MaxBasePowerPlayer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"MaxPower_{unique_id}", None),
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
        battle_config=battle_config,
    )

    env = SingleAgentWrapper(agent, opponent_policy)
    env = _wrap_action_masker(env, enabled=use_action_masking)
    return env


def build_vec_env(
        n_envs: int,
        battle_format: str,
        opponent_generator,
        rounds_per_opponent: int,
        agent_team_generator=None,
        battle_team_generator=None,
        use_action_masking: bool = False,
        strict: bool = True,
        battle_config: BattleConfig | None = None,
) -> SubprocVecEnv:
    """Construct a vectorized environment with ``n_envs`` parallel workers.

    Each worker runs in its own subprocess with its own asyncio event loop and
    Pokémon Showdown connections, enabling true parallel rollout collection.

    :param n_envs: Number of parallel environment workers.
    :param battle_format: Showdown battle format name.
    :param opponent_generator: Optional generator for opponent teams.
    :param rounds_per_opponent: Battles played before rotating opponent teams.
    :param agent_team_generator: Optional generator for agent teams.
    :param battle_team_generator: Optional generator yielding both teams.
    :param use_action_masking: Whether to wrap each env with ``ActionMasker``.
    :param battle_config: Generation config. Defaults to Gen 1.
    :returns: A ``SubprocVecEnv`` wrapping ``n_envs`` independent environments."""

    def make_env(worker_id: int):
        def _init():
            set_random_seed(worker_id)
            return build_env(
                battle_format=battle_format,
                opponent_generator=opponent_generator,
                rounds_per_opponent=rounds_per_opponent,
                agent_team_generator=agent_team_generator,
                battle_team_generator=battle_team_generator,
                use_action_masking=use_action_masking,
                strict=strict,
                battle_config=battle_config,
                worker_id=worker_id,
            )
        return _init

    return SubprocVecEnv([make_env(i) for i in range(n_envs)])
