import time

import gymnasium
import numpy as np
from poke_env import RandomPlayer, LocalhostServerConfiguration, AccountConfiguration, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.environment import SingleAgentWrapper

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from env.battle_config import BattleConfig
from env.singles_env_wrapper import PokemonRLWrapper


class _PokemonEnvBridge(gymnasium.Wrapper):
    """Bridges SingleAgentWrapper to expose PokemonRLWrapper methods to SB3.

    ``SingleAgentWrapper`` is a plain ``gymnasium.Env`` (not a ``Wrapper``), so
    ``get_wrapper_attr`` and attribute delegation stop at it.  This wrapper sits
    on top and explicitly forwards the two methods that SB3/logging need so that
    ``SubprocVecEnv.env_method`` and direct calls both work.
    """

    def action_masks(self) -> np.ndarray:
        return self.env.env.action_masks()

    def get_last_battle(self):
        return self.env.env.get_last_battle()


def build_env(
        battle_format: str,
        opponent_generator,
        rounds_per_opponent: int,
        agent_team_generator=None,
        battle_team_generator=None,
        strict: bool = True,
        battle_config: BattleConfig | None = None,
        worker_id: int = 0,
) -> _PokemonEnvBridge:
    """Construct the single-agent battle environment.

    Action masking is handled natively by ``PokemonRLWrapper.action_masks()``,
    which ``MaskablePPO`` discovers via the gymnasium wrapper chain — no
    ``ActionMasker`` wrapper is needed.
    
    :param strict: If true, action-order converters will throw an error if the move is
            illegal. Otherwise, it will return default. Defaults to True.
    :param battle_format: Showdown battle format name.
    :param opponent_generator: Optional generator for opponent teams.
    :param rounds_per_opponent: Battles played before rotating opponent teams.
    :param agent_team_generator: Optional generator for agent teams.
    :param battle_team_generator: Optional generator yielding both teams.
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
    return _PokemonEnvBridge(env)


def build_vec_env(
        n_envs: int,
        battle_format: str,
        opponent_generator,
        rounds_per_opponent: int,
        agent_team_generator=None,
        battle_team_generator=None,
        strict: bool = True,
        battle_config: BattleConfig | None = None,
) -> VecMonitor:
    """Construct a vectorized environment with ``n_envs`` parallel workers.

    Each worker runs in its own subprocess with its own asyncio event loop and
    Pokémon Showdown connections, enabling true parallel rollout collection.

    :param n_envs: Number of parallel environment workers.
    :param battle_format: Showdown battle format name.
    :param opponent_generator: Optional generator for opponent teams.
    :param rounds_per_opponent: Battles played before rotating opponent teams.
    :param agent_team_generator: Optional generator for agent teams.
    :param battle_team_generator: Optional generator yielding both teams.
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
                strict=strict,
                battle_config=battle_config,
                worker_id=worker_id,
            )
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    return VecMonitor(vec_env)
