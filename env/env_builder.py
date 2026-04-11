import time

import gymnasium
import numpy as np
from poke_env import LocalhostServerConfiguration, AccountConfiguration
from poke_env.environment import SingleAgentWrapper

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from curriculum.models import OpponentPlayerSpec
from curriculum.registry import build_opponent_player
from env.battle_config import BattleConfig
from env.singles_env_wrapper import PokemonRLWrapper


class _PokemonEnvBridge(gymnasium.Wrapper):
    """Bridges SingleAgentWrapper to expose PokemonRLWrapper methods to SB3.

    ``SingleAgentWrapper`` is a plain ``gymnasium.Env`` (not a ``Wrapper``), so
    ``get_wrapper_attr`` and attribute delegation stop at it.  This wrapper sits
    on top and explicitly forwards the two methods that SB3/logging need so that
    ``SubprocVecEnv.env_method`` and direct calls both work.
    """

    def __init__(
        self,
        env: SingleAgentWrapper,
        *,
        battle_format: str,
        unique_id: str,
        opponent_player_spec: OpponentPlayerSpec,
    ):
        super().__init__(env)
        self._battle_format = battle_format
        self._unique_id = unique_id
        self._current_opponent_player_spec = opponent_player_spec
        self._pending_opponent_player_spec: OpponentPlayerSpec | None = None
        self._opponent_player_revision = 1

    def action_masks(self) -> np.ndarray:
        return self.env.env.action_masks()

    def get_last_battle(self):
        return self.env.env.get_last_battle()

    def get_opponent_player_spec(self) -> OpponentPlayerSpec:
        """Return the currently active opponent-player spec."""
        return self._current_opponent_player_spec

    def schedule_opponent_player(self, opponent_player_spec: OpponentPlayerSpec):
        """Queue an opponent policy change for the next environment reset."""
        if opponent_player_spec == self._current_opponent_player_spec:
            self._pending_opponent_player_spec = None
            return
        self._pending_opponent_player_spec = opponent_player_spec

    def reset(self, *args, **kwargs):
        self._maybe_swap_opponent_player()
        return super().reset(*args, **kwargs)

    def _maybe_swap_opponent_player(self):
        if self._pending_opponent_player_spec is None:
            return

        self.env.opponent = build_opponent_player(
            self._pending_opponent_player_spec,
            battle_format=self._battle_format,
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration(
                f"Opp_{self._unique_id}_{self._opponent_player_revision}",
                None,
            ),
        )
        self._opponent_player_revision += 1
        self._current_opponent_player_spec = self._pending_opponent_player_spec
        self._pending_opponent_player_spec = None


DEFAULT_OPPONENT_PLAYER_SPEC = OpponentPlayerSpec(id="max-power")


def build_env(
        battle_format: str,
        opponent_generator,
        rounds_per_opponent: int,
        opponent_player_spec: OpponentPlayerSpec | None = None,
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
    :param opponent_player_spec: Optional curriculum-controlled opponent Player spec.
    :param agent_team_generator: Optional generator for agent teams.
    :param battle_team_generator: Optional generator yielding both teams.
    :param battle_config: Generation config. Defaults to Gen 1.
    :param worker_id: Index used to ensure unique account names across parallel workers.
    :returns: A configured ``SingleAgentWrapper`` environment."""

    unique_id = f"{int(time.time() * 1000) % 100000}_{worker_id}"
    resolved_opponent_player_spec = opponent_player_spec or DEFAULT_OPPONENT_PLAYER_SPEC
    opponent_policy = build_opponent_player(
        resolved_opponent_player_spec,
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"Opp_{unique_id}_0", None),
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
    return _PokemonEnvBridge(
        env,
        battle_format=battle_format,
        unique_id=unique_id,
        opponent_player_spec=resolved_opponent_player_spec,
    )


def _fork_generator(gen, worker_id: int):
    """Return a seed-offset copy of *gen* for a given worker.

    If *gen* is ``None`` or does not support ``fork`` it is returned as-is.

    :param gen: Generator to fork, or ``None``.
    :param worker_id: Zero-based worker index.
    :returns: Forked generator or the original value.
    """
    if gen is None or not hasattr(gen, "fork"):
        return gen
    return gen.fork(worker_id)


def build_vec_env(
        n_envs: int,
        battle_format: str,
        opponent_generator,
        rounds_per_opponent: int,
        opponent_player_spec: OpponentPlayerSpec | None = None,
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
    :param opponent_player_spec: Optional curriculum-controlled opponent Player spec.
    :param agent_team_generator: Optional generator for agent teams.
    :param battle_team_generator: Optional generator yielding both teams.
    :param battle_config: Generation config. Defaults to Gen 1.
    :returns: A ``SubprocVecEnv`` wrapping ``n_envs`` independent environments."""

    def make_env(worker_id: int):
        forked_opponent = _fork_generator(opponent_generator, worker_id)
        forked_agent = _fork_generator(agent_team_generator, worker_id)
        forked_battle = _fork_generator(battle_team_generator, worker_id)

        def _init():
            set_random_seed(worker_id)
            return build_env(
                battle_format=battle_format,
                opponent_generator=forked_opponent,
                rounds_per_opponent=rounds_per_opponent,
                opponent_player_spec=opponent_player_spec,
                agent_team_generator=forked_agent,
                battle_team_generator=forked_battle,
                strict=strict,
                battle_config=battle_config,
                worker_id=worker_id,
            )
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    return VecMonitor(vec_env)
