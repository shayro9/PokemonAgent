from weakref import WeakKeyDictionary

import numpy as np
import gymnasium as gym
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SinglesEnv

from env.battle_config import BattleConfig
from env.action_mask_gen_1 import ActionMaskGen1
from env.reward import get_state_value_optimizable
from teams.generators import InfinitePoolGenerator

def print_state(battle, *, battle_config: BattleConfig | None = None, prefix="[PokemonRLWrapper]") -> str:
    """Render and print a human-readable battle state snapshot.
    
    :param battle: Battle object to describe.
    :param battle_config: BattleConfig to use for state construction. Defaults to Gen 1.
    :param prefix: Message prefix shown before the state details.
    :returns: The formatted state message that was printed."""
    cfg = battle_config if battle_config is not None else BattleConfig.gen1()
    state_obj = cfg.battle_state_cls(battle)
    message = f"{prefix}\n" + state_obj.describe()
    print(message)
    return message


class PokemonRLWrapper(SinglesEnv):
    def __init__(
            self,
            *,
            rounds_per_opponents: int = 2_000,
            battle_team_generator: InfinitePoolGenerator | None = None,
            agent_team_generator: InfinitePoolGenerator | None = None,
            opponent_team_generator: InfinitePoolGenerator | None = None,
            battle_config: BattleConfig | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._battle_config = battle_config if battle_config is not None else BattleConfig.gen1()
        self.battle_team_generator = battle_team_generator
        self.agent_team_generator = agent_team_generator
        self.opponent_team_generator = opponent_team_generator
        self._last_team_update_round = None
        self._last_finished_battle = None

        self._action_space = gym.spaces.Discrete(self._battle_config.action_space_size)
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(self._battle_config.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self._reward_buffer: WeakKeyDictionary[AbstractBattle, float] = (
            WeakKeyDictionary()
        )
        self._obs_cache: WeakKeyDictionary[AbstractBattle, np.ndarray] = (
            WeakKeyDictionary()
        )
        self.action_mask = ActionMaskGen1()

        self.rounds_played: int = 0
        self.rounds_per_opponents = rounds_per_opponents

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def action_to_order(self, action, battle, fake=False, strict=True):
        if not self._is_player_turn(battle=battle):
            return super().action_to_order(action, battle, fake, strict)

        canonical_action = action
        mask = self.action_mask.get_mask()

        if not (0 <= canonical_action < len(mask)):
            if strict:
                raise ValueError(f"Action {canonical_action} out of bounds ({len(mask)}).")
            canonical_action = self.action_mask.ACTION_DEFAULT
        elif not mask[canonical_action]:
            if strict:
                raise ValueError(
                    f"Invalid action {canonical_action}. Valid: {np.flatnonzero(mask).tolist()}"
                )
            canonical_action = self.action_mask.ACTION_DEFAULT

        try:
            return super().action_to_order(canonical_action, battle, fake, strict)
        except ValueError as e:
            print(self._battle_config.battle_state_cls(battle).describe())
            print(mask)
            print(f"Error converting action {canonical_action} to order: {e}")
            return super().action_to_order(canonical_action, battle, fake, strict=False)

    def action_masks(self) -> np.ndarray:
        return self.action_mask.get_mask()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def embed_battle(self, battle: Battle) -> np.ndarray:
        if not self._is_player_turn(battle):
            return self._obs_cache.get(
                battle,
                np.zeros(self._battle_config.obs_dim, dtype=np.float32),
            )

        mask = self.get_action_mask(battle)
        self.action_mask.set(mask)

        obs = self._battle_config.battle_state_cls(battle).to_array()
        self._obs_cache[battle] = obs
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def calc_reward(self, battle) -> float:
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = 0.0

        value = get_state_value_optimizable(battle)
        reward = value - self._reward_buffer[battle]
        self._reward_buffer[battle] = value

        if battle.finished and self._is_player_turn(battle):
            self.rounds_played += 1
            self._last_finished_battle = battle

        return reward

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self.action_mask.reset()
        if (
                self.rounds_played % self.rounds_per_opponents == 0
                and self._last_team_update_round != self.rounds_played
        ):
            if self.battle_team_generator is not None:
                agent1_team, agent2_team = next(self.battle_team_generator)
                self.agent1.update_team(agent1_team)
                self.agent2.update_team(agent2_team)
            else:
                if self.agent_team_generator is not None:
                    self.agent1.update_team(next(self.agent_team_generator))
                if self.opponent_team_generator is not None:
                    self.agent2.update_team(next(self.opponent_team_generator))
            self._last_team_update_round = self.rounds_played
        return super().reset(*args, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_player_turn(self, battle) -> bool:
        return getattr(battle, "player_username", None) == self.agent1.username

    def get_last_battle(self):
        return self._last_finished_battle
