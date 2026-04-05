from collections import deque

import wandb
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from poke_env.environment import SingleAgentWrapper

from env.singles_env_wrapper import PokemonRLWrapper
from training.config import LOG_FREQ


class BattleMetricsCallback(BaseCallback):
    def __init__(self, env, log_freq: int = 100, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.log_freq = log_freq
        self._episode_rewards = []
        self._episode_lengths = []
        self._results = deque(maxlen=LOG_FREQ)
        self._switch_actions = deque(maxlen=LOG_FREQ)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        actions = self.locals.get("actions", [])

        for i, info in enumerate(infos):
            # Track raw action values for distribution logging
            if i < len(actions):
                action = actions[i]
                self._switch_actions.append(int(action))

            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

                battle = self._get_battle(env_idx=i)
                if battle is not None:
                    if battle.won:
                        self._results.append(1)
                    elif battle.lost:
                        self._results.append(0)
                    else:
                        self._results.append(0.5)

        if self.n_calls % self.log_freq == 0 and self._episode_rewards:
            action_list = list(self._switch_actions)
            switch_rate = (
                sum(1 for a in action_list if 0 <= a <= 5) / len(action_list)
                if action_list else 0.0
            )

            wandb.log({
                "win_rate": np.mean(self._results) if self._results else 0.0,
                "mean_episode_reward": np.mean(self._episode_rewards[-50:]),
                "mean_episode_length": np.mean(self._episode_lengths[-50:]),
                "action_distribution": wandb.Histogram(action_list) if action_list else wandb.Histogram([0]),
                "switch_action_rate": switch_rate,
            }, step=self.num_timesteps)

        return True

    def _get_battle(self, env_idx: int = 0):
        env = self.env
        if isinstance(env, VecEnv):
            battles = env.env_method("get_last_battle")
            return battles[env_idx] if env_idx < len(battles) else None
        while hasattr(env, "env"):
            env = env.env
        if isinstance(env, PokemonRLWrapper):
            return env.get_last_battle()
        return None
