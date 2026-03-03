from collections import deque

import wandb
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from poke_env.environment import SingleAgentWrapper

from env.singles_env_wrapper import PokemonRLWrapper


class BattleMetricsCallback(BaseCallback):
    def __init__(self, env: SingleAgentWrapper, log_freq: int = 100, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.log_freq = log_freq
        self._episode_rewards = []
        self._episode_lengths = []
        self._results = deque(maxlen=100)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

                battle = self._get_battle()
                if battle is not None:
                    if battle.won:
                        self._results.append(1)
                    elif battle.lost:
                        self._results.append(0)
                    else:
                        self._results.append(0.5)

        if self.n_calls % self.log_freq == 0 and self._episode_rewards:
            wandb.log({
                "win_rate": np.mean(self._results) if self._results else 0.0,
                "mean_episode_reward": np.mean(self._episode_rewards[-50:]),
                "mean_episode_length": np.mean(self._episode_lengths[-50:]),
            }, step=self.num_timesteps)

        return True

    def _get_battle(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        if isinstance(env, PokemonRLWrapper):
            battle = env.get_last_battle()
            return battle
        return None
