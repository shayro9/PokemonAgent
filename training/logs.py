import wandb
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from poke_env.environment import SingleAgentWrapper


class BattleMetricsCallback(BaseCallback):
    def __init__(self, env: SingleAgentWrapper, log_freq: int = 100, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.log_freq = log_freq
        self._episode_rewards = []
        self._episode_lengths = []
        self._wins = 0
        self._losses = 0
        self._draws = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self._episode_rewards.append(ep_reward)
                self._episode_lengths.append(ep_length)

                if ep_reward > 0:
                    self._wins += 1
                elif ep_reward < 0:
                    self._losses += 1
                else:
                    self._draws += 1

        if self.n_calls % self.log_freq == 0 and self._episode_rewards:
            total = self._wins + self._losses + self._draws
            win_rate = self._wins / total if total > 0 else 0.0
            wandb.log({
                "win_rate": win_rate,
                "mean_episode_reward": np.mean(self._episode_rewards[-50:]),
                "mean_episode_length": np.mean(self._episode_lengths[-50:]),
            }, step=self.num_timesteps)

        return True
