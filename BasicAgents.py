from poke_env.player import RandomPlayer, Player
import numpy as np
from stable_baselines3 import DQN


class DebugRLPlayer(RandomPlayer):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env

    def choose_move(self, battle):
        state = self.env.embed_battle(battle)
        reward = self.env.calc_reward(battle)

        print("STATE:", np.round(state, 3))
        print("REWARD:", reward)
        print("-" * 50)

        return self.choose_random_move(battle)


class FrozenSB3Player(Player):
    def __init__(self, env, model_path, eps=0.05, *args, **kwargs):
        kwargs.setdefault("start_listening", False)
        super().__init__(*args, **kwargs)
        self.env = env
        self.model_path = model_path
        self.model = DQN.load(model_path)
        self.eps = eps

    def reload(self):
        self.model = DQN.load(self.model_path)

    def choose_move(self, battle):
        obs = self.env.embed_battle(battle)
        action, _ = self.model.predict(obs, deterministic=True)
        order = self.env.action_to_order(np.int64(action), battle, fake=self.env.fake, strict=self.env.strict)
        return order
