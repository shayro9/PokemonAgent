from poke_env.player import RandomPlayer, Player
import numpy as np
from stable_baselines3 import DQN


class DebugRLPlayer(RandomPlayer):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env

    def choose_move(self, battle):
        self.env.print_state(battle)
        print("=" * 50)

        return self.choose_random_move(battle)