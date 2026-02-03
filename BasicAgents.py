from poke_env.player import RandomPlayer
from env_wrapper import PokemonRLWrapper
import numpy as np


class DebugRLPlayer(PokemonRLWrapper, RandomPlayer):
    def __init__(self, *args, **kwargs):
        PokemonRLWrapper.__init__(self)
        RandomPlayer.__init__(self, *args, **kwargs)

    def choose_move(self, battle):
        # Embed state
        state = self.embed_battle(battle)

        # Compute reward
        reward = self.calc_reward(battle)

        print("STATE:", np.round(state, 3))
        print("REWARD:", reward)
        print("-" * 50)

        # Act randomly
        return self.choose_random_move(battle)
