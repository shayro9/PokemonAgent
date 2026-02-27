from poke_env.player import RandomPlayer, Player
import numpy as np
from stable_baselines3 import DQN
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.battle import Battle


class DebugRLPlayer(RandomPlayer):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env

    def choose_move(self, battle: Battle):
        # self.env.print_state(battle)
        # print("=" * 50)
        # mask = self.env.action_masks()
        # print(mask)
        # print("=" * 50)
        # print(battle.active_pokemon.identifier('p1'))
        # print(battle.active_pokemon.identifier('p2'))
        # battle.opponent_active_pokemon.stats = battle.opponent_active_pokemon.base_stats
        # print(calculate_damage(battle.active_pokemon.identifier('p1'), battle.opponent_active_pokemon.identifier('p2'), battle.available_moves[0], battle))
        print(battle.available_moves)

        return self.choose_random_move(battle)
