import asyncio

from poke_env.concurrency import POKE_LOOP
from poke_env.player import RandomPlayer, Player
import numpy as np
from stable_baselines3 import DQN
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.battle import Battle

from env.embed import embed_move
from env.singles_env_wrapper import print_state, PokemonRLWrapper


class DebugRLPlayer(RandomPlayer):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env

    def choose_move(self, battle: Battle):
        print_state(battle, prefix="[DebugRLPlayer]")
        print("=" * 50)
        # mask = self.env.action_masks()
        # print(mask)
        # print("=" * 50)
        # print(battle.active_pokemon.identifier('p1'))
        # print(battle.active_pokemon.identifier('p2'))
        # battle.opponent_active_pokemon.stats = battle.opponent_active_pokemon.base_stats
        # print(calculate_damage(battle.active_pokemon.identifier('p1'), battle.opponent_active_pokemon.identifier('p2'), battle.available_moves[0], battle))

        types = battle.opponent_active_pokemon.types
        my_types = battle.active_pokemon.types
        gen = battle.gen
        for move in battle.available_moves:
            print("type - {}"
                  .format(move.type))
            print(embed_move(move, types, my_types, gen))

        return self.choose_random_move(battle)
