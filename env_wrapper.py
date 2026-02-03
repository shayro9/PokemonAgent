import numpy as np
from poke_env.environment import SinglesEnv


class PokemonRLWrapper(SinglesEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_hp = 1.0

    def embed_battle(self, battle):
        my_hp = battle.active_pokemon.current_hp_fraction
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction

        my_status = 1.0 if battle.active_pokemon.status else 0.0
        opp_status = 1.0 if battle.opponent_active_pokemon.status else 0.0

        moves_attack = -np.ones(4)
        moves_accuracy = -np.ones(4)
        moves_type = -np.ones(4)

        for i, move in enumerate(battle.available_moves):
            moves_attack[i] = move.base_power / 100
            moves_accuracy[i] = move.accuracy / 100
            moves_type[i] = move.type.value

        state = np.concatenate([
            [my_hp, my_status],
            [opp_hp, opp_status],
            moves_attack,
            moves_accuracy,
            moves_type
        ]).astype(np.float32)

        return state

    def calc_reward(self, battle) -> float:
        reward = 0
        current_hp = battle.opponent_active_pokemon.current_hp_fraction
        if self.last_hp is not None:
            reward += self.last_hp - current_hp

        self.last_hp = current_hp

        return reward
