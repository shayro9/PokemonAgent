import numpy as np
from poke_env.environment import SinglesEnv
import gymnasium as gym


class PokemonRLWrapper(SinglesEnv):
    def __init__(
        self,
        *,
        team,
        opponent_teams: list[str] | None,
        rounds_per_opponents: int = 2_000,
        opponent_team_generator=None,
        **kwargs,
    ):
        super().__init__(
            team=team,
            **kwargs
        )
        self.opponent_teams = opponent_teams or []
        self.opponent_team_generator = opponent_team_generator
        self._last_team_update_round = None

        self._action_space = gym.spaces.Discrete(4)
        self.action_spaces = {
            agent: self._action_space for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=-1.0, high=1.0, shape=(25,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.last_hp = {}
        self.last_fainted = {}
        self.rounds_played = 0
        self.rounds_per_opponents = rounds_per_opponents

    def action_to_order(self, action, battle, fake=False, strict=True):
        real_action = action + 6 if action < 4 else action
        return super().action_to_order(real_action, battle, fake, strict)

    def reset(self, *args, **kwargs):
        self.last_hp = {}
        self.last_fainted = {}
        if (
            self.rounds_played % self.rounds_per_opponents == 0
            and self._last_team_update_round != self.rounds_played
        ):
            if self.opponent_team_generator is not None:
                self.agent2.update_team(next(self.opponent_team_generator))
            elif self.opponent_teams:
                i = (self.rounds_played // self.rounds_per_opponents) % len(self.opponent_teams)
                self.agent2.update_team(self.opponent_teams[i])
            self._last_team_update_round = self.rounds_played
        return super().reset(*args, **kwargs)

    def embed_battle(self, battle):
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_hp = my.current_hp_fraction
        opp_hp = opp.current_hp_fraction

        my_status = float(my.status is not None)
        opp_status = float(opp.status is not None)
        opp_preparing = float(opp.preparing)

        my_boosts = np.array(list(my.boosts.values())) / 6.0
        opp_boosts = np.array(list(opp.boosts.values())) / 6.0

        bucket = my.weight // opp.weight
        bucket = max(0, min(bucket, 5))
        weight_one_hot = np.zeros(6, dtype=np.float32)
        weight_one_hot[bucket] = 1.0

        state = np.concatenate([
            [my_hp, my_status], my_boosts,
            [opp_hp, opp_status, opp_preparing], opp_boosts,
            weight_one_hot
        ]).astype(np.float32)

        return state

    def calc_reward(self, battle) -> float:
        if battle.player_username != self.agent1.username:
            return 0.0

        my_hp = battle.active_pokemon.current_hp_fraction
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction

        last_my_hp, last_opp_hp = self.last_hp.get(
            battle, (1.0, 1.0)
        )

        damage_to_opp = last_opp_hp - opp_hp
        damage_to_me = last_my_hp - my_hp
        reward = damage_to_opp - damage_to_me
        self.last_hp[battle] = (my_hp, opp_hp)

        if battle.finished:
            self.rounds_played += 1
            if battle.won:
                reward += 5.0
            elif battle.lost:
                reward -= 5.0

        reward = np.clip(reward, -1.0, 1.0)

        return reward
