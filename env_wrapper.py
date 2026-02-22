import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data import GenData
import gymnasium as gym


class PokemonRLWrapper(SinglesEnv):
    def __init__(
            self,
            *,
            team,
            opponent_teams: list[str] | None,
            rounds_per_opponents: int = 2_000,
            battle_team_generator=None,
            agent_team_generator=None,
            opponent_team_generator=None,
            **kwargs,
    ):
        super().__init__(
            team=team,
            **kwargs
        )
        self.opponent_teams = opponent_teams or []
        self.battle_team_generator = battle_team_generator
        self.agent_team_generator = agent_team_generator
        self.opponent_team_generator = opponent_team_generator
        self._last_team_update_round = None

        self._action_space = gym.spaces.Discrete(4)
        self.action_spaces = {
            agent: self._action_space for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=-1.0, high=1.0, shape=(51,), dtype=np.float32
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
            if self.battle_team_generator is not None:
                agent1_team, agent2_team = next(self.battle_team_generator)
                self.agent1.update_team(agent1_team)
                self.agent2.update_team(agent2_team)
            else:
                if self.agent_team_generator is not None:
                    self.agent1.update_team(next(self.agent_team_generator))

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

        my_stats = np.minimum(np.array(list(my.stats.values())) / 255.0, 1.0)
        opp_boosts = np.array(list(opp.boosts.values())) / 6.0
        opp_base_stats = np.array(list(opp.base_stats.values())) / 255.0

        my_types = my.types
        opp_types = opp.types
        types_multipliers = calc_types_vector(my_types, opp_types, battle.gen)

        bucket = int(my.weight // opp.weight)
        bucket = max(0, min(bucket, 5))
        weight_one_hot = np.zeros(6, dtype=np.float32)
        weight_one_hot[bucket] = 1.0

        opp_preparing = float(opp.preparing)

        state = np.concatenate([
            [my_hp], my_stats,
            [opp_hp], opp_base_stats, opp_boosts, [opp_preparing],
            types_multipliers,
            weight_one_hot,
        ]).astype(np.float32)

        return state

    def print_state(self, battle, *, prefix="[PokemonRLWrapper]"):
        state = self.embed_battle(battle)

        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        layout = [
            ("my_hp", 1),
            ("my_stats", len(my.stats)),

            ("opp_hp", 1),
            ("opp_base_stats", len(opp.base_stats)),
            ("opp_boosts", len(opp.boosts)),
            ("opp_preparing", 1),

            ("type_multipliers (4)", 4),

            ("weight_bucket (one-hot)", 6),
        ]

        idx = 0
        lines = [f"{prefix} STATE BREAKDOWN"]

        for name, size in layout:
            chunk = state[idx: idx + size]

            if size == 1:
                lines.append(f"  {name:28}: {chunk[0]: .3f}")
            else:
                formatted = ", ".join(f"{x: .3f}" for x in chunk)
                lines.append(f"  {name:28}: [{formatted}]")

            idx += size

        lines.append(f"\n  TOTAL DIMENSIONS: {len(state)}")

        message = "\n".join(lines)
        print(message)
        return message

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


def calc_types_vector(my_types: list[PokemonType], opp_types: list[PokemonType], gen: int):
    vec = []

    my_types = list(my_types) + [None]
    opp_types = list(opp_types) + [None]

    my_types = my_types[:2]
    opp_types = opp_types[:2]

    for my_t in my_types:
        for opp_t in opp_types:
            if my_t is None or opp_t is None:
                vec.append(0.0)  # neutral
            else:
                mult = my_t.damage_multiplier(opp_t, type_chart=GenData.from_gen(gen).type_chart)
                vec.append(-1.0 if mult == 0.0 else float(np.log2(mult)))

    return np.array(vec, dtype=np.float32)
