import numpy as np
from poke_env.battle import Move
from poke_env.environment import SinglesEnv
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data import GenData
import gymnasium as gym
from action_masking import *


def _sample_valid_action(mask: np.ndarray, default: int) -> int:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return int(default)
    return int(np.random.choice(valid))


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
        self._latest_battle = None

        self._action_space = gym.spaces.Discrete(26)
        self.action_spaces = {
            agent: self._action_space for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=-1.0, high=1.0, shape=(171,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.last_hp = {}
        self.last_fainted = {}
        self.rounds_played = 0
        self.rounds_per_opponents = rounds_per_opponents

    def _is_agent1_battle(self, battle) -> bool:
        return getattr(battle, "player_username", None) == self.agent1.username

    def action_to_order(self, action, battle, fake=False, strict=True):
        if not self._is_agent1_battle(battle=battle):
            return super().action_to_order(action, battle, fake, strict)

        canonical_action = action

        mask = get_valid_action_mask(
            battle=battle,
            allow_switches=False,
            allow_moves=True,
            allow_mega=False,
            allow_zmove=False,
            allow_dynamax=False,
            allow_terastallize=False,
        )

        if not (0 <= canonical_action < len(mask)):
            if strict:
                raise ValueError(f"Action {canonical_action} out of bounds for action space size {len(mask)}.")
            canonical_action = ACTION_DEFAULT

        # Validity check
        elif not mask[canonical_action]:
            if strict:
                raise ValueError(
                    f"Invalid action {canonical_action} selected. Valid actions: {np.flatnonzero(mask).tolist()}")
            canonical_action = ACTION_DEFAULT

        return super().action_to_order(canonical_action, battle, fake, strict)

    def action_masks(self) -> np.ndarray:
        """MaskablePPO-compatible action mask for the latest observed battle state."""
        battle = self._latest_battle

        if battle is None:
            # Conservative fallback for the first call before embed_battle runs.
            return np.ones(self._action_space.n, dtype=bool)

        action_masks = get_valid_action_mask(battle)
        return action_masks

    def reset(self, *args, **kwargs):
        self.last_hp = {}
        self.last_fainted = {}
        self._latest_battle = None
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
        if self._is_agent1_battle(battle):
            self._latest_battle = battle

        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_hp = my.current_hp_fraction
        opp_hp = opp.current_hp_fraction

        my_status = embed_status(my.status)
        opp_status = embed_status(opp.status)

        my_stats = np.minimum(np.array(list(my.stats.values())) / 255.0, 1.0)
        my_boosts = np.array(list(my.boosts.values())) / 6.0

        opp_boosts = np.array(list(opp.boosts.values())) / 6.0
        opp_base_stats = np.array(list(opp.base_stats.values())) / 255.0

        my_types = my.types
        opp_types = opp.types
        types_multipliers = calc_types_vector(my_types, opp_types, battle.gen)

        bucket = int(my.weight // opp.weight)
        bucket = max(0, min(bucket, 5))
        weight_one_hot = np.zeros(6, dtype=np.float32)
        weight_one_hot[bucket] = 1.0

        my_moves = np.zeros(MAX_MOVES * MOVE_EMBED_LEN, dtype=np.float32)

        for i, move in enumerate(battle.available_moves[:MAX_MOVES]):
            emb = embed_move(move, opp_types, battle.gen)
            my_moves[i * MOVE_EMBED_LEN: (i + 1) * MOVE_EMBED_LEN] = emb

        opp_preparing = float(opp.preparing)

        state = np.concatenate([
            [my_hp], my_stats, my_boosts,
            [opp_hp], opp_base_stats, opp_boosts, [opp_preparing],
            my_moves,
            types_multipliers,
            weight_one_hot,
        ]).astype(np.float32)

        return state

    def print_state(self, battle, *, prefix="[PokemonRLWrapper]"):
        state = self.embed_battle(battle)

        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        move_block = [
            ("category", 3),
            ("multiplier", 1),
            ("status", 7),
            ("boosts_target", 7),
            ("boosts_self", 7),
            ("recoil", 1),
            ("drain", 1),
            ("multi_hit", 2),
        ]

        layout = [
            ("my_hp", 1),
            ("my_stats", len(my.stats)),
            ("my_boosts"), len(my.boosts),

            ("opp_hp", 1),
            ("opp_base_stats", len(opp.base_stats)),
            ("opp_boosts", len(opp.boosts)),
            ("opp_preparing", 1),
        ]

        for i in range(1, 5):
            layout.append((f"move{i}", 4))
            layout.extend(move_block)

        # Add final features
        layout.extend([
            ("type_multipliers (4)", 4),
            ("weight_bucket (one-hot)", 6),
        ])

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
            battle.battle_tag, (1.0, 1.0)
        )

        damage_to_opp = last_opp_hp - opp_hp
        damage_to_me = last_my_hp - my_hp
        reward = damage_to_opp - damage_to_me
        self.last_hp[battle.battle_tag] = (my_hp, opp_hp)

        reward = np.clip(reward, -1.0, 1.0)

        if battle.finished:
            self.rounds_played += 1
            if battle.won:
                reward += 5.0
            elif battle.lost:
                reward -= 5.0

        return reward
