import numpy as np
from poke_env.battle import Move
from poke_env.environment import SinglesEnv
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data import GenData
from embedding import *
import gymnasium as gym

MAX_MOVES = 4
MOVE_EMBED_LEN = 4 + len(MoveCategory) + 1 + len(Status) + 7 + 7 + 2 + 2

ACTION_DEFAULT = -2
ACTION_FORFEIT = -1
ACTION_SWITCH_RANGE = range(0, 6)
ACTION_MOVE_RANGE = range(6, 10)
ACTION_MEGA_RANGE = range(10, 14)
ACTION_ZMOVE_RANGE = range(14, 18)
ACTION_DYNAMAX_RANGE = range(18, 22)
ACTION_TERASTALLIZE_RANGE = range(22, 26)


def _slot_is_available(sequence, slot: int) -> bool:
    return 0 <= slot < len(sequence)


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
                low=-1.0, high=1.0, shape=(163,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.last_hp = {}
        self.last_fainted = {}
        self.rounds_played = 0
        self.rounds_per_opponents = rounds_per_opponents

    def get_valid_action_mask(
            self,
            battle,
            *,
            allow_switches: bool = True,
            allow_moves: bool = True,
            allow_mega: bool = True,
            allow_zmove: bool = True,
            allow_dynamax: bool = True,
            allow_terastallize: bool = True,
    ) -> np.ndarray:
        """Returns a mask over the canonical action space [0..25]."""
        mask = np.zeros(26, dtype=bool)

        if allow_switches:
            available_switches = getattr(battle, "available_switches", [])
            for slot, action in enumerate(ACTION_SWITCH_RANGE):
                mask[action] = _slot_is_available(available_switches, slot)

        if allow_moves:
            available_moves = getattr(battle, "available_moves", [])
            for slot, action in enumerate(ACTION_MOVE_RANGE):
                mask[action] = _slot_is_available(available_moves, slot)

            if allow_mega and getattr(battle, "can_mega_evolve", False):
                for slot, action in enumerate(ACTION_MEGA_RANGE):
                    mask[action] = _slot_is_available(available_moves, slot)

            if allow_zmove and getattr(battle, "can_z_move", False):
                available_z_moves = getattr(battle, "available_z_moves", [])
                for slot, action in enumerate(ACTION_ZMOVE_RANGE):
                    mask[action] = _slot_is_available(available_z_moves, slot)

            if allow_dynamax and getattr(battle, "can_dynamax", False):
                for slot, action in enumerate(ACTION_DYNAMAX_RANGE):
                    mask[action] = _slot_is_available(available_moves, slot)

            if allow_terastallize and getattr(battle, "can_tera", False):
                for slot, action in enumerate(ACTION_TERASTALLIZE_RANGE):
                    mask[action] = _slot_is_available(available_moves, slot)

        return mask

    def action_to_order(self, action, battle, fake=False, strict=True):
        # Current training setup is 1v1 move-only: [0..3] maps to canonical move actions [6..9].
        canonical_action = int(action) + ACTION_MOVE_RANGE.start if int(action) < 4 else int(action)
        valid_mask = self.get_valid_action_mask(
            battle,
            allow_switches=False,
            allow_moves=True,
            allow_mega=False,
            allow_zmove=False,
            allow_dynamax=False,
            allow_terastallize=False,
        )

        if not (0 <= canonical_action < len(valid_mask)) or not valid_mask[canonical_action]:
            canonical_action = ACTION_DEFAULT

        return super().action_to_order(canonical_action, battle, fake, strict)

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

        my_moves = np.zeros(MAX_MOVES * MOVE_EMBED_LEN, dtype=np.float32)

        for i, move in enumerate(battle.available_moves[:MAX_MOVES]):
            emb = embed_move(move, opp_types, battle.gen)
            my_moves[i * MOVE_EMBED_LEN: (i + 1) * MOVE_EMBED_LEN] = emb

        opp_preparing = float(opp.preparing)

        state = np.concatenate([
            [my_hp], my_stats,
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
