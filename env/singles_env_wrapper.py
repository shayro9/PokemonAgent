import numpy as np
import gymnasium as gym
from poke_env.environment import SinglesEnv

from env.action_masking import get_valid_action_mask, ACTION_DEFAULT
from env.battle_state import BattleState, OBS_SIZE
from env.combat_utils import (
    build_protect_belief,
    detect_opponent_move,
    did_no_damage,
    estimate_protect_attempt_prior,
    snapshot_opponent_pp,
)
from env.reward import calc_reward


def print_state(battle, *, prefix="[PokemonRLWrapper]") -> str:
    """Render and print a human-readable battle state snapshot.
    
    :param battle: Battle object to describe.
    :param prefix: Message prefix shown before the state details.
    :returns: The formatted state message that was printed."""
    state_obj = BattleState.from_battle(battle)
    message = f"{prefix}\n" + state_obj.describe()
    print(message)
    return message


class PokemonRLWrapper(SinglesEnv):
    def __init__(
            self,
            *,
            team=None,
            opponent_teams: list[str] | None = None,
            rounds_per_opponents: int = 2_000,
            battle_team_generator=None,
            agent_team_generator=None,
            opponent_team_generator=None,
            **kwargs,
    ):
        super().__init__(team=team, **kwargs)

        self.opponent_teams = opponent_teams or []
        self.battle_team_generator = battle_team_generator
        self.agent_team_generator = agent_team_generator
        self.opponent_team_generator = opponent_team_generator
        self._last_team_update_round = None
        self._latest_battle = None

        self._action_space = gym.spaces.Discrete(26)
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.last_hp: dict = {}
        self.my_last_move: dict = {}
        self.last_opp_pp: dict = {}

        self.last_protect_chance: dict = {}
        self.protect_belief: dict = {}

        self.rounds_played: int = 0
        self.rounds_per_opponents = rounds_per_opponents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_agent1_battle(self, battle) -> bool:
        return getattr(battle, "player_username", None) == self.agent1.username

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

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
                raise ValueError(f"Action {canonical_action} out of bounds ({len(mask)}).")
            canonical_action = ACTION_DEFAULT
        elif not mask[canonical_action]:
            if strict:
                raise ValueError(
                    f"Invalid action {canonical_action}. Valid: {np.flatnonzero(mask).tolist()}"
                )
            canonical_action = ACTION_DEFAULT

        self._update_last_move(battle, canonical_action)

        return super().action_to_order(canonical_action, battle, fake, strict)

    def action_masks(self) -> np.ndarray:
        if self._latest_battle is None:
            return np.ones(self._action_space.n, dtype=bool)
        return get_valid_action_mask(self._latest_battle)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def embed_battle(self, battle) -> np.ndarray:
        if self._is_agent1_battle(battle):
            self._latest_battle = battle
            self._update_battle_state(battle)

        return BattleState.from_battle(
            battle,
            opp_protect_belief=self.protect_belief.get(battle.battle_tag, 1.0),
        ).to_array()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def calc_reward(self, battle) -> float:
        reward, done = calc_reward(
            battle,
            self.last_hp,
            is_agent_battle=self._is_agent1_battle(battle),
        )

        self._update_last_hp(battle)

        if done and self._is_agent1_battle(battle):
            self.rounds_played += 1
        return reward

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self.last_hp = {}
        self.my_last_move = {}
        self.last_opp_pp = {}
        self.last_protect_chance = {}
        self.protect_belief = {}
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

    def _update_battle_state(self, battle) -> None:
        """Update per-turn bookkeeping. Call once per step before embed_battle."""
        tag = battle.battle_tag

        my_last_move = self.my_last_move.get(tag)
        opp_last_move = detect_opponent_move(battle, self.last_opp_pp.get(tag, {}))

        no_damage = did_no_damage(battle, self.last_hp, my_last_move)

        if opp_last_move is not None:
            protected = opp_last_move.is_protect_move
        elif not no_damage:
            protected = False
        else:
            protected = None

        last_protect_chance = self.last_protect_chance.get(tag, 1.0)
        prior_protect_estimate = estimate_protect_attempt_prior(battle)

        self.last_opp_pp[tag] = snapshot_opponent_pp(battle)

        if my_last_move is not None:
            protect_belief = build_protect_belief(
                my_last_move=my_last_move,
                last_chance=last_protect_chance,
                protect_attempt_prior=prior_protect_estimate,
                protected=protected,
            )
            self.last_protect_chance[tag] = protect_belief.expected_next_protect_chance()
            self.protect_belief[tag] = protect_belief.expected_next_protect_belief()
        else:
            self.protect_belief[tag] = prior_protect_estimate

    def _update_last_move(self, battle, canonical_action):
        if 6 <= canonical_action <= 25:
            move_index = (canonical_action - 6) % 4
            self.my_last_move[battle.battle_tag] = (
                battle.available_moves[move_index]
                if move_index < len(battle.available_moves)
                else None
            )
        else:
            self.my_last_move[battle.battle_tag] = None

    def _update_last_hp(self, battle):
        my_hp = battle.active_pokemon.current_hp_fraction
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction
        opp_key = f"opp_{battle.opponent_active_pokemon.species}"

        self.last_hp[opp_key] = (my_hp, opp_hp)
