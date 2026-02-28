import numpy as np
import gymnasium as gym
from poke_env.environment import SinglesEnv
from poke_env.battle import MoveCategory

from env.action_masking import get_valid_action_mask, ACTION_DEFAULT
from env.battle_state import BattleState, OBS_SIZE
from env.reward import calc_reward
from env.combat_utils import build_protect_belief


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
        self.last_move = {}
        self._protect_next_chance: dict = {}
        self._protect_posterior: dict = {}
        self.protect_attempt_prior: float = 1.0
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
        self.last_move[battle.battle_tag] = battle.available_moves[canonical_action] \
            if canonical_action < len(battle.available_moves) else None
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

        tag = getattr(battle, "battle_tag", None)
        protect_posterior = self._protect_posterior.get(tag, 0.0)
        protect_next_chance = self._protect_next_chance.get(tag, 1.0)

        return BattleState.from_battle(
            battle,
            protect_posterior=protect_posterior,
            protect_next_chance=protect_next_chance,
        ).to_array()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _update_protect_belief(self, battle, previous_opp_hp: float | None) -> None:
        """Update per-battle Protect belief after a completed turn."""
        tag = battle.battle_tag
        last_move = self.last_move.get(tag)

        if last_move is None or previous_opp_hp is None:
            self._protect_posterior[tag] = 0.0
            self._protect_next_chance[tag] = 1.0
            return

        if getattr(last_move, "category", None) is None or last_move.category == MoveCategory.STATUS:
            self._protect_posterior[tag] = 0.0
            self._protect_next_chance[tag] = 1.0
            return

        current_opp_hp = battle.opponent_active_pokemon.current_hp_fraction
        if current_opp_hp is None:
            self._protect_posterior[tag] = 0.0
            self._protect_next_chance[tag] = 1.0
            return

        no_damage = current_opp_hp >= previous_opp_hp - 1e-6
        last_chance = self._protect_next_chance.get(tag, 1.0)

        if not no_damage:
            self._protect_posterior[tag] = 0.0
            self._protect_next_chance[tag] = 1.0
            return

        belief = build_protect_belief(
            last_move=last_move,
            last_chance=last_chance,
            protect_attempt_prior=self.protect_attempt_prior,
        )
        posterior = belief.posterior_protect_success_given_no_damage()
        next_chance = belief.expected_next_protect_chance_given_no_damage()

        self._protect_posterior[tag] = posterior
        self._protect_next_chance[tag] = next_chance


    def calc_reward(self, battle) -> float:
        previous_opp_hp = None
        if self._is_agent1_battle(battle):
            previous_opp_hp = self.last_hp.get(battle.battle_tag, (None, None))[1]

        reward, done = calc_reward(
            battle,
            self.last_hp,
            is_agent_battle=self._is_agent1_battle(battle),
        )

        if self._is_agent1_battle(battle):
            self._update_protect_belief(battle, previous_opp_hp)

        if done and self._is_agent1_battle(battle):
            self.rounds_played += 1
        return reward

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self.last_hp = {}
        self._latest_battle = None
        self._protect_next_chance = {}
        self._protect_posterior = {}

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
