import numpy as np
import gymnasium as gym
from poke_env.environment import SinglesEnv

from env.action_masking import get_valid_action_mask, ACTION_DEFAULT
from env.battle_state import BattleState, OBS_SIZE
from combat.combat_utils import detect_opponent_move, did_no_damage, snapshot_opponent_pp, tracker_key
from combat.protect_belief import estimate_protect_attempt_prior, build_protect_belief
from combat.stats_belief import build_stat_belief
from combat.stat_belief_updates import update_stat_belief
from env.battle_tracker import BattleTracker
from env.reward import calc_reward
from debug.logs import log_fallback


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
        self._last_finished_battle = None

        self._action_space = gym.spaces.Discrete(26)
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self._trackers: dict[str, BattleTracker] = {}

        self.rounds_played: int = 0
        self.rounds_per_opponents = rounds_per_opponents

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def action_to_order(self, action, battle, fake=False, strict=True):
        if not self._is_agent1_battle(battle=battle):
            return super().action_to_order(action, battle, fake, strict)

        canonical_action = action
        if self._latest_battle != battle:
            print("S")
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

        try:
            return super().action_to_order(canonical_action, battle, fake, strict)
        except ValueError:
            log_fallback(battle, canonical_action)
            return super().action_to_order(canonical_action, battle, fake, strict=False)

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

        tracker = self._get_tracker(battle)
        stat_vec = tracker.stat_belief.to_array() if tracker.stat_belief is not None else None
        return BattleState.from_battle(
            battle,
            opp_protect_belief=tracker.protect_belief,
            opp_stat_belief=stat_vec,
        ).to_array()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def calc_reward(self, battle) -> float:
        tracker = self._get_tracker(battle)
        reward, done = calc_reward(
            battle,
            tracker,
            is_agent_battle=self._is_agent1_battle(battle),
        )
        tracker.commit(battle)
        if done and self._is_agent1_battle(battle):
            self.rounds_played += 1
            self._last_finished_battle = battle
        return reward

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self._trackers = {}
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_agent1_battle(self, battle) -> bool:
        return getattr(battle, "player_username", None) == self.agent1.username

    def _get_tracker(self, battle) -> BattleTracker:
        tag = tracker_key(battle)
        if tag not in self._trackers:
            self._trackers[tag] = BattleTracker()
        return self._trackers[tag]


    def _update_battle_state(self, battle) -> None:
        """Update per-turn bookkeeping. Call once per step before embed_battle."""
        tracker = self._get_tracker(battle)
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        # ── protect belief (unchanged) ───────────────────────────────────
        opp_last_move = detect_opponent_move(battle, tracker.last_opp_pp)
        no_damage = did_no_damage(battle, tracker, tracker.my_last_move)
        protected = opp_last_move.is_protect_move if opp_last_move else (None if no_damage else False)
        prior = estimate_protect_attempt_prior(battle)

        tracker.last_opp_pp = snapshot_opponent_pp(battle)

        if tracker.my_last_move is not None:
            belief = build_protect_belief(tracker.my_last_move, tracker.last_protect_chance, protected, prior)
            tracker.last_protect_chance = belief.expected_next_protect_chance()
            tracker.protect_belief = belief.expected_next_protect_belief()
        else:
            tracker.protect_belief = prior

        # ── stat belief ───────────────────────────────────────────────────

        # Initialize on the very first call for this battle
        if tracker.stat_belief is None:
            tracker.stat_belief = build_stat_belief(opp, battle.gen)
            return

        tracker.stat_belief = update_stat_belief(
            tracker.stat_belief, battle, tracker, opp_last_move
        )

    def _update_last_move(self, battle, canonical_action) -> None:
        tracker = self._get_tracker(battle)
        if 6 <= canonical_action <= 25:
            idx = (canonical_action - 6) % 4
            tracker.my_last_move = battle.available_moves[idx] if idx < len(battle.available_moves) else None
        else:
            tracker.my_last_move = None

    def get_last_battle(self):
        return self._last_finished_battle
