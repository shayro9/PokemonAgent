from dataclasses import dataclass

from poke_env.battle import Move

from combat.combat_utils import clip_probability
from env.embed import MAX_MOVES


@dataclass(frozen=True)
class ProtectBelief:
    """
    Bayesian miss-vs-Protect model conditioned on observing ``no_damage=True``.

    Notation:
      - ``a``: move accuracy, ``m = 1 - a`` (miss probability)
      - ``p``: current Protect success chance if they attempted this turn
      - ``q``: prior probability they attempted to Protect this turn

    Core equations:
      - ``P(no_damage) = q*p + m*(1 - q*p)``
      - ``r = P(protect_success | no_damage) = (q*p) / P(no_damage)``
      - ``E[next] = r*(p/3) + (1-r)*1``

    Parameters:
        accuracy: chance our move hits (a)
        last_chance: chance Protect would have succeeded if attempted this turn (p)
        protect_attempt_prior: prior that opponent attempted to Protect this turn (q)
    """

    accuracy: float = 1.0
    last_chance: float = 1.0
    protected: bool | None = None
    protect_attempt_prior: float = 1.0

    @property
    def miss_probability(self) -> float:
        return 1.0 - self.accuracy

    @property
    def protect_success_probability(self) -> float:
        return self.protect_attempt_prior * self.last_chance

    @property
    def no_damage_probability(self) -> float:
        qp = self.protect_success_probability
        m = self.miss_probability
        return qp + m * (1.0 - qp)

    def posterior_protect_success_given_no_damage(self) -> float:
        """P(Protect succeeded | no_damage)."""
        denominator = self.no_damage_probability
        if denominator <= 0.0:
            return 0.0
        return self.protect_success_probability / denominator

    def expected_next_protect_chance(self) -> float:
        """E[next Protect success chance] given what we observed."""
        if self.protected is True:
            return self.last_chance / 3.0
        if self.protected is False:
            return 1.0

        # unknown — Bayesian fallback
        posterior = self.posterior_protect_success_given_no_damage()
        return posterior * (self.last_chance / 3.0) + (1.0 - posterior) * 1.0

    def expected_next_protect_belief(self) -> float:
        """E[next Protect success chance] given what we observed."""
        expected_protect_chance = self.expected_next_protect_chance()
        if self.protected is None:
            return expected_protect_chance

        return self.protect_attempt_prior * expected_protect_chance


def build_protect_belief(my_last_move: Move = None, last_chance: float = 1.0, protected: bool = False,
                         protect_attempt_prior: float = 1.0) -> ProtectBelief:
    accuracy = my_last_move.accuracy if my_last_move else 1.0
    if not isinstance(accuracy, (int, float)):
        accuracy = 1.0

    return ProtectBelief(
        accuracy=clip_probability(float(accuracy)),
        last_chance=clip_probability(float(last_chance)),
        protected=protected,
        protect_attempt_prior=clip_probability(float(protect_attempt_prior)),
    )


def estimate_protect_attempt_prior(battle) -> float:
    """Estimate the prior probability that the opponent will attempt to Protect this turn.

    Computed as the fraction of the opponent's remaining PP that belongs to
    Protect-category moves. If no moves have been revealed yet, returns ``1.0``
    as an uninformative prior (assume Protect is possible). Assuming protect on reset.

    :param battle: The current battle state.
    :returns: A value in ``[0, 1]`` representing the estimated probability
              that the opponent attempts a Protect move this turn.
    """
    # Prior on reset
    if not battle:
        return 0.25

    moves = list((battle.opponent_active_pokemon.moves or {}).values())
    if not moves:
        return 0.25

    protect_moves = len([move for move in moves if move.is_protect_move])

    if len(moves) < MAX_MOVES and protect_moves == 0:
        return 1.0 - len(moves) / MAX_MOVES

    return (protect_moves + MAX_MOVES - len(moves)) / MAX_MOVES
