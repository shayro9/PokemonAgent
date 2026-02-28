from dataclasses import dataclass

from poke_env.battle import MoveCategory, Move

from .embed import MAX_MOVES


def did_no_damage(battle, last_hp: dict, last_move, eps=1e-6) -> bool:
    """
    Returns True if our last action did no damage to the opponent.
    Requires last_move to be a physical or special move (not status).
    """
    if last_move is None:
        return False
    if last_move.category == MoveCategory.STATUS:
        return False

    opp = battle.opponent_active_pokemon

    opp_key = f"opp_{battle.opponent_active_pokemon.species}"
    current_hp = opp.current_hp_fraction
    _, previous_hp = last_hp.get(opp_key, (1.0, 1.0))

    # True if HP did not drop (within tolerance)
    return current_hp >= previous_hp - eps


def _clip_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


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

    accuracy: float
    last_chance: float = 1.0
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

    def expected_next_protect_chance_given_no_damage(self) -> float:
        """E[next chance] after observing no damage."""
        posterior = self.posterior_protect_success_given_no_damage()
        next_if_protect_succeeded = self.last_chance / 3.0
        next_if_reset = 1.0
        return posterior * next_if_protect_succeeded + (1.0 - posterior) * next_if_reset


def build_protect_belief(last_move: Move, last_chance: float = 1.0,
                         protect_attempt_prior: float = 1.0) -> ProtectBelief:
    accuracy = last_move.accuracy
    if not isinstance(accuracy, (int, float)):
        accuracy = 1.0

    return ProtectBelief(
        accuracy=_clip_probability(float(accuracy)),
        last_chance=_clip_probability(float(last_chance)),
        protect_attempt_prior=_clip_probability(float(protect_attempt_prior)),
    )


def protect_chance(
        last_move: Move,
        last_chance: float = 1.0,
        no_damage: bool = False,
        protect_attempt_prior: float = 1.0,
) -> float:
    """Return ``E[next Protect success chance]`` from the Bayesian belief model.

    If ``no_damage`` is False, the Protect chain is treated as reset (returns ``1.0``).
    If ``no_damage`` is True, computes:
      ``r = P(protect_success | no_damage)`` and
      ``E[next] = r*(p/3) + (1-r)*1``.
    """
    if not no_damage:
        return 1.0

    belief = build_protect_belief(
        last_move=last_move,
        last_chance=last_chance,
        protect_attempt_prior=protect_attempt_prior,
    )
    return belief.expected_next_protect_chance_given_no_damage()


def estimate_protect_attempt_prior(battle) -> float:
    """Estimate the prior probability that the opponent will attempt to Protect this turn.

    Computed as the fraction of the opponent's remaining PP that belongs to
    Protect-category moves. If no moves have been revealed yet, returns ``1.0``
    as an uninformative prior (assume Protect is possible).

    :param battle: The current battle state.
    :returns: A value in ``[0, 1]`` representing the estimated probability
              that the opponent attempts a Protect move this turn.
    """
    moves = list((battle.opponent_active_pokemon.moves or {}).values())
    if not moves:
        return 1.0

    pp_sum = sum(move.current_pp for move in moves)
    if pp_sum == 0:
        return 0.0

    protect_pp = sum(move.current_pp for move in moves if move.is_protect_move)

    if len(moves) < MAX_MOVES and protect_pp == 0:
        return 1.0 - len(moves) / MAX_MOVES

    return protect_pp / pp_sum
