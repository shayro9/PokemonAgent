"""
combat/stat_belief.py
=====================
Gaussian belief system over an opponent's five in-battle stats
(atk, def, spa, spd, spe).  HP is excluded because it can be read
directly from ``battle.opponent_active_pokemon.max_hp``.

Design highlights
------------
* **Modular prior** — the default prior is derived from base stats + level, but
  any ``StatPriorFn`` callable can be substituted (learned, DB-sampled, flat).
* **Immutable updates** — every evidence method returns a *new* ``StatBelief``.
* **Independent Bayesian updates** — each stat dimension is updated with a
  standard conjugate Gaussian-Gaussian posterior, keeping the model tractable
  while still propagating uncertainty correctly.
* **Model-ready output** — ``to_array()`` returns a normalised 10-dim array
  ``[mean/STAT_NORM ×5, std/STAT_NORM ×5]``

Stat order
-----------------
    INDEX   STAT
      0     atk
      1     def
      2     spa
      3     spd
      4     spe

Damage inference math
---------------------
Simplified Gen-9 formula (ignoring floor ops, treated as noise):

    D = ((level_factor * BP * A / D_stat / 50) + 2) * modifier

where  level_factor = (2 * level / 5 + 2).

Rearranging to isolate the unknown stat:

    A_est   = (D_raw / modifier - 2) * 50 * known_def / (level_factor * BP)
    Def_est = (level_factor * BP * known_atk) / ((D_raw / modifier - 2) * 50)

``D_raw`` = ``damage_fraction * max_hp``  (max_hp read directly from battle).

Speed inference
---------------
We don't observe speed directly; we observe *who moved first*.

Approximation: if we moved first, treat it as a soft Gaussian observation
centred at (our_spe * SPEED_RATIO_FIRST) with variance SPEED_OBS_VAR, and
vice versa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAT_NORM: float = 512.0
STAT_KEYS = ("atk", "def", "spa", "spd", "spe")
N_STATS = len(STAT_KEYS)          # 5

ATK_IDX, DEF_IDX, SPA_IDX, SPD_IDX, SPE_IDX = range(N_STATS)

# Assumed IV / EV defaults for the prior mean (standard competitive assumptions)
PRIOR_IV: int = 31
PRIOR_EV: int = 84
# Prior standard deviation expressed as a fraction of the mean stat value.
PRIOR_STD_FRAC: float = 0.20
# Observation noise for the damage-based Gaussian likelihood.
DAMAGE_OBS_NOISE_FRAC: float = 0.12

# Multipliers used for the speed-order soft observation.
SPEED_RATIO_FIRST: float = 0.80
SPEED_RATIO_SECOND: float = 1.20
SPEED_OBS_VAR_FRAC: float = 0.25

# Minimum variance floor to prevent posterior collapse
MIN_VAR: float = (5.0 / STAT_NORM) ** 2


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class StatPriorFn(Protocol):
    def __call__(self, opp, gen: int) -> "StatBelief": ...


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatBelief:
    """
    Gaussian belief over an opponent's five actual in-battle stats.

    Attributes
    ----------
    mean : np.ndarray, shape (5,)
        Current posterior mean for each stat.
    var : np.ndarray, shape (5,)
        Current posterior variance for each stat (must be > 0).
    """

    mean: np.ndarray   # (5,)  raw stat units
    var:  np.ndarray   # (5,)  raw stat units²

    def to_array(self) -> np.ndarray:
        """Return a 10-dim normalized array ``[mean/STAT_NORM ×5, std/STAT_NORM ×5]``.

        :returns: Float32 array of shape (10,).
        """
        std = np.sqrt(np.maximum(self.var, MIN_VAR))
        return np.concatenate([
            self.mean / STAT_NORM,
            std       / STAT_NORM,
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Evidence updates
    # ------------------------------------------------------------------

    def update_from_damage_received(
        self,
        *,
        damage_fraction: float,
        my_max_hp: float,
        my_defense: float,
        opp_atk_boost: float = 1.0, # opponent's Atk/SpA boost multiplier
        base_power: float,
        move_is_special: bool,
        level_factor: float,
        modifier: float = 1.0,
        extra_noise_frac: float = 0.0,
    ) -> StatBelief:
        """Update belief on opp Atk (physical) or SpA (special) from damage received.

        ``my_defense`` should already be the *effective* (boosted) value.
        The inferred effective opponent attack is divided by ``opp_atk_boost``
        to recover the unboosted base stat the belief tracks.

        :param damage_fraction: Fraction of our max HP lost this turn.
        :param my_max_hp: Our max HP stat (raw).
        :param my_defense: Our effective defense stat after boosts.
        :param opp_atk_boost: Opponent's attack boost multiplier (from ``boost_multiplier()``).
        :param base_power: Move base power.
        :param move_is_special: ``True`` for SpA/SpD inference.
        :param level_factor: ``2 * opp_level / 5 + 2``.
        :param modifier: STAB, type, weather, screens, crit product.
        :param extra_noise_frac: Additional fractional noise.
        :returns: Updated ``StatBelief``.
        """
        d_raw = damage_fraction * my_max_hp
        denom = level_factor * base_power
        if denom <= 0 or modifier <= 0 or (d_raw / modifier) <= 2:
            return self

        # Effective attack inferred from damage formula
        a_eff = (d_raw / modifier - 2) * 50.0 * my_defense / denom
        if a_eff <= 0:
            return self

        # Convert effective → base by removing the opponent's boost
        a_est = a_eff / opp_atk_boost
        if a_est <= 0:
            return self

        noise_frac = DAMAGE_OBS_NOISE_FRAC + extra_noise_frac
        obs_var = (noise_frac * a_est) ** 2

        stat_idx = SPA_IDX if move_is_special else ATK_IDX
        return self._gaussian_update_single(stat_idx, a_est, obs_var)

    def update_from_damage_dealt(
        self,
        *,
        damage_fraction: float,
        opp_max_hp: float,
        my_attack: float,
        opp_def_boost: float = 1.0, # opponent's Def/SpD boost multiplier
        base_power: float,
        move_is_special: bool,
        level_factor: float,
        modifier: float = 1.0,
        extra_noise_frac: float = 0.0,
    ) -> StatBelief:
        """Update belief on opp Def (physical) or SpD (special) from damage dealt.

        ``my_attack`` should already be the *effective* (boosted) value.
        The inferred effective opponent defense is divided by ``opp_def_boost``
        to recover the unboosted base stat the belief tracks.

        :param damage_fraction: Fraction of opponent's max HP lost this turn.
        :param opp_max_hp: Opponent's actual max HP.
        :param my_attack: Our effective attack stat after boosts.
        :param opp_def_boost: Opponent's defense boost multiplier (from ``boost_multiplier()``).
        :param base_power: Move base power.
        :param move_is_special: ``True`` for SpA/SpD inference.
        :param level_factor: ``2 * our_level / 5 + 2``.
        :param modifier: STAB, type, weather, screens, crit product.
        :param extra_noise_frac: Additional fractional noise.
        :returns: Updated ``StatBelief``.
        """
        if opp_max_hp <= 0:
            return self

        d_raw = damage_fraction * opp_max_hp
        if d_raw <= 2:
            return self

        numer = level_factor * base_power * my_attack
        if modifier <= 0 or (d_raw / modifier) <= 2:
            return self

        # Effective defense inferred from damage formula
        def_eff = numer / ((d_raw / modifier - 2) * 50.0)
        if def_eff <= 0:
            return self

        # Convert effective → base by removing the opponent's boost
        def_est = def_eff / opp_def_boost
        if def_est <= 0:
            return self

        noise_frac = DAMAGE_OBS_NOISE_FRAC + extra_noise_frac
        obs_var = (noise_frac * def_est) ** 2

        stat_idx = SPD_IDX if move_is_special else DEF_IDX
        return self._gaussian_update_single(stat_idx, def_est, obs_var)

    def update_from_speed_order(
        self,
        *,
        our_spe: float,
        we_moved_first: bool,
    ) -> StatBelief:
        """Update belief on opp Spe from observed move order.

        ``our_spe`` should be the *effective* (boosted) value so the inferred
        opponent speed estimate is also in effective units.  Since opponent
        speed boosts are tracked in ``opp.boosts``, the caller in
        ``stat_belief_updates.py`` should un-boost before passing here if
        the belief is meant to track base speed.

        In practice, speed boosts are rare and short-lived, so we accept a
        small inaccuracy here rather than adding noisy boost-reversal logic
        for an observation that already has high variance.

        :param our_spe: Our effective Speed stat after boosts.
        :param we_moved_first: ``True`` if we acted before the opponent.
        :returns: Updated ``StatBelief``.
        """
        ratio = SPEED_RATIO_FIRST if we_moved_first else SPEED_RATIO_SECOND
        spe_obs = our_spe * ratio
        obs_var = (our_spe * SPEED_OBS_VAR_FRAC) ** 2
        return self._gaussian_update_single(SPE_IDX, spe_obs, obs_var)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def describe(self) -> str:
        lines = ["[StatBelief]"]
        for i, key in enumerate(STAT_KEYS):
            std = math.sqrt(max(self.var[i], MIN_VAR))
            lines.append(f"  {key:4s}  mean={self.mean[i]:6.1f}  std={std:5.1f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

        """Conjugate Gaussian-Gaussian update for one stat dimension."""
    def _gaussian_update_single(self, idx: int, obs_mean: float, obs_var: float) -> "StatBelief":
        prior_var = max(float(self.var[idx]), MIN_VAR)
        obs_var   = max(obs_var, MIN_VAR)

        precision_post = 1.0 / prior_var + 1.0 / obs_var
        post_var  = 1.0 / precision_post
        post_mean = post_var * (self.mean[idx] / prior_var + obs_mean / obs_var)

        new_mean = self.mean.copy()
        new_var  = self.var.copy()
        new_mean[idx] = post_mean
        new_var[idx]  = max(post_var, MIN_VAR)

        return replace(self, mean=new_mean, var=new_var)


# ---------------------------------------------------------------------------
# Prior functions
# ---------------------------------------------------------------------------

def base_stat_level_prior(opp, gen: int) -> StatBelief:
    """Default prior: mean from base stats + level, wide variance.

    :param opp: poke-env Pokémon object (opponent active pokemon).
    :param gen: Battle generation number.
    :returns: Prior ``StatBelief``.
    """
    level = getattr(opp, "level", 100) or 100
    base  = opp.base_stats

    mean = np.array([
        _stat_formula(base["atk"], level),
        _stat_formula(base["def"], level),
        _stat_formula(base["spa"], level),
        _stat_formula(base["spd"], level),
        _stat_formula(base["spe"], level),
    ], dtype=np.float64)

    var = np.maximum((PRIOR_STD_FRAC * mean) ** 2, MIN_VAR)
    return StatBelief(mean=mean, var=var)


def flat_uninformative_prior(_opp, _gen: int) -> StatBelief:
    """Wide-variance prior that ignores species entirely."""
    mean = np.full(N_STATS, 200.0, dtype=np.float64)
    var  = np.full(N_STATS, 100.0 ** 2, dtype=np.float64)
    return StatBelief(mean=mean, var=var)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_stat_belief(opp, gen: int, prior_fn: StatPriorFn = base_stat_level_prior) -> StatBelief:
    return prior_fn(opp, gen)


def level_factor(level: int) -> float:
    return 2.0 * level / 5.0 + 2.0


def _stat_formula(base: int, level: int) -> float:
    return math.floor((2 * base + PRIOR_IV + math.floor(PRIOR_EV / 4)) * level / 100 + 5)