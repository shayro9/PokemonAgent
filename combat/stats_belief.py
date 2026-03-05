"""
combat/stat_belief.py
=====================
Gaussian belief system over an opponent's six in-battle stats
(hp, atk, def, spa, spd, spe).

Design highlights
------------
* **Modular prior** — the default prior is derived from base stats + level, but
  any ``StatPriorFn`` callable can be substituted (learned, DB-sampled, flat).
* **Immutable updates** — every evidence method returns a *new* ``StatBelief``.
* **Independent Bayesian updates** — each stat dimension is updated with a
  standard conjugate Gaussian-Gaussian posterior, keeping the model tractable
  while still propagating uncertainty correctly.
* **Model-ready output** — ``to_array()`` returns a normalised 12-dim array
  ``[mean/STAT_NORM ×6, std/STAT_NORM ×6]``

Stat order
-----------------
    INDEX   STAT
      0     hp
      1     atk
      2     def
      3     spa
      4     spd
      5     spe

Damage inference math
---------------------
Simplified Gen-9 formula (ignoring floor ops, treated as noise):

    D = ((level_factor * BP * A / D_stat / 50) + 2) * modifier

where  level_factor = (2 * level / 5 + 2).

Rearranging to isolate the unknown stat:

    A_est   = (D_raw / modifier - 2) * 50 * known_def / (level_factor * BP)
    Def_est = (level_factor * BP * known_atk) / ((D_raw / modifier - 2) * 50)

``D_raw`` = ``damage_fraction * max_hp``.

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
from typing import Callable, Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAT_NORM: float = 512.0
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
N_STATS = len(STAT_KEYS)          # 6

# Indices into the stat vector
HP_IDX, ATK_IDX, DEF_IDX, SPA_IDX, SPD_IDX, SPE_IDX = range(N_STATS)

# Assumed IV / EV defaults for the prior mean (standard competitive assumptions)
PRIOR_IV: int = 31
PRIOR_EV: int = 84

# Prior standard deviation expressed as a fraction of the mean stat value.
# Reflects EV spread uncertainty: 0 EVs vs 252 EVs can shift a stat by ~25%.
PRIOR_STD_FRAC: float = 0.20

# Observation noise for the damage-based Gaussian likelihood.
# The ±15% random roll alone contributes ~7.5% of the stat value;
# we add a small extra term for modifier uncertainty.
DAMAGE_OBS_NOISE_FRAC: float = 0.12

# Multipliers used for the speed-order soft observation.
SPEED_RATIO_FIRST: float = 0.80
SPEED_RATIO_SECOND: float = 1.20
SPEED_OBS_VAR_FRAC: float = 0.25  # as a fraction of our_spe

# Minimum variance floor to prevent posterior collapse
MIN_VAR: float = (5.0 / STAT_NORM) ** 2


# ---------------------------------------------------------------------------
# Protocol / type alias for prior functions
# ---------------------------------------------------------------------------

class StatPriorFn(Protocol):
    """Callable that builds an initial ``StatBelief`` for an opponent Pokémon.

    :param opp: poke-env Pokémon object (opponent active Pokémon).
    :param gen: Battle generation number.
    :returns: Prior ``StatBelief``.
    """
    def __call__(self, opp, gen: int) -> "StatBelief": ...


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatBelief:
    """
    Gaussian belief over an opponent's six actual in-battle stats.

    All values are in *raw stat units* (not normalized) so that arithmetic
    remains interpretable.  Normalisation only happens at ``to_vector()`` time.

    Attributes
    ----------
    mean : np.ndarray, shape (6,)
        Current posterior mean for each stat.
    var : np.ndarray, shape (6,)
        Current posterior variance for each stat (must be > 0).
    """

    mean: np.ndarray   # (6,)  raw stat units
    var:  np.ndarray   # (6,)  raw stat units²

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Return a 12-dim normalized array ``[mean/STAT_NORM ×6, std/STAT_NORM ×6]``.

        :returns: Float32 array of shape (12,).
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
        base_power: float,
        move_is_special: bool,
        level_factor: float,
        modifier: float = 1.0,
        extra_noise_frac: float = 0.0,
    ) -> StatBelief:
        """Update belief on opp Atk (physical) or SpA (special) from damage we received.

        Solves for the attacker stat using the simplified damage formula::

            A_est = (D_raw / modifier - 2) * 50 * my_defense / (level_factor * BP)

        where  ``D_raw = damage_fraction * my_max_hp``.

        :param damage_fraction: Fraction of *our* max HP lost this turn.
        :param my_max_hp: Our Pokémon's max HP stat (raw, known exactly).
        :param my_defense: Our relevant defense stat (Def or SpD, raw).
        :param base_power: Move base power.
        :param move_is_special: ``True`` for SpA/SpD, ``False`` for Atk/Def.
        :param level_factor: ``2 * opp_level / 5 + 2``.
        :param modifier: Product of STAB, type multiplier, weather, etc.
        :param extra_noise_frac: Additional fractional noise on top of the
            default roll uncertainty (use for unknown modifiers).
        :returns: Updated ``StatBelief``.
        """
        d_raw = damage_fraction * my_max_hp
        denom = level_factor * base_power
        if denom <= 0 or modifier <= 0 or (d_raw / modifier) <= 2:
            return self  # degenerate observation — no update

        a_est = (d_raw / modifier - 2) * 50.0 * my_defense / denom
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
        my_attack: float,
        base_power: float,
        move_is_special: bool,
        level_factor: float,
        modifier: float = 1.0,
        extra_noise_frac: float = 0.0,
    ) -> StatBelief:
        """Update belief on opp Def (physical) or SpD (special) from damage we dealt.

        Uses the current HP-mean belief to convert the fraction into raw HP::

            D_raw   = damage_fraction * opp_hp_mean_est
            Def_est = level_factor * BP * my_attack / ((D_raw / modifier - 2) * 50)

        :param damage_fraction: Fraction of *opponent's* max HP lost this turn.
        :param my_attack: Our relevant attack stat (Atk or SpA, raw).
        :param base_power: Move base power.
        :param move_is_special: ``True`` for SpA/SpD, ``False`` for Atk/Def.
        :param level_factor: ``2 * our_level / 5 + 2``.
        :param modifier: Product of STAB, type multiplier, etc.
        :param extra_noise_frac: Additional fractional noise for uncertain mods.
        :returns: Updated ``StatBelief``.
        """
        opp_hp_est = self.mean[HP_IDX]
        if opp_hp_est <= 0:
            return self

        d_raw = damage_fraction * opp_hp_est
        if d_raw <= 2:
            return self  # almost no damage — too noisy to infer defense from

        numer = level_factor * base_power * my_attack
        if modifier <= 0 or (d_raw / modifier) <= 2:
            return self
        def_est = numer / ((d_raw / modifier - 2) * 50.0)
        if def_est <= 0:
            return self

        hp_noise_frac = np.sqrt(max(self.var[HP_IDX], MIN_VAR)) / opp_hp_est
        noise_frac = DAMAGE_OBS_NOISE_FRAC + hp_noise_frac + extra_noise_frac
        obs_var = (noise_frac * def_est) ** 2

        stat_idx = SPD_IDX if move_is_special else DEF_IDX
        return self._gaussian_update_single(stat_idx, def_est, obs_var)

    def update_from_speed_order(
        self,
        *,
        our_spe: float,
        we_moved_first: bool,
    ) -> StatBelief:
        """Update belief on opp Spe from the observed move-order this turn.

        Uses a soft-Gaussian approximation rather than a full truncated-Gaussian
        update.  The center of the likelihood is placed at a fraction of our
        speed that is consistent with the observed order:

        * moved first  → opp_spe ~ N(our_spe * 0.80, σ²)
        * moved second → opp_spe ~ N(our_spe * 1.20, σ²)

        :param our_spe: Our Pokémon's current Speed stat (raw, after boosts).
        :param we_moved_first: ``True`` if our Pokémon acted before the opponent.
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
        """Return a human-readable table of mean ± std for each stat.

        :returns: Formatted string for debugging / logging.
        """
        lines = ["[StatBelief]"]
        for i, key in enumerate(STAT_KEYS):
            std = math.sqrt(max(self.var[i], MIN_VAR))
            lines.append(f"  {key:4s}  mean={self.mean[i]:6.1f}  std={std:5.1f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gaussian_update_single(
        self,
        idx: int,
        obs_mean: float,
        obs_var: float,
    ) -> "StatBelief":
        """Conjugate Gaussian-Gaussian update for one stat dimension.

        Standard result: given prior N(μ, σ²) and likelihood N(x, σ_obs²),
        the posterior is::

            σ²_post = 1 / (1/σ² + 1/σ_obs²)
            μ_post  = σ²_post * (μ/σ² + x/σ_obs²)

        :param idx: Index into the stat vector to update.
        :param obs_mean: Point estimate derived from battle evidence.
        :param obs_var: Variance of that estimate (reflects measurement noise).
        :returns: New ``StatBelief`` with the updated dimension.
        """
        prior_var = float(self.var[idx])
        prior_var = max(prior_var, MIN_VAR)
        obs_var   = max(obs_var,   MIN_VAR)

        precision_post = 1.0 / prior_var + 1.0 / obs_var
        post_var  = 1.0 / precision_post
        post_mean = post_var * (self.mean[idx] / prior_var + obs_mean / obs_var)

        new_mean = self.mean.copy()
        new_var  = self.var.copy()
        new_mean[idx] = post_mean
        new_var[idx]  = max(post_var, MIN_VAR)

        return replace(self, mean=new_mean, var=new_var)


# ---------------------------------------------------------------------------
# Prior functions  (modular — pass any StatPriorFn to build_stat_belief)
# ---------------------------------------------------------------------------

def base_stat_level_prior(opp, gen: int) -> StatBelief:
    """Default prior: mean computed from base stats + level, wide variance.

    Uses the standard stat formula with assumed IV=31 and EV=84 (neutral
    investment), and a neutral nature (multiplier=1.0).  Variance is set to
    ``(PRIOR_STD_FRAC * mean)²`` to reflect EV-spread and nature uncertainty.

    This function is intentionally simple so it can be swapped for a learned
    or DB-sampled prior without changing any downstream code.

    :param opp: poke-env Pokémon object (opponent active pokemon).
    :param gen: Battle generation number (reserved for future gen-specific logic).
    :returns: Prior ``StatBelief``.
    """
    level = getattr(opp, "level", 100) or 100
    base  = opp.base_stats          # dict: {hp, atk, def, spa, spd, spe}

    mean = np.array([
        _stat_formula_hp(base["hp"], level),
        _stat_formula(base["atk"], level),
        _stat_formula(base["def"], level),
        _stat_formula(base["spa"], level),
        _stat_formula(base["spd"], level),
        _stat_formula(base["spe"], level),
    ], dtype=np.float64)

    var = np.maximum((PRIOR_STD_FRAC * mean) ** 2, MIN_VAR)
    return StatBelief(mean=mean, var=var)


def flat_uninformative_prior(_opp, _gen: int) -> StatBelief:
    """Wide-variance prior that ignores species entirely.

    Useful as a baseline or when species information is unavailable.
    Mean = 200 for all stats (rough competitive midpoint), std = 100.

    :param _opp: Ignored.
    :param _gen: Ignored.
    :returns: Flat ``StatBelief``.
    """
    mean = np.full(N_STATS, 200.0, dtype=np.float64)
    var  = np.full(N_STATS, 100.0 ** 2, dtype=np.float64)
    return StatBelief(mean=mean, var=var)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_stat_belief(
    opp,
    gen: int,
    prior_fn: StatPriorFn = base_stat_level_prior,
) -> StatBelief:
    """Build the initial stat belief for an opponent Pokémon.

    Separating construction from the ``StatBelief`` class keeps the prior
    logic interchangeable: pass ``flat_uninformative_prior``, a learned model,
    or a DB-lookup function without changing any call sites.

    :param opp: poke-env Pokémon object.
    :param gen: Battle generation number.
    :param prior_fn: Callable matching ``StatPriorFn``. Defaults to
        ``base_stat_level_prior``.
    :returns: Initial ``StatBelief`` from the chosen prior.
    """
    return prior_fn(opp, gen)


def level_factor(level: int) -> float:
    """Return the level scaling term used in the damage formula.

    :param level: Pokémon level.
    :returns: ``2 * level / 5 + 2`` as a float.
    """
    return 2.0 * level / 5.0 + 2.0


# ---------------------------------------------------------------------------
# Stat formula helpers (private)
# ---------------------------------------------------------------------------

def _stat_formula(base: int, level: int) -> float:
    """Standard non-HP stat formula with assumed defaults.

    ``floor((2*base + IV + floor(EV/4)) * level / 100 + 5) * nature``

    IV=31, EV=84, nature=1.0 (neutral).

    :param base: Base stat value.
    :param level: Pokémon level.
    :returns: Estimated actual stat as a float.
    """
    return math.floor((2 * base + PRIOR_IV + math.floor(PRIOR_EV / 4)) * level / 100 + 5)


def _stat_formula_hp(base: int, level: int) -> float:
    """HP stat formula with assumed defaults.

    ``floor((2*base + IV + floor(EV/4)) * level / 100 + level + 10)``

    :param base: Base HP value.
    :param level: Pokémon level.
    :returns: Estimated max HP as a float.
    """
    return math.floor((2 * base + PRIOR_IV + math.floor(PRIOR_EV / 4)) * level / 100 + level + 10)