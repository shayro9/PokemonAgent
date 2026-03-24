from dataclasses import dataclass
from functools import lru_cache

from poke_env.battle import MoveCategory, Move, Weather, PokemonType, SideCondition, Status, Pokemon, Battle
from poke_env.data import GenData


@lru_cache(maxsize=None)
def type_chart_for_gen(gen: int):
    """Return cached type chart data for a generation."""
    return GenData.from_gen(gen).type_chart


def tracker_key(battle) -> str:
    """Build a stable tracker-history key scoped to battle and opponent species."""
    return f"{battle.battle_tag}"


def clip_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def boost_multiplier(stage: int) -> float:
    """Return the Gen-6+ stat multiplier for a given boost stage.

    Uses the standard formula ``max(2, 2+stage) / max(2, 2-stage)``.
    Valid for Atk, Def, SpA, SpD, Spe (not accuracy/evasion).

    Examples:
        stage  -6 → 0.25
        stage  -1 → 0.67
        stage   0 → 1.00
        stage  +1 → 1.50
        stage  +2 → 2.00
        stage  +6 → 4.00

    :param stage: Boost stage in the range [-6, +6].
    :returns: Multiplicative stat modifier as a float.
    """
    stage = max(-6, min(6, stage))
    return max(2, 2 + stage) / max(2, 2 - stage)


def calc_modifier(
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
    battle: Battle,
    attacker_is_us: bool,
) -> tuple[float, float]:
    """Compute the combined damage modifier and residual noise for a move.

    Does NOT include boost multipliers — those are applied separately in
    ``stat_belief_updates.py`` so the belief can track unboosted base stats.

    :param move: The move being used.
    :param attacker: poke-env Pokemon using the move.
    :param defender: poke-env Pokemon receiving the move.
    :param battle: poke-env battle object.
    :param attacker_is_us: ``True`` when our Pokemon is the attacker.
    :returns: ``(modifier, extra_noise_frac)`` tuple.
    """
    mod = 1.0

    # --- STAB ---
    if move.type in attacker.types:
        mod *= attacker.stab_multiplier

    # --- Type effectiveness ---
    defending_types = [t for t in defender.types if t is not None]
    if defending_types:
        mod *= move.type.damage_multiplier(*defending_types, type_chart=type_chart_for_gen(battle.gen))

    # --- Weather ---
    weather = next(iter(battle.weather), None)
    if weather is not None:
        if weather == Weather.SUNNYDAY:
            if move.type == PokemonType.FIRE:
                mod *= 1.5
            elif move.type == PokemonType.WATER:
                mod *= 0.5
        elif weather == Weather.RAINDANCE:
            if move.type == PokemonType.WATER:
                mod *= 1.5
            elif move.type == PokemonType.FIRE:
                mod *= 0.5

    # --- Burn (halves attacker's physical damage) ---
    if (
        move.category == MoveCategory.PHYSICAL
        and getattr(attacker, "status", None) == Status.BRN
    ):
        mod *= 0.5

    # --- Screens ---
    side_conditions = (
        battle.opponent_side_conditions if attacker_is_us else battle.side_conditions
    )
    if move.category == MoveCategory.PHYSICAL and SideCondition.REFLECT in side_conditions:
        mod *= 0.5
    elif move.category == MoveCategory.SPECIAL and SideCondition.LIGHT_SCREEN in side_conditions:
        mod *= 0.5

    # --- Crit ---
    _CRIT_PROB = {0: 1 / 24, 1: 1 / 8, 2: 1 / 2}
    crit_ratio = getattr(move, "crit_ratio", 0)
    p_crit = 1.0 if crit_ratio >= 3 else _CRIT_PROB.get(crit_ratio, 1 / 24)
    mod *= (1.0 + 0.5 * p_crit)
    extra_noise = 0.05 + p_crit * 0.5

    return mod, extra_noise