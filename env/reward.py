from poke_env.battle import Status, AbstractBattle

from env.states.gen1.battle_state_gen_1 import MAX_TEAM_SIZE

# ============================================================================
# HP & FAINTING REWARDS
# ============================================================================
DAMAGE_CLIP = 1.0
HP_VALUE = 1.0                    # Base reward per HP fraction dealt/taken
FAINTED_VALUE = 5.0               # Reward for fainting opponent, penalty for own faint

# ============================================================================
# STATUS CONDITION REWARDS
# ============================================================================
# Reward for inflicting status on opponent (negative for own team)
_STATUS_WEIGHTS = {
    Status.SLP: 1.2,  # Sleep is devastating (prevents moves)
    Status.PAR: 0.5,  # Paralysis reduces speed (modest value)
    Status.BRN: 0.6,  # Burn reduces Atk (moderate value)
    Status.TOX: 0.4,  # Toxic damage over time
    Status.PSN: 0.3,  # Regular poison (weak value)
    Status.FRZ: 1.2,  # Freeze prevents moves entirely
}

# ============================================================================
# STAT BOOST REWARDS (NEW)
# ============================================================================
# Reward for positive boosts on own Pokémon
OWN_BOOST_VALUES = {
    'atk': 0.8,      # +Atk encourages offense
    'def': 0.6,      # +Def is defensive (less critical than offense)
    'spa': 0.8,      # +SpA encourages special offense
    'spd': 0.6,      # +SpD is defensive
    'spe': 1.0,      # +Spe is highest priority (speed = control)
    'accuracy': 0.4, # +Accuracy is situational
    'evasion': 0.5,  # +Evasion is good but unreliable
}

# Penalty for negative boosts on opponent Pokémon (even more valuable!)
OPP_BOOST_PENALTIES = {
    'atk': 1.2,      # -Atk on opponent = massive value (prevents sweeps)
    'def': 0.8,      # -Def on opponent = moderate value
    'spa': 1.2,      # -SpA on opponent = massive value
    'spd': 0.8,      # -SpD on opponent = moderate value
    'spe': 1.5,      # -Spe on opponent = highest value (control battle)
    'accuracy': 0.6, # -Accuracy on opponent = useful
    'evasion': 0.8,  # -Evasion on opponent = useful
}

# ============================================================================
# GAME-ENDING REWARDS
# ============================================================================
WIN_BONUS = 15.0
LOSS_PENALTY = -10.0


def _calculate_boost_value(boosts: dict, boost_values: dict, negate: bool = False) -> float:
    """Calculate total value of stat boosts.
    
    Args:
        boosts: Dictionary of boost values (poke-env format)
        boost_values: Weight dictionary for each stat
        negate: If True, negative boosts are valued (opponent penalties)
    
    Returns:
        Total boost value
    """
    value = 0.0
    for stat, weight in boost_values.items():
        if stat in boosts:
            boost_amount = boosts[stat]
            if negate:
                # For opponent: negative boosts are good (penalize them)
                value += -boost_amount * weight  # -(-2) = +2 (good!)
            else:
                # For own team: positive boosts are good
                value += boost_amount * weight   # +(+2) = +2 (good!)
    return value


def get_state_value(battle: AbstractBattle) -> float:
    """Calculate the state value for the battle with boost-aware rewards.

    In order to calc reward, just need to calculate the delta of values between states.
    
    Reward components:
    1. HP damage dealt/taken (normalized by team size)
    2. Status conditions inflicted/suffered
    3. Stat boosts on own Pokémon (NEW: emphasized)
    4. Stat reductions on opponent (NEW: extra emphasized)
    5. Faints (own penalty, opponent bonus)
    6. Game outcome (win/loss)

    :param battle: poke-env battle object.
    :return: value for the battle.
    """
    current_value = 0.0

    # ========== OWN TEAM VALUE ==========
    for mon in battle.team.values():
        # HP value (normalized)
        current_value += mon.current_hp_fraction * HP_VALUE
        
        # Status condition penalty
        if mon.fainted:
            current_value -= FAINTED_VALUE
        elif mon.status is not None:
            current_value -= _STATUS_WEIGHTS.get(mon.status, 0.0)
        
        # Positive stat boosts (setup moves rewarded here)
        current_value += _calculate_boost_value(mon.boosts, OWN_BOOST_VALUES, negate=False)

    # Penalty for fainted/missing team members
    current_value += (MAX_TEAM_SIZE - len(battle.team)) * HP_VALUE

    # ========== OPPONENT TEAM VALUE ==========
    for mon in battle.opponent_team.values():
        # HP value (negative, so opponent damage is positive)
        current_value -= mon.current_hp_fraction * HP_VALUE
        
        # Status condition bonus (inflicting status is good)
        if mon.fainted:
            current_value += FAINTED_VALUE
        elif mon.status is not None:
            current_value += _STATUS_WEIGHTS.get(mon.status, 0.0)
        
        # Negative stat boosts on opponent (huge reward for crippling opponent)
        current_value += _calculate_boost_value(mon.boosts, OPP_BOOST_PENALTIES, negate=True)

    # Bonus for reducing opponent's available team
    current_value -= (MAX_TEAM_SIZE - len(battle.opponent_team)) * HP_VALUE

    # ========== GAME-ENDING REWARDS ==========
    if battle.won:
        current_value += WIN_BONUS
    elif battle.lost:
        current_value += LOSS_PENALTY

    return current_value