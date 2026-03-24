from poke_env.battle import Status, AbstractBattle

from env.states.gen1.battle_state_gen_1 import MAX_TEAM_SIZE

DAMAGE_CLIP = 1.0
_STATUS_WEIGHTS = {Status.SLP: 0.8, Status.PAR: 0.4, Status.BRN: 0.4, Status.TOX: 0.3, Status.PSN: 0.2, Status.FRZ: 0.8}
HP_VALUE = 1.0
FAINTED_VALUE = 5.0
WIN_BONUS = 15.0
LOSS_PENALTY = -10.0


def get_state_value(battle: AbstractBattle) -> float:
    """ Calculate the state value for the battle.

    In order to calc reward just need to calculate the delta of values between states.

    :param battle: poke-env battle object.
    :return:value for the battle.
    """
    current_value = 0.0

    for mon in battle.team.values():
        current_value += mon.current_hp_fraction * HP_VALUE
        if mon.fainted:
            current_value -= FAINTED_VALUE
        elif mon.status is not None:
            current_value -= FAINTED_VALUE

    current_value += (MAX_TEAM_SIZE - len(battle.team)) * HP_VALUE

    for mon in battle.opponent_team.values():
        current_value -= mon.current_hp_fraction * HP_VALUE
        if mon.fainted:
            current_value += FAINTED_VALUE
        elif mon.status is not None:
            current_value += _STATUS_WEIGHTS.get(mon.status, 0.0)

    current_value -= (MAX_TEAM_SIZE - len(battle.opponent_team)) * HP_VALUE

    if battle.won:
        current_value += WIN_BONUS
    elif battle.lost:
        current_value += LOSS_PENALTY

    return current_value