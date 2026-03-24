import numpy as np

from combat.combat_utils import tracker_key
from env.tracker import Tracker
from poke_env.battle import Status, AbstractBattle

from env.states.gen1.battle_state_gen_1 import MAX_TEAM_SIZE

DAMAGE_CLIP = 1.0
_STATUS_WEIGHTS = {Status.SLP: 0.8, Status.PAR: 0.4, Status.BRN: 0.4, Status.TOX: 0.3, Status.PSN: 0.2, Status.FRZ: 0.8}
HP_VALUE = 1.0
FAINTED_VALUE = 5.0
WIN_BONUS = 15.0
LOSS_PENALTY = -10.0


def calc_reward(
        battle,
        tracker: Tracker,
        *,
        is_agent_battle: bool,
) -> tuple[float, bool]:
    """Stateless reward function for a single battle step.

    :param battle: poke-env battle object.
    :param tracker: values history
    :param is_agent_battle: ``True`` when this battle belongs to the learning agent.
    :returns: ``(reward, done)`` where ``done`` is ``True`` once the battle has finished.
    """
    if not is_agent_battle:
        return 0.0, battle.finished

    my_hp = battle.active_pokemon.current_hp_fraction
    opp_hp = battle.opponent_active_pokemon.current_hp_fraction

    # HP delta
    damage_to_opp = (tracker.last_opp_hp - opp_hp) * HP_VALUE
    damage_to_me = (tracker.last_my_hp - my_hp) * HP_VALUE

    # Status: reward only on the TURN it's applied (None → status transition)
    my_status = battle.active_pokemon.status
    opp_status = battle.opponent_active_pokemon.status

    newly_opp_status = (opp_hp > 0) and (opp_status is not None) and (tracker.last_opp_status is None)
    newly_me_status = (my_status is not None) and (tracker.last_my_status is None)

    newly_opp_status_value = _STATUS_WEIGHTS.get(opp_status, 0.2)
    newly_me_status_value = _STATUS_WEIGHTS.get(my_status, 0.2)

    reward = (damage_to_opp + newly_opp_status_value * newly_opp_status
              - damage_to_me - newly_me_status_value * newly_me_status)

    if battle.finished:
        reward += WIN_BONUS if battle.won else (LOSS_PENALTY if battle.lost else 0.0)

    return reward, battle.finished

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