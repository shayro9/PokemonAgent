import numpy as np

from combat.combat_utils import tracker_key
from env.battle_tracker import BattleTracker

DAMAGE_CLIP = 1.0
STATUS_VALUE = 0.2
HP_VALUE = 1.0
WIN_BONUS = 5.0
LOSS_PENALTY = -5.0


def calc_reward(
        battle,
        tracker: BattleTracker,
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

    newly_opp_status = (opp_hp > 0) and (opp_status is not None) \
                         and (tracker.last_opp_status is None or tracker.last_opp_status != opp_status)
    newly_me_status = (my_status is not None) \
                        and (tracker.last_my_status is None or tracker.last_my_status != my_status)

    reward = (damage_to_opp + STATUS_VALUE * newly_opp_status
              - damage_to_me - STATUS_VALUE * newly_me_status)

    if battle.finished:
        reward += WIN_BONUS if battle.won else (LOSS_PENALTY if battle.lost else 0.0)

    return reward, battle.finished
