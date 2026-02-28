import numpy as np

DAMAGE_CLIP = 1.0
WIN_BONUS = 5.0
LOSS_PENALTY = -5.0


def calc_reward(
        battle,
        last_hp: dict,
        *,
        is_agent_battle: bool,
) -> tuple[float, bool]:
    """Stateless reward function for a single battle step.

    :param battle: poke-env battle object.
    :param last_hp: Mutable dict keyed by ``battle.battle_tag`` to ``(my_hp, opp_hp)``.
    :param is_agent_battle: ``True`` when this battle belongs to the learning agent.
    :returns: ``(reward, done)`` where ``done`` is ``True`` once the battle has finished.
    """
    if not is_agent_battle:
        return 0.0, battle.finished

    my_hp = battle.active_pokemon.current_hp_fraction
    opp_hp = battle.opponent_active_pokemon.current_hp_fraction
    opp_key = f"opp_{battle.opponent_active_pokemon.species}"

    last_my_hp, last_opp_hp = last_hp.get(opp_key, (1.0, 1.0))

    damage_to_opp = last_opp_hp - opp_hp
    damage_to_me = last_my_hp - my_hp

    reward = float(np.clip(damage_to_opp - damage_to_me, -DAMAGE_CLIP, DAMAGE_CLIP))

    if battle.finished:
        if battle.won:
            reward += WIN_BONUS
        elif battle.lost:
            reward += LOSS_PENALTY

    return reward, battle.finished
