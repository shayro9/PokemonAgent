from types import SimpleNamespace

from env.reward import calc_reward


def _make_battle(*, battle_tag: str, species: str, my_hp: float, opp_hp: float):
    active = SimpleNamespace(current_hp_fraction=my_hp, status=None)
    opponent = SimpleNamespace(species=species, current_hp_fraction=opp_hp, status=None)
    return SimpleNamespace(
        battle_tag=battle_tag,
        active_pokemon=active,
        opponent_active_pokemon=opponent,
        finished=False,
        won=False,
        lost=False,
    )


def test_calc_reward_uses_tag_and_species_for_independent_hp_history():
    battle_a = _make_battle(
        battle_tag="battle-a",
        species="pikachu",
        my_hp=0.8,
        opp_hp=0.4,
    )
    battle_b = _make_battle(
        battle_tag="battle-b",
        species="pikachu",
        my_hp=0.8,
        opp_hp=0.4,
    )

    last_hp = {
        "battle-a|pikachu": (1.0, 0.9),
        "battle-b|pikachu": (1.0, 0.6),
    }

    reward_a, done_a = calc_reward(battle_a, last_hp, is_agent_battle=True)
    reward_b, done_b = calc_reward(battle_b, last_hp, is_agent_battle=True)

    assert done_a is False
    assert done_b is False
    assert reward_a == 0.3
    assert reward_b == 0.0



def test_calc_reward_tracks_species_separately_within_same_battle():
    battle_pikachu = _make_battle(
        battle_tag="battle-switch",
        species="pikachu",
        my_hp=0.8,
        opp_hp=0.4,
    )
    battle_charizard = _make_battle(
        battle_tag="battle-switch",
        species="charizard",
        my_hp=0.8,
        opp_hp=0.4,
    )

    last_hp = {
        "battle-switch|pikachu": (1.0, 0.9),
        "battle-switch|charizard": (1.0, 0.6),
    }

    reward_pikachu, _ = calc_reward(battle_pikachu, last_hp, is_agent_battle=True)
    reward_charizard, _ = calc_reward(battle_charizard, last_hp, is_agent_battle=True)

    assert reward_pikachu == 0.3
    assert reward_charizard == 0.0
