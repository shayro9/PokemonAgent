from env_wrapper import PokemonRLWrapper
from tests.fakes import FakeBattle
import numpy as np


def test_calc_reward_damage():
    env = PokemonRLWrapper()
    env.last_hp = 1.0

    battle = FakeBattle(opp_hp=0.7)

    r = env.calc_reward(battle)

    assert abs(r - 0.3) < 1e-6


def test_calc_reward_sequential():
    env = PokemonRLWrapper()
    env.last_hp = 1.0

    b1 = FakeBattle(opp_hp=0.8)
    b2 = FakeBattle(opp_hp=0.5)

    r1 = env.calc_reward(b1)
    r2 = env.calc_reward(b2)

    assert np.isclose(r1, 0.2)
    assert np.isclose(r2, 0.3)


def test_calc_reward_no_damage():
    env = PokemonRLWrapper()
    env.last_hp = 0.5

    battle = FakeBattle(opp_hp=0.5)

    r = env.calc_reward(battle)

    assert r < 1e-6
