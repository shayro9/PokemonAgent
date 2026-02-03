import numpy as np
from env_wrapper import PokemonRLWrapper
from tests.fakes import FakeBattle, FakeMove


def test_embed_battle_shape_and_dtype():
    env = PokemonRLWrapper()

    battle = FakeBattle(
        moves=[
            FakeMove(100, 100, 10),
            FakeMove(50, 90, 3),
        ]
    )

    state = env.embed_battle(battle)

    # 2 + 2 + 4 + 4 + 4 = 16
    assert state.shape == (16,)
    assert state.dtype == np.float32


def test_embed_battle_hp_and_status():
    env = PokemonRLWrapper()

    battle = FakeBattle(
        my_hp=0.75,
        opp_hp=0.25,
        my_status="brn",
        opp_status=None,
    )

    state = env.embed_battle(battle)

    assert np.isclose(state[0], 0.75)  # my_hp
    assert state[1] == 1.0  # my_status
    assert np.isclose(state[2], 0.25)  # opp_hp
    assert state[3] == 0.0  # opp_status


def test_embed_battle_move_padding():
    env = PokemonRLWrapper()

    battle = FakeBattle(
        moves=[FakeMove(80, 100, 5)]
    )

    state = env.embed_battle(battle)

    moves_attack = state[4:8]
    moves_accuracy = state[8:12]
    moves_type = state[12:16]

    assert np.isclose(moves_attack[0], 0.8)
    assert moves_attack[1] == -1
    assert moves_accuracy[1] == -1
    assert moves_type[3] == -1
