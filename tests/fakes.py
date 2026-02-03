class FakeMove:
    def __init__(self, base_power, accuracy, type_value):
        self.base_power = base_power
        self.accuracy = accuracy
        self.type = type("Type", (), {"value": type_value})


class FakePokemon:
    def __init__(self, hp_frac, status=None):
        self.current_hp_fraction = hp_frac
        self.status = status


class FakeBattle:
    def __init__(
        self,
        my_hp=1.0,
        opp_hp=1.0,
        my_status=None,
        opp_status=None,
        moves=None,
    ):
        self.active_pokemon = FakePokemon(my_hp, my_status)
        self.opponent_active_pokemon = FakePokemon(opp_hp, opp_status)
        self.available_moves = moves or []
