from dataclasses import dataclass, field

from poke_env.battle import AbstractBattle
from poke_env.battle.status import Status

from combat.event_parser import detect_opponent_move_from_events, detect_my_move_from_events


@dataclass
class BattleSnapshot:
    my_hp: float = 1.0
    opp_hp: float = 1.0
    my_status: Status | None = None
    opp_status: Status | None = None
    my_move: str | None = None
    opp_move: str | None = None

@dataclass
class Tracker:
    history: list[BattleSnapshot] = field(default_factory=list)

    @property
    def opp_last_move(self):
        return self.history[-1].opp_move

    @property
    def my_last_move(self):
        return self.history[-1].my_move

    @property
    def last_opp_hp(self):
        return self.history[-1].opp_hp if self.history else 1.0

    @property
    def last_my_hp(self):
        return self.history[-1].my_hp if self.history else 1.0

    @property
    def last_my_status(self):
        return self.history[-1].my_status if self.history else None

    @property
    def last_opp_status(self):
        return self.history[-1].opp_status if self.history else None

    def commit(self, battle: AbstractBattle):
        snapshot = BattleSnapshot(
            my_hp=battle.active_pokemon.current_hp_fraction,
            opp_hp=battle.opponent_active_pokemon.current_hp_fraction,
            my_status=battle.active_pokemon.status,
            opp_status=battle.opponent_active_pokemon.status,
            my_move=detect_my_move_from_events(battle),
            opp_move=detect_opponent_move_from_events(battle),
        )
        self.history.append(snapshot)
