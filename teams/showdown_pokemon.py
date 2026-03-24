from dataclasses import dataclass, field
from typing import List, Optional


from teams.teams_util import format_stats_dict


@dataclass
class MonArgs:
    nickname: Optional[str] = None
    species: Optional[str] = None
    item: Optional[str] = None
    ability: Optional[str] = None
    moves: Optional[List[str]] = None
    nature: Optional[str] = None
    evs: Optional[str] = None
    ivs: Optional[str] = None
    gender: Optional[str] = None
    level: Optional[int] = None
    shiny: Optional[bool] = None
    teratype: Optional[str] = None
    happiness: Optional[int] = None
    pokeball: Optional[str] = None
    hiddenpowertype: Optional[str] = None
    gigantamax: Optional[bool] = None
    dynamaxlevel: Optional[int] = None
    
    def to_showdown(self) -> str:
        display_nickname = self.nickname or self.species
        display_species = "" if self.species == display_nickname else (self.species or "")
        display_item = self.item or ""
        display_ability = self.ability or ""
        display_moves = ",".join(self.moves) if self.moves else ""
        display_nature = self.nature or ""
        display_evs = self.evs or ""

        if self.gender in ["N", None, ""]:
            display_gender = ""
        else:
            display_gender = self.gender

        display_ivs = self.ivs or ""
        display_shiny = "S" if self.shiny else ""
        display_level = str(self.level) if (self.level and self.level != 100) else ""

        trailing = [
            str(self.happiness) if self.happiness and self.happiness != 255 else "",
            self.pokeball or "",
            self.hiddenpowertype or "",
            "G" if self.gigantamax else "",
            str(self.dynamaxlevel) if self.dynamaxlevel and self.dynamaxlevel != 10 else "",
            self.teratype or ""
        ]

        while trailing and not trailing[-1]:
            trailing.pop()

        trailing_part = ",".join(trailing)

        return (
            f"{display_nickname}|{display_species}|{display_item}|{display_ability}|{display_moves}|"
            f"{display_nature}|{display_evs}|{display_gender}|{display_ivs}|{display_shiny}|"
            f"{display_level}|{trailing_part}"
        )

@dataclass
class TeamArgs:
    mons: List[MonArgs] = field(default_factory=list)

    def to_showdown(self) -> str:
        """Return full packed team string (Showdown format)."""
        return "]".join(mon.to_showdown() for mon in self.mons)

def _mon_from_dict(mon: dict) -> MonArgs:
    """Extract generate_team kwargs from a pool/matchup entry."""
    return MonArgs(
        nickname=mon.get('name'),
        species=mon.get('species'),
        item=mon.get('item'),
        ability=mon.get('ability'),
        moves=mon.get('moves'),
        nature=mon.get('nature'),
        evs=format_stats_dict(mon.get('evs')),
        ivs=format_stats_dict(mon.get('ivs')),
        gender=mon.get('gender'),
        level=mon.get('level'),
        shiny=mon.get('shiny'),
        teratype=mon.get('teraType'),
    )

def team_from_dict(team: dict) -> TeamArgs:
    return TeamArgs(mons=[_mon_from_dict(m) for m in team])