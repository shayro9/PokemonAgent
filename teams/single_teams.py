import random
from collections.abc import Iterator, Sequence


STEELIX_TEAM = """
Steelix @ Leftovers  
Ability: Sturdy  
EVs: 252 HP / 252 Def / 4 Atk  
Impish Nature  
- Earthquake  
- Heavy Slam  
- Curse  
- Toxic
"""

GARCHOMP_TEAM = """
Garchomp @ Rocky Helmet  
Ability: Rough Skin  
EVs: 252 HP / 252 Def / 4 Atk  
Impish Nature  
- Earthquake  
- Dragon Tail  
- Stealth Rock  
- Fire Fang
"""

CONKELDURR_TEAM = """
Conkeldurr @ Leftovers  
Ability: Guts  
EVs: 252 HP / 252 Atk / 4 Def  
Adamant Nature  
- Drain Punch  
- Mach Punch  
- Knock Off  
- Bulk Up
"""

ROTOM_WASH_TEAM = """
Rotom-Wash @ Leftovers  
Ability: Levitate  
EVs: 252 HP / 200 Def / 56 SpA  
Bold Nature  
- Hydro Pump  
- Volt Switch  
- Will-O-Wisp  
- Pain Split
"""

CORVIKNIGHT_TEAM = """
Corviknight @ Leftovers  
Ability: Pressure  
EVs: 252 HP / 168 Def / 88 SpD  
Impish Nature  
- Brave Bird  
- Roost  
- Defog  
- Bulk Up
"""

TOXAPEX_TEAM = """
Toxapex @ Black Sludge  
Ability: Regenerator  
EVs: 252 HP / 252 Def / 4 SpD  
Bold Nature  
- Scald  
- Recover  
- Toxic  
- Haze
"""

EXADRILL_TEAM = """
Excadrill @ Leftovers  
Ability: Mold Breaker  
EVs: 252 Atk / 4 Def / 252 Spe  
Jolly Nature  
- Earthquake  
- Iron Head  
- Rapid Spin  
- Swords Dance
"""

HIPPOWDON_TEAM = """
Hippowdon @ Leftovers  
Ability: Sand Stream  
EVs: 252 HP / 252 Def / 4 SpD  
Impish Nature  
- Earthquake  
- Slack Off  
- Toxic  
- Whirlwind
"""

BRELOOM_TEAM = """
Breloom @ Life Orb  
Ability: Technician  
EVs: 252 Atk / 4 Def / 252 Spe  
Jolly Nature  
- Spore  
- Mach Punch  
- Bullet Seed  
- Swords Dance
"""

VOLCARONA_TEAM = """
Volcarona @ Heavy-Duty Boots  
Ability: Flame Body  
EVs: 248 HP / 252 SpA / 8 Spe  
Modest Nature  
- Fiery Dance  
- Bug Buzz  
- Roost  
- Quiver Dance
"""

ALL_SOLO_TEAMS = [
    ("steelix", STEELIX_TEAM),
    ("garchomp", GARCHOMP_TEAM),
    ("conkeldurr", CONKELDURR_TEAM),
    ("rotom_wash", ROTOM_WASH_TEAM),
    ("corviknight", CORVIKNIGHT_TEAM),
    ("toxapex", TOXAPEX_TEAM),
    ("excadrill", EXADRILL_TEAM),
    ("hippowdon", HIPPOWDON_TEAM),
    ("breloom", BRELOOM_TEAM),
    ("volcarona", VOLCARONA_TEAM),
]


def shuffled_team_generator(teams: Sequence[str]) -> Iterator[str]:
    if not teams:
        raise ValueError("teams must contain at least one team")

    pool = list(teams)
    while True:
        random.shuffle(pool)
        for team in pool:
            yield team
