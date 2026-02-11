import json
import random
from typing import List, Optional, Dict


def single_simple_team_generator(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        valid_pokemon = [
            name for name, details in data.items()
            if len(details.get('moves', [])) >= 4 and details.get('abilities')
        ]

        if not valid_pokemon:
            raise ValueError("No Pokémon in the dataset have 4 or more moves.")

    while True:
        pokemon_name = random.choice(valid_pokemon)
        moves = random.sample(data[pokemon_name]['moves'], k=4)

        yield generate_team(species=pokemon_name, moves=moves)


def generate_team(
        *,
        nickname: Optional[str] = None,
        species: str,
        item: Optional[str] = None,
        ability: Optional[str] = None,
        moves: Optional[List[str]] = None,
        nature: Optional[str] = None,
        evs: Optional[str] = None,
        gender: Optional[str] = None,
        ivs: Optional[str] = None,
        shiny: Optional[bool] = None,
        level: Optional[int] = None,
        happiness: Optional[int] = None,
        pokeball: Optional[str] = None,
        hiddenpowertype: Optional[str] = None,
        gigantamax: Optional[bool] = None,
        dynamaxlevel: Optional[int] = None,
        teratype: Optional[str] = None,
) -> str:
    display_nickname = nickname or species
    display_species = "" if species == display_nickname else species

    display_item = item or ""
    display_ability = ability or "0"

    display_moves = ",".join(moves) if moves else ""

    display_nature = nature or ""

    if not evs or evs == "0,0,0,0,0,0" or evs == "":
        display_evs = "1"
    else:
        display_evs = evs

    display_gender = gender or ""

    if not ivs or ivs == "31,31,31,31,31,31":
        display_ivs = ""
    else:
        display_ivs = ivs

    display_shiny = "S" if shiny else ""

    display_level = "" if level == 100 else (str(level) if level is not None else "")
    display_happiness = "" if happiness == 255 else (str(happiness) if happiness is not None else "")

    display_pokeball = "" if pokeball == "Poké Ball" else (pokeball or "")
    display_hptype = hiddenpowertype or ""
    display_gmax = "G" if gigantamax else ""
    display_dmaxlv = "" if dynamaxlevel == 10 else (str(dynamaxlevel) if dynamaxlevel is not None else "")
    display_tera = teratype or ""

    trailing_fields = [display_pokeball, display_hptype, display_gmax, display_dmaxlv, display_tera]

    if not any(trailing_fields):
        trailing_part = display_happiness
    else:
        trailing_part = f"{display_happiness},{display_pokeball},{display_hptype},{display_gmax},{display_dmaxlv},{display_tera}"

    return f"{display_nickname}|{display_species}|{display_item}|{display_ability}|{display_moves}|{display_nature}|{display_evs}|{display_gender}|{display_ivs}|{display_shiny}|{display_level}|{trailing_part}]"
