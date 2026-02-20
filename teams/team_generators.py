import json
import random
from typing import List, Optional


def load_pokemon_pool(data_path: str) -> list[dict]:
    with open(data_path, 'r', encoding='utf-8') as f:
        pokemon_pool = json.load(f)['pool']

    if not pokemon_pool:
        raise ValueError("The database is empty. Run the Node.js script first!")

    return pokemon_pool


def split_pokemon_pool(
        pokemon_pool: list[dict],
        train_fraction: float,
        seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    shuffled_pool = list(pokemon_pool)
    random.Random(seed).shuffle(shuffled_pool)
    split_index = int(len(shuffled_pool) * train_fraction)

    if split_index <= 0 or split_index >= len(shuffled_pool):
        raise ValueError(
            "train_fraction created an empty train or eval split; use a larger pool or adjust the split."
        )

    return shuffled_pool[:split_index], shuffled_pool[split_index:]


def single_simple_team_generator(data_path: str | None = None, pokemon_pool: list[dict] | None = None):
    if pokemon_pool is None:
        if data_path is None:
            raise ValueError("Either data_path or pokemon_pool must be provided.")
        pokemon_pool = load_pokemon_pool(data_path)

    if not pokemon_pool:
        raise ValueError("The database is empty. Run the Node.js script first!")

    while True:
        sampled_mon = random.choice(pokemon_pool)

        yield generate_team(
            nickname=sampled_mon.get('name'),
            species=sampled_mon.get('species'),
            item=sampled_mon.get('item'),
            ability=sampled_mon.get('ability'),
            moves=sampled_mon.get('moves'),
            nature=sampled_mon.get('nature'),
            evs=format_stats_dict(sampled_mon.get('evs')),
            ivs=format_stats_dict(sampled_mon.get('ivs')),
            gender=sampled_mon.get('gender'),
            level=sampled_mon.get('level'),
            shiny=sampled_mon.get('shiny'),
            teratype=sampled_mon.get('teraType')
        )


def format_stats_dict(stats: Optional[dict]) -> str:
    """Helper to convert {hp: 252, atk: 0...} to '252,0,0,0,0,0'"""
    if not stats:
        return ""
    keys = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
    return ",".join(str(stats.get(k, 0)) for k in keys)


def generate_team(
        nickname: Optional[str] = None,
        species: str = "",
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
    display_species = "" if species == display_nickname else (species or "")
    display_item = item or ""
    display_ability = ability or ""
    display_moves = ",".join(moves) if moves else ""
    display_nature = nature or ""
    display_evs = evs or ""
    if gender in ["N", None, ""]:
        display_gender = ""
    else:
        display_gender = gender
    display_ivs = ivs or ""
    display_shiny = "S" if shiny else ""
    display_level = str(level) if (level and level != 100) else ""

    trailing = [
        str(happiness) if happiness and happiness != 255 else "",
        pokeball or "",
        hiddenpowertype or "",
        "G" if gigantamax else "",
        str(dynamaxlevel) if dynamaxlevel and dynamaxlevel != 10 else "",
        teratype or ""
    ]
    while trailing and not trailing[-1]:
        trailing.pop()

    trailing_part = ",".join(trailing)

    # Showdown Packed Format:
    # Nickname|Species|Item|Ability|Moves|Nature|EVs|Gender|IVs|Shiny|Level|Happiness,Pokeball,HPType,Gmax,Dmax,Tera
    team = f"{display_nickname}|{display_species}|{display_item}|{display_ability}|{display_moves}|{display_nature}|{display_evs}|{display_gender}|{display_ivs}|{display_shiny}|{display_level}|{trailing_part}"
    return team
