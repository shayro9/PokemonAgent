import json
import random
from typing import List, Optional


def load_pokemon_pool(data_path: str) -> list[dict]:
    """Load the generated Pokémon pool JSON file.
    
    :param data_path: Path to the generated JSON dataset.
    :returns: The list stored in the dataset's ``pool`` field."""
    with open(data_path, 'r', encoding='utf-8') as f:
        pokemon_pool = json.load(f)['pool']

    if not pokemon_pool:
        raise ValueError("The database is empty. Run the Node.js script first!")

    return pokemon_pool

def _mon_kwargs(mon: dict) -> dict:
    """Extract generate_team kwargs from a pool/matchup entry."""
    return dict(
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


def split_pokemon_pool(
        pokemon_pool: list[dict],
        train_fraction: float,
        seed: int,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split a Pokémon pool into train and evaluation subsets.
    
    :param pokemon_pool: Full list of generated Pokémon entries.
    :param train_fraction: Fraction of entries allocated to the train split.
    :param seed: Random seed used for deterministic shuffling.
    :returns: A tuple of ``(train_pool, eval_pool)``."""
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


def single_simple_team_generator(
        data_path: str | None = None,
        pokemon_pool: list[dict] | None = None,
        seed: int | None = None,
):
    """Yield packed Showdown teams sampled from the provided pool.
    
    :param data_path: Optional path to a generated dataset file.
    :param pokemon_pool: Optional in-memory pool of Pokémon entries.
    :param seed: Optional random seed for deterministic sampling.
    :returns: An infinite generator of packed team strings."""
    if pokemon_pool is None:
        if data_path is None:
            raise ValueError("Either data_path or pokemon_pool must be provided.")
        pokemon_pool = load_pokemon_pool(data_path)

    if not pokemon_pool:
        raise ValueError("The database is empty. Run the Node.js script first!")

    rng = random.Random(seed) if seed is not None else random

    while True:
        sampled_mon = rng.choice(pokemon_pool)
        yield generate_team(**_mon_kwargs(sampled_mon))


def matchup_generator(
        data_path: str = None,
        matchup_pool: list[dict] | None = None,
        seed: int | None = None,
        shuffle_each_epoch: bool = True,
):
    if matchup_pool is None:
        if data_path is None:
            raise ValueError("Either data_path or matchup_pool must be provided.")
        with open(data_path) as f:
            matchup_pool = json.load(f)['pool']

    if not matchup_pool:
        raise ValueError("Matchup pool is empty.")

    print(f"[matchup_generator] Using {len(matchup_pool)} matchups")
    rng = random.Random(seed)

    while True:
        if shuffle_each_epoch:
            rng.shuffle(matchup_pool)
        for matchup in matchup_pool:
            yield (
                generate_team(**_mon_kwargs(matchup['agent'])),
                generate_team(**_mon_kwargs(matchup['opponent'])),
            )


def format_stats_dict(stats: Optional[dict]) -> str:
    """Convert a stats mapping into packed EV/IV CSV order.
    
    :param stats: Mapping with stat keys (hp, atk, def, spa, spd, spe).
    :returns: Comma-separated stat values in canonical showdown order."""
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
    """Build a Showdown packed-team string from team member attributes.
    
    :param nickname: Display nickname of the Pokémon.
    :param species: Base species name.
    :param item: Held item name.
    :param ability: Ability name.
    :param moves: Ordered move names.
    :param nature: Nature name.
    :param evs: Comma-separated EV values.
    :param gender: Gender marker.
    :param ivs: Comma-separated IV values.
    :param shiny: Whether the Pokémon is shiny.
    :param level: Level value.
    :param happiness: Happiness value.
    :param pokeball: Pokéball name.
    :param hiddenpowertype: Hidden Power type.
    :param gigantamax: Whether the Pokémon is Gigantamax-capable.
    :param dynamaxlevel: Dynamax level.
    :param teratype: Tera type.
    :returns: A packed Showdown team segment string."""
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
    team = (f"{display_nickname}|{display_species}|{display_item}|{display_ability}|{display_moves}|{display_nature}|"
            f"{display_evs}|{display_gender}|{display_ivs}|{display_shiny}|{display_level}|{trailing_part}")
    return team
