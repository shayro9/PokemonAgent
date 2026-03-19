import json
import random
from typing import List, Optional, Iterator


# ---------------------------------------------------------------------------
# Stats / field helpers
# ---------------------------------------------------------------------------

def format_stats(stats: Optional[dict]) -> str:
    """Convert a stats dict into packed EV/IV CSV order (hp,atk,def,spa,spd,spe)."""
    if not stats:
        return ""
    return ",".join(str(stats.get(k, 0)) for k in ["hp", "atk", "def", "spa", "spd", "spe"])


def mon_to_kwargs(mon: dict) -> dict:
    """Extract :func:`build_slot` kwargs from a raw pool/matchup entry.

    Normalises Gen 1 ``'No Ability'`` → ``''`` so Showdown accepts the packed team.
    """
    ability = mon.get("ability") or ""
    if ability.lower() in ("no ability", "noability"):
        ability = ""
    return dict(
        nickname=mon.get("name"),
        species=mon.get("species"),
        item=mon.get("item"),
        ability=ability,
        moves=mon.get("moves"),
        nature=mon.get("nature"),
        evs=format_stats(mon.get("evs")),
        ivs=format_stats(mon.get("ivs")),
        gender=mon.get("gender"),
        level=mon.get("level"),
        shiny=mon.get("shiny"),
        teratype=mon.get("teraType"),
    )


def build_slot(
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
    """Build a single-slot Showdown packed-team string.

    Format: ``Nickname|Species|Item|Ability|Moves|Nature|EVs|Gender|IVs|Shiny|Level|Trailing``

    :returns: One packed slot — no ``]`` separator.
    """
    display_nickname = nickname or species
    display_species  = "" if species == display_nickname else (species or "")
    display_moves    = ",".join(moves) if moves else ""
    display_gender   = gender if gender not in ("N", None, "") else ""
    display_level    = str(level) if (level and level != 100) else ""
    display_shiny    = "S" if shiny else ""

    trailing = [
        str(happiness) if happiness and happiness != 255 else "",
        pokeball or "",
        hiddenpowertype or "",
        "G" if gigantamax else "",
        str(dynamaxlevel) if dynamaxlevel and dynamaxlevel != 10 else "",
        teratype or "",
    ]
    while trailing and not trailing[-1]:
        trailing.pop()

    return (
        f"{display_nickname}|{display_species}|{item or ''}|{ability or ''}|"
        f"{display_moves}|{nature or ''}|{evs or ''}|{display_gender}|"
        f"{ivs or ''}|{display_shiny}|{display_level}|{','.join(trailing)}"
    )


# ---------------------------------------------------------------------------
# Pool I/O
# ---------------------------------------------------------------------------

def load_pool(data_path: str) -> list[dict]:
    """Load and return the ``pool`` list from a generated JSON dataset."""
    with open(data_path, "r", encoding="utf-8") as f:
        pool = json.load(f)["pool"]
    if not pool:
        raise ValueError("The database is empty. Run the Node.js script first!")
    return pool


def split_pool(
    pool: list[dict],
    train_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split *pool* into ``(train_pool, eval_pool)``.

    :param train_fraction: Fraction allocated to training (0 < f < 1).
    :param seed: Random seed for deterministic shuffling.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    split = int(len(shuffled) * train_fraction)

    if split <= 0 or split >= len(shuffled):
        raise ValueError(
            "train_fraction produced an empty split. "
            "Use a larger pool or adjust the fraction."
        )
    return shuffled[:split], shuffled[split:]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_team(pool: list[dict], n: int = 1, side: str = "agent") -> str:
    """Sample *n* distinct Pokémon from *pool* and return a packed team string.

    Multiple Pokémon are joined with ``]`` as Showdown expects.

    :param pool: Pool loaded via :func:`load_pool`.
    :param n: Number of Pokémon (1–6).
    :param side: Key inside each matchup entry (``'agent'`` or ``'opponent'``).
                 Falls back to the entry itself for non-matchup pools.
    :raises ValueError: If *n* is outside 1–6.
    """
    if not 1 <= n <= 6:
        raise ValueError(f"n must be between 1 and 6, got {n}")
    entries = random.sample(pool, min(n, len(pool)))
    return "]".join(build_slot(**mon_to_kwargs(e.get(side, e))) for e in entries)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def iter_teams(
    pool: list[dict],
    side: str = "agent",
    n: int = 1,
    seed: Optional[int] = None,
) -> Iterator[str]:
    """Yield packed team strings indefinitely.

    :param pool: Pool loaded via :func:`load_pool`.
    :param side: ``'agent'`` or ``'opponent'``.
    :param n: Pokémon per team (1–6).
    :param seed: Optional random seed.
    """
    rng = random.Random(seed)
    while True:
        entries = rng.sample(pool, min(n, len(pool)))
        yield "]".join(build_slot(**mon_to_kwargs(e.get(side, e))) for e in entries)


def iter_matchups(
    pool: list[dict],
    seed: Optional[int] = None,
    shuffle_each_epoch: bool = True,
) -> Iterator[tuple[str, str]]:
    """Yield ``(agent_packed, opponent_packed)`` pairs indefinitely.

    :param pool: Matchup pool loaded via :func:`load_pool`.
    :param seed: Optional random seed.
    :param shuffle_each_epoch: Reshuffle at the start of each pass.
    """
    rng = random.Random(seed)
    while True:
        order = list(pool)
        if shuffle_each_epoch:
            rng.shuffle(order)
        for matchup in order:
            yield (
                build_slot(**mon_to_kwargs(matchup["agent"])),
                build_slot(**mon_to_kwargs(matchup["opponent"])),
            )


# ---------------------------------------------------------------------------
# Backward-compatible aliases  (existing call-sites keep working unchanged)
# ---------------------------------------------------------------------------

def format_stats_dict(stats: Optional[dict]) -> str:
    return format_stats(stats)

def _mon_kwargs(mon: dict) -> dict:
    return mon_to_kwargs(mon)

def generate_team(**kwargs) -> str:
    return build_slot(**kwargs)

def load_pokemon_pool(data_path: str) -> list[dict]:
    return load_pool(data_path)

def split_pokemon_pool(pool: list[dict], train_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    return split_pool(pool, train_fraction, seed)

def sample_team_of_n(pool: list[dict], n: int, side: str = "agent") -> str:
    return sample_team(pool, n, side)

def single_simple_team_generator(
    data_path: Optional[str] = None,
    pokemon_pool: Optional[list[dict]] = None,
    seed: Optional[int] = None,
) -> Iterator[str]:
    if pokemon_pool is None:
        if data_path is None:
            raise ValueError("Either data_path or pokemon_pool must be provided.")
        pokemon_pool = load_pool(data_path)
    return iter_teams(pokemon_pool, seed=seed)

def matchup_generator(
    data_path: Optional[str] = None,
    matchup_pool: Optional[list[dict]] = None,
    seed: Optional[int] = None,
    shuffle_each_epoch: bool = True,
) -> Iterator[tuple[str, str]]:
    if matchup_pool is None:
        if data_path is None:
            raise ValueError("Either data_path or matchup_pool must be provided.")
        matchup_pool = load_pool(data_path)
    print(f"[matchup_generator] Using {len(matchup_pool)} matchups")
    return iter_matchups(matchup_pool, seed=seed, shuffle_each_epoch=shuffle_each_epoch)
