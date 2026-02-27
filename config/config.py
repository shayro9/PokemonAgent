from typing import Iterable
from dataclasses import dataclass

from teams.single_teams import ALL_SOLO_TEAMS
from teams.team_generators import *


TEAM_BY_NAME = {name: team for name, team in ALL_SOLO_TEAMS}
DEFAULT_DATA_PATH = "data/gen9randombattle_db.json"


def parse_pool(raw_pool: str | None, pool_all: bool) -> list[str]:
    if pool_all:
        return [name for name, _ in ALL_SOLO_TEAMS]

    if not raw_pool:
        return []

    names = [name.strip() for name in raw_pool.split(",") if name.strip()]
    unknown = [name for name in names if name not in TEAM_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown pokemon(s) in pool: {', '.join(unknown)}. "
            f"Valid names: {', '.join(sorted(TEAM_BY_NAME))}"
        )
    return names


def resolve_seed(explicit: Optional[int], fallback: int) -> int:
    return fallback if explicit is None else explicit


def _resolve_generated_pools(
        data_path: str,
        train_split: float,
        split_seed: int,
) -> tuple[list[dict], list[dict]]:
    pokemon_pool = load_pokemon_pool(data_path)

    train_pool, eval_pool = split_pokemon_pool(
        pokemon_pool=pokemon_pool,
        train_fraction=train_split,
        seed=split_seed,
    )

    print(
        f"Generated pool split: train={len(train_pool)} samples, eval={len(eval_pool)} samples (seed={split_seed}, "
        f"train_split={train_split})"
    )
    return train_pool, eval_pool


@dataclass(frozen=True)
class OpponentsResolved:
    train_names: list[str]
    eval_names: list[str]
    train_gen: Optional[Iterable]
    eval_gen: Optional[Iterable]
    train_agent_gen: Optional[Iterable]
    eval_agent_gen: Optional[Iterable]


def resolve_opponents(args) -> OpponentsResolved:
    """
    Single source of truth for:
      - train opponent names or generator
      - eval opponent names or generator
    """

    # ---- TRAIN opponents ----
    train_names: list[str] = []
    train_gen = None
    train_agent_gen = None
    eval_agent_gen = None

    if args.random_generated:
        # build generators (possibly split)
        train_seed = resolve_seed(args.train_generator_seed, args.seed)
        eval_seed = resolve_seed(args.eval_generator_seed, args.seed)

        agent_data_path = args.agent_data_path or DEFAULT_DATA_PATH
        opponent_data_path = args.opponent_data_path or DEFAULT_DATA_PATH

        if args.split_generated_pool:
            if agent_data_path == opponent_data_path:
                train_pool, eval_pool = _resolve_generated_pools(
                    data_path=agent_data_path,
                    train_split=args.train_split,
                    split_seed=args.split_seed,
                )
                train_opponent_pool = train_pool
                eval_opponent_pool = eval_pool
            else:
                train_opponent_pool, eval_opponent_pool = _resolve_generated_pools(
                    data_path=opponent_data_path,
                    train_split=args.train_split,
                    split_seed=args.split_seed,
                )
        else:
            shared_agent_pool = load_pokemon_pool(agent_data_path)
            shared_opponent_pool = (
                shared_agent_pool
                if agent_data_path == opponent_data_path
                else load_pokemon_pool(opponent_data_path)
            )
            train_opponent_pool = eval_opponent_pool = shared_opponent_pool

        train_gen = single_simple_team_generator(pokemon_pool=train_opponent_pool, seed=train_seed)
        eval_gen = single_simple_team_generator(pokemon_pool=eval_opponent_pool, seed=eval_seed)

        if args.train_team is None:
            train_agent_gen = train_gen
            eval_agent_gen = eval_gen
    else:
        # name-based pools
        train_names = parse_pool(args.pool, args.pool_all)
        eval_gen = None

    # ---- EVAL opponents ----
    if args.eval_pool is not None or args.eval_pool_all:
        eval_names = parse_pool(args.eval_pool, args.eval_pool_all)
    else:
        # Otherwise:
        # - if training used names => eval uses same names
        # - if training used generated => eval will use generator (handled above)
        eval_names = train_names

    # if not generated, eval_gen must be None
    if not args.random_generated:
        eval_gen = None

    return OpponentsResolved(
        train_names=train_names,
        eval_names=eval_names,
        train_gen=train_gen,
        eval_gen=eval_gen,
        train_agent_gen=train_agent_gen,
        eval_agent_gen=eval_agent_gen,
    )
