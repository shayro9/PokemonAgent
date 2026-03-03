from typing import Iterable, Optional
from dataclasses import dataclass

from teams.single_teams import ALL_SOLO_TEAMS
from teams.team_generators import single_simple_team_generator, load_pokemon_pool, split_pokemon_pool, matchup_generator


TEAM_BY_NAME = {name: team for name, team in ALL_SOLO_TEAMS}
DEFAULT_DATA_PATH = "data/gen9randombattle_db.json"


def parse_pool(raw_pool: str | None, pool_all: bool) -> list[str]:
    """Parse and validate a comma-separated pool of opponent names.
    
    :param raw_pool: Comma-separated opponent names.
    :param pool_all: Whether all predefined solo teams should be used.
    :returns: A list of validated opponent names."""
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
    """Resolve the effective seed value.
    
    :param explicit: User-provided seed, if any.
    :param fallback: Default seed to use when ``explicit`` is ``None``.
    :returns: The effective seed value."""
    return fallback if explicit is None else explicit


def _resolve_generated_pools(
        data_path: str,
        train_split: float,
        split_seed: int,
) -> tuple[list[dict], list[dict]]:
    """Load generated data and split it into train and evaluation pools.
    
    :param data_path: Path to the generated dataset.
    :param train_split: Fraction of examples used for training.
    :param split_seed: Random seed used for deterministic splitting.
    :returns: A tuple of ``(train_pool, eval_pool)`` lists."""
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
    battle_team_generator: Optional[Iterable] = None


def _resolve_train_eval_pools(
    *,
    data_path: str,
    do_split: bool,
    train_split: float,
    split_seed: int,
):
    """Return (train_pool, eval_pool). If not split -> same pool for both."""
    if do_split:
        return _resolve_generated_pools(
            data_path=data_path,
            train_split=train_split,
            split_seed=split_seed,
        )
    pool = load_pokemon_pool(data_path)
    return pool, pool


def resolve_opponents(args) -> OpponentsResolved:
    """Resolve train/eval opponent names and generators from CLI args."""
    train_names: list[str] = []
    eval_names: list[str] = []
    train_gen = None
    eval_gen = None
    train_agent_gen = None
    eval_agent_gen = None

    if getattr(args, 'matchup_data_path', None):
        matchup_pool = load_pokemon_pool(args.matchup_data_path)

        if args.split_generated_pool:
            train_pool, eval_pool = split_pokemon_pool(
                pokemon_pool=matchup_pool,
                train_fraction=args.train_split,
                seed=args.split_seed,
            )
            print(
                f"Matchup pool split: train={len(train_pool)}, eval={len(eval_pool)} "
                f"(seed={args.split_seed}, train_split={args.train_split})"
            )
        else:
            train_pool = eval_pool = matchup_pool

        return OpponentsResolved(
            train_names=[],
            eval_names=[],
            train_gen=None,
            eval_gen=matchup_generator(matchup_pool=eval_pool, seed=args.seed + 1),
            train_agent_gen=None,
            eval_agent_gen=None,
            battle_team_generator=matchup_generator(matchup_pool=train_pool, seed=args.seed),
        )

    if not args.random_generated:
        # ---- NAME-BASED TRAIN opponents ----
        train_names = parse_pool(args.pool, args.pool_all)

        # ---- NAME-BASED EVAL opponents ----
        if args.eval_pool is not None or args.eval_pool_all:
            eval_names = parse_pool(args.eval_pool, args.eval_pool_all)
        else:
            eval_names = train_names

        # Generators must be None in name-based mode
        return OpponentsResolved(
            train_names=train_names,
            eval_names=eval_names,
            train_gen=None,
            eval_gen=None,
            train_agent_gen=None,
            eval_agent_gen=None,
        )

    # ---- GENERATED MODE ----
    train_seed = resolve_seed(args.train_generator_seed, args.seed)
    eval_seed = resolve_seed(args.eval_generator_seed, args.seed)

    # “none” means default
    agent_data_path = args.agent_data_path or DEFAULT_DATA_PATH
    opponent_data_path = args.opponent_data_path or DEFAULT_DATA_PATH

    do_split = bool(args.split_generated_pool)

    # Resolve pools with “same path” optimization
    if agent_data_path == opponent_data_path:
        train_pool, eval_pool = _resolve_train_eval_pools(
            data_path=agent_data_path,
            do_split=do_split,
            train_split=args.train_split,
            split_seed=args.split_seed,
        )
        train_agent_pool = train_opp_pool = train_pool
        eval_agent_pool = eval_opp_pool = eval_pool
    else:
        train_agent_pool, eval_agent_pool = _resolve_train_eval_pools(
            data_path=agent_data_path,
            do_split=do_split,
            train_split=args.train_split,
            split_seed=args.split_seed,
        )
        train_opp_pool, eval_opp_pool = _resolve_train_eval_pools(
            data_path=opponent_data_path,
            do_split=do_split,
            train_split=args.train_split,
            split_seed=args.split_seed,
        )

    # Opponent generators always exist in generated mode
    train_gen = single_simple_team_generator(pokemon_pool=train_opp_pool, seed=train_seed)
    eval_gen = single_simple_team_generator(pokemon_pool=eval_opp_pool, seed=eval_seed)

    # Agent generators only when you did NOT provide an explicit --train-team
    if args.train_team is None:
        if agent_data_path == opponent_data_path:
            train_agent_gen = train_gen
            eval_agent_gen = eval_gen
        else:
            train_agent_gen = single_simple_team_generator(pokemon_pool=train_agent_pool, seed=train_seed)
            eval_agent_gen = single_simple_team_generator(pokemon_pool=eval_agent_pool, seed=eval_seed)

    # Names are irrelevant in generated mode, but keep them consistent/empty
    if args.eval_pool is not None or args.eval_pool_all:
        eval_names = parse_pool(args.eval_pool, args.eval_pool_all)
        eval_gen = None
    else:
        eval_names = train_names  # usually empty

    return OpponentsResolved(
        train_names=train_names,
        eval_names=eval_names,
        train_gen=train_gen,
        eval_gen=eval_gen,
        train_agent_gen=train_agent_gen,
        eval_agent_gen=eval_agent_gen,
    )
