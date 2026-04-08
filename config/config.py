from typing import Iterable, Optional
from dataclasses import dataclass

from teams.generators import team_generator, matchup_generator, InfinitePoolGenerator
from data.prossesing import load_pool, split_pool


DEFAULT_DATA_PATH = "data/matchups_gen1ou_db.json"


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
    pokemon_pool = load_pool(data_path)

    train_pool, eval_pool = split_pool(
        pool=pokemon_pool,
        fraction=train_split,
        seed=split_seed,
    )

    print(
        f"Generated pool split: train={len(train_pool)} samples, eval={len(eval_pool)} samples (seed={split_seed}, "
        f"train_split={train_split})"
    )
    return train_pool, eval_pool


@dataclass(frozen=True)
class OpponentsResolved:
    train_gen: Optional[InfinitePoolGenerator]
    eval_gen: Optional[InfinitePoolGenerator]
    train_agent_gen: Optional[InfinitePoolGenerator]
    eval_agent_gen: Optional[InfinitePoolGenerator]
    train_battle_team_generator: Optional[InfinitePoolGenerator] = None
    eval_battle_team_generator: Optional[InfinitePoolGenerator] = None


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
    pool = load_pool(data_path)
    return pool, pool


def resolve_opponents(args) -> OpponentsResolved:
    """Resolve train/eval opponent names and generators from CLI args."""
    """Resolve train/eval opponent names and generators from CLI args."""
    if getattr(args, 'matchup_data_path', None):
        matchup_pool = load_pool(args.matchup_data_path)

        if args.split_generated_pool:
            train_pool, eval_pool = split_pool(
                pool=matchup_pool,
                fraction=args.train_split,
                seed=args.split_seed,
            )
            print(
                f"Matchup pool split: train={len(train_pool)}, eval={len(eval_pool)} "
                f"(seed={args.split_seed}, train_split={args.train_split})"
            )
        else:
            train_pool = eval_pool = matchup_pool

        return OpponentsResolved(
            train_gen=None,
            eval_gen=None,
            train_agent_gen=None,
            eval_agent_gen=None,
            train_battle_team_generator=matchup_generator(pool=train_pool, seed=args.seed),
            eval_battle_team_generator=matchup_generator(pool=eval_pool, seed=args.seed),
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
    train_gen = team_generator(pool=train_opp_pool, seed=train_seed)
    eval_gen = team_generator(pool=eval_opp_pool, seed=eval_seed)

    if agent_data_path == opponent_data_path:
        train_agent_gen = train_gen
        eval_agent_gen = eval_gen
    else:
        train_agent_gen = team_generator(pool=train_agent_pool, seed=train_seed)
        eval_agent_gen = team_generator(pool=eval_agent_pool, seed=eval_seed)

    return OpponentsResolved(
        train_gen=train_gen,
        eval_gen=eval_gen,
        train_agent_gen=train_agent_gen,
        eval_agent_gen=eval_agent_gen,
    )
