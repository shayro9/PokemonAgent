import random

from data.prossesing import resolve_pool
from teams.showdown_pokemon import team_from_dict


def _infinite_pool_generator(
    pool: list[dict],
    transform_fn,
    seed: int | None = None,
    shuffle_each_epoch: bool = True,
):
    rng = random.Random(seed) if seed is not None else random
    local_pool = list(pool)

    while True:
        if shuffle_each_epoch:
            rng.shuffle(local_pool)

        for entry in local_pool:
            yield transform_fn(entry)


def team_generator(data_path=None, pool=None, **kwargs):
    pool = resolve_pool(data_path, pool, "Provide data_path or pool.")

    return _infinite_pool_generator(
        pool,
        transform_fn=lambda e: team_from_dict(e).to_showdown(),
        **kwargs
    )


def matchup_generator(data_path=None, pool=None, **kwargs):
    pool = resolve_pool(data_path, pool, "Provide data_path or pool.")

    def transform_matchup(entry):
        return (
            team_from_dict(entry["agent"]).to_showdown(),
            team_from_dict(entry["opponent"]).to_showdown(),
        )

    return _infinite_pool_generator(pool, transform_fn=transform_matchup, **kwargs)