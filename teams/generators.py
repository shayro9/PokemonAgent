import random

from data.prossesing import resolve_pool
from teams.showdown_pokemon import team_from_dict


class InfinitePoolGenerator:
    def __init__(
        self,
        pool: list[dict],
        transform_fn,
        seed: int | None = None,
        shuffle_each_epoch: bool = True,
    ):
        self._pool = list(pool)
        self._transform_fn = transform_fn
        self._seed = seed
        self._shuffle_each_epoch = shuffle_each_epoch
        self._gen = self._make_generator()

    def _make_generator(self):
        rng = random.Random(self._seed) if self._seed is not None else random
        local_pool = list(self._pool)

        while True:
            if self._shuffle_each_epoch:
                rng.shuffle(local_pool)
            for entry in local_pool:
                yield self._transform_fn(entry)

    def reset(self):
        """Restart the sequence from the beginning using the original seed."""
        self._gen = self._make_generator()

    @property
    def team_size(self) -> int:
        """Return the number of Pokémon in a team."""
        if not self._pool:
            return 0
        entry = self._pool[0]
        if isinstance(entry, dict) and "agent" in entry:
            return len(entry["agent"])
        return len(entry)

    def __next__(self):
        return next(self._gen)

    def __iter__(self):
        return self


def team_generator(data_path=None, pool=None, **kwargs):
    pool = resolve_pool(data_path, pool, "Provide data_path or pool.")

    return InfinitePoolGenerator(
        pool,
        transform_fn=lambda e: team_from_dict(e).to_showdown(),
        **kwargs,
    )


def matchup_generator(data_path=None, pool=None, **kwargs):
    pool = resolve_pool(data_path, pool, "Provide data_path or pool.")

    def transform_matchup(entry):
        return (
            team_from_dict(entry["agent"]).to_showdown(),
            team_from_dict(entry["opponent"]).to_showdown(),
        )

    return InfinitePoolGenerator(pool, transform_fn=transform_matchup, **kwargs)