import json
import random

def load_pool(data_path: str) -> list[dict]:
    """Load the generated Pokémon pool JSON file.

    :param data_path: Path to the generated JSON dataset.
    :returns: The list stored in the dataset's ``pool`` field."""
    with open(data_path, 'r', encoding='utf-8') as f:
        _pool = json.load(f)['pool']

    if not _pool:
        raise ValueError("The database is empty. Run the Node.js script first!")

    return _pool

def split_pool(
        pool: list[dict],
        fraction: float,
        seed: int,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split a Pokémon pool into train and evaluation subsets.

    :param pool: Full list of entries.
    :param fraction: Fraction of entries allocated to the train split.
    :param seed: Random seed used for deterministic shuffling.
    :returns: A tuple of ``(train_pool, eval_pool)``."""
    if not 0 < fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    shuffled_pool = list(pool)
    random.Random(seed).shuffle(shuffled_pool)
    split_index = int(len(shuffled_pool) * fraction)

    if split_index <= 0 or split_index >= len(shuffled_pool):
        raise ValueError(
            "train_fraction created an empty train or eval split; use a larger pool or adjust the split."
        )

    return shuffled_pool[:split_index], shuffled_pool[split_index:]

def resolve_pool(data_path, pool, error_msg: str):
    if pool is None:
        if data_path is None:
            raise ValueError(error_msg)
        pool = load_pool(data_path)

    if not pool:
        raise ValueError("Pool is empty.")

    return pool