import numpy as np
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BOOST_NORM      = 6.0
STAT_NORM   = 600.0
STAB_NORM   = 2.25

ALL_STATUSES    = list(Status)

GEN1_TRACKED_EFFECTS = [Effect.CONFUSION, Effect.ENCORE]
GEN1_BOOST_KEYS = ["atk", "def", "spa", "spe", "accuracy", "evasion"]
GEN1_STAT_KEYS      = ["hp", "atk", "def", "spc", "spe"]

MODERN_STAT_KEYS    = ["hp", "atk", "def", "spa", "spd", "spe"]
MODERN_BOOST_KEYS   = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

MAX_TEAM_SIZE = 6
MAX_MOVES = 4

def normalize(x: float, max_x: float = 1.0, symmetric: bool = False) -> float:
    if max_x <= 0:
        return 0.0
    y = x / max_x
    if symmetric:
        return max(-1.0, min(1.0, y))
    return min(1.0, max(0.0, y))

def normalize_vector(vec, vec_max, symmetric: bool = False) -> np.ndarray:
    if np.any(vec_max <= 0):
        return np.zeros_like(vec, dtype=np.float32)

    y = (vec / vec_max)

    if symmetric:
        y = np.clip(y, -1.0, 1.0)
    else:
        y = np.clip(y, 0.0, 1.0)

    return y.astype(np.float32)

def encode_enum(value, enums_list) -> np.ndarray:
    """Encode an enum condition over enums_list.

    - ``None``            → all-zero vector (unknown / no value).
    - Single enum value   → one-hot vector  (e.g. a Status).
    - Collection (dict/set/list) → multi-hot vector, one bit per member
                           (e.g. pokemon.effects which may have several
                           active Effects at once).

    :param value: ``None``, a single enum member, or a collection of enum members.
    :param enums_list: ordered list of enum members to encode against.
    :returns: Float32 binary vector of length ``len(enums_list)``.
    """
    if enums_list is None:
        raise ValueError("enums_list cannot be None")
    if value is None:
        return np.zeros(len(enums_list), dtype=np.float32)
    if isinstance(value, (dict, set, list, frozenset)):
        return np.array(
            [1.0 if s in value else 0.0 for s in enums_list],
            dtype=np.float32,
        )
    return np.array(
        [1.0 if value == s else 0.0 for s in enums_list],
        dtype=np.float32,
    )

def encode_dicts(_dict: dict, _keys: list[str]) -> np.ndarray:
    """Extract raw boost stage values in the given key order.

    :param _dict: dictionary to encode
    :param _keys: Ordered list of keys to extract.
    :returns: Float32 array of raw values stage values.
    """
    return np.array(
        [_dict.get(k) or 0 for k in _keys],
        dtype=np.float32,
    )

def pull_attribute(obj, key, default_value, type_value):
    if obj is None or key is None:
        return type_value(default_value)
    val = getattr(obj, key, default_value)
    return type_value(val) if val is not None else default_value
