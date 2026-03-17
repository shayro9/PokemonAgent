import numpy as np
from poke_env.battle.status import Status


BOOST_NORM      = 6.0
GEN1_BOOST_KEYS = ["atk", "def", "spa", "spe", "accuracy", "evasion"]
ALL_STATUSES    = list(Status)

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

    vec = np.asarray(list(vec), dtype=np.float32)

    y = (vec / vec_max)

    if symmetric:
        y = np.clip(y, -1.0, 1.0)
    else:
        y = np.clip(y, 0.0, 1.0)

    return y.astype(np.float32)

def encode_enum(value, enums_list) -> np.ndarray:
    """One-hot encode an enum condition over enums_list.

    :param value: value to encode
    :param enums_list: enums to encode
    :returns: Float32 one-hot vector of length ``len(enums_list)``.
    """
    if enums_list is None:
        raise ValueError("enums_list cannot be None")
    if value is not None:
        return np.array(
            [1.0 if value == s else 0.0 for s in enums_list],
            dtype=np.float32,
        )
    else:
        return np.zeros(len(enums_list), dtype=np.float32)

#############################################################
#############################################################
############# Tell me if you think is relevant ##############
############# | | | | | | | | | | | | | | | |  ##############
############# V V V V V V V V V V V V V V V V  ##############
# it's like .values() but making sure every item is in keys #
def encode_dicts(_dict: dict, _keys: list[str]) -> np.ndarray:
    """Extract raw boost stage values in the given key order.

    :param _dict: dictionary to encode
    :param _keys: Ordered list of keys to extract.
    :returns: Float32 array of raw values stage values.
    """
    return np.array(
        [_dict.get(k, 0) for k in _keys],
        dtype=np.float32,
    )
