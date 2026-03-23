from typing import Optional


def format_stats_dict(stats: Optional[dict]) -> str:
    """Convert a stats mapping into packed EV/IV CSV order.

    :param stats: Mapping with stat keys (hp, atk, def, spa, spd, spe).
    :returns: Comma-separated stat values in canonical showdown order."""
    if not stats:
        return ""
    keys = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
    return ",".join(str(stats.get(k, 0)) for k in keys)