"""Registry and constructors for opponent Player implementations."""

from importlib import import_module

from poke_env import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player.player import Player

from curriculum.models import OpponentPlayerSpec


_RESERVED_PLAYER_KWARGS = {
    "account_configuration",
    "battle_format",
    "server_configuration",
}

_OPPONENT_PLAYER_REGISTRY: dict[str, type[Player]] = {
    "heuristic": SimpleHeuristicsPlayer,
    "max-power": MaxBasePowerPlayer,
    "random": RandomPlayer,
}


def register_opponent_player(name: str, player_cls: type[Player]) -> None:
    """Register a reusable opponent Player class."""
    if not name.strip():
        raise ValueError("Opponent player registry name must be non-empty.")
    if not issubclass(player_cls, Player):
        raise TypeError("Registered opponent player must inherit from poke_env.player.Player.")
    _OPPONENT_PLAYER_REGISTRY[name] = player_cls


def get_registered_opponent_players() -> dict[str, type[Player]]:
    """Return a copy of the registered built-in/custom player classes."""
    return dict(_OPPONENT_PLAYER_REGISTRY)


def resolve_opponent_player_class(spec: OpponentPlayerSpec) -> type[Player]:
    """Resolve a Player class from a curriculum spec."""
    if spec.id is not None:
        try:
            return _OPPONENT_PLAYER_REGISTRY[spec.id]
        except KeyError as exc:
            choices = ", ".join(sorted(_OPPONENT_PLAYER_REGISTRY))
            raise ValueError(
                f"Unknown opponent player id '{spec.id}'. Registered ids: {choices}"
            ) from exc

    module_path, separator, attr_name = spec.class_path.rpartition(".")
    if not separator:
        raise ValueError(
            f"Invalid class_path '{spec.class_path}'. Expected 'module.submodule.ClassName'."
        )

    module = import_module(module_path)
    player_cls = getattr(module, attr_name)
    if not isinstance(player_cls, type) or not issubclass(player_cls, Player):
        raise TypeError(
            f"Resolved class '{spec.class_path}' must inherit from poke_env.player.Player."
        )
    return player_cls


def build_opponent_player(
    spec: OpponentPlayerSpec,
    *,
    battle_format: str,
    server_configuration,
    account_configuration,
) -> Player:
    """Instantiate an opponent Player from a curriculum spec."""
    reserved = _RESERVED_PLAYER_KWARGS.intersection(spec.kwargs)
    if reserved:
        reserved_args = ", ".join(sorted(reserved))
        raise ValueError(
            f"Opponent player kwargs cannot override framework-managed arguments: {reserved_args}"
        )

    player_cls = resolve_opponent_player_class(spec)
    return player_cls(
        account_configuration=account_configuration,
        battle_format=battle_format,
        server_configuration=server_configuration,
        **spec.kwargs,
    )
