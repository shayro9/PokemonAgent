"""A poke_env Player that picks moves using a saved MaskablePPO policy."""
import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from sb3_contrib import MaskablePPO

from env.battle_config import BattleConfig
from env.singles_env_wrapper import PokemonRLWrapper


class PolicyPlayer(Player):
    """A Showdown player driven by a trained MaskablePPO policy.

    :param model_path: Path to the saved ``.zip`` model file.
    :param wrapper: A :class:`~env.singles_env_wrapper.PokemonRLWrapper`
        used as a stateless helper for embedding and action decoding. When
        omitted, policy inference falls back to poke-env's static helpers and
        does not create extra Showdown clients.
    :param battle_config: Generation config used to build observations.
        Defaults to Gen 1.
    :param deterministic: Use deterministic action selection (default True).
    :param verbose: Print the full BattleState description on every turn.
    """

    def __init__(
        self,
        model_path: str,
        wrapper: PokemonRLWrapper | None = None,
        *,
        battle_config: BattleConfig | None = None,
        deterministic: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._battle_config = battle_config if battle_config is not None else BattleConfig.gen1()
        self.model: MaskablePPO = MaskablePPO.load(model_path)
        self._wrapper = wrapper
        self._deterministic = deterministic
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def choose_move(self, battle):
        if self._verbose:
            turn = getattr(battle, "turn", "?")
            print(f"\n{'='*60}")
            print(f"  Turn {turn}")
            print('='*60)
            print(self._battle_config.battle_state_cls(battle).describe())

        if self._must_choose_default(battle):
            return SinglesEnv.action_to_order(np.int64(-2), battle, strict=False)

        action_mask = self._get_action_mask(battle)
        observation = {
            "observation": self._battle_config.battle_state_cls(battle).to_array(),
            "action_mask": action_mask,
        }
        action, _ = self.model.predict(
            observation,
            deterministic=self._deterministic,
            action_masks=observation["action_mask"],
        )

        order = self._action_to_order(action, battle)

        if self._verbose:
            print(f"\n→ Chosen action: {action}  →  {order}")

        return order

    def _get_action_mask(self, battle) -> np.ndarray:
        if self._wrapper is not None:
            return self._wrapper.get_action_mask(battle)
        return np.asarray(SinglesEnv.get_action_mask(battle), dtype=bool)

    @staticmethod
    def _must_choose_default(battle) -> bool:
        if getattr(battle, "_wait", False):
            return True

        valid_orders = getattr(battle, "valid_orders", None)
        if not valid_orders:
            return False

        return all(str(order) == "/choose default" for order in valid_orders)

    def _action_to_order(self, action, battle):
        if self._wrapper is not None:
            return self._wrapper.action_to_order(action, battle, strict=False)
        return SinglesEnv.action_to_order(action, battle, strict=False)




