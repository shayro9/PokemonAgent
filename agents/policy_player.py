"""A poke_env Player that picks moves using a saved MaskablePPO policy."""
from poke_env.player import Player
from sb3_contrib import MaskablePPO

from env.battle_config import BattleConfig
from env.action_mask_gen_1 import ActionMaskGen1
from env.singles_env_wrapper import PokemonRLWrapper


class PolicyPlayer(Player):
    """A Showdown player driven by a trained MaskablePPO policy.

    :param model_path: Path to the saved ``.zip`` model file.
    :param wrapper: A :class:`~env.singles_env_wrapper.PokemonRLWrapper`
        used as a stateless helper for embedding and action decoding.
    :param battle_config: Generation config used to build observations.
        Defaults to Gen 1.
    :param deterministic: Use deterministic action selection (default True).
    :param verbose: Print the full BattleState description on every turn.
    """

    def __init__(
        self,
        model_path: str,
        wrapper: PokemonRLWrapper,
        *,
        battle_config: BattleConfig | None = None,
        deterministic: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model: MaskablePPO = MaskablePPO.load(model_path)
        self._wrapper = wrapper
        self._battle_config = battle_config if battle_config is not None else BattleConfig.gen1()
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

        observation = {
                "observation": self._battle_config.battle_state_cls(battle).to_array(),
                "action_mask": self._wrapper.get_action_mask(battle),
            }
        action, _ = self.model.predict(
            observation,
            deterministic=self._deterministic,
            action_masks=observation["action_mask"],
        )

        order = self._wrapper.action_to_order(action, battle, strict=False)

        if self._verbose:
            print(f"\n→ Chosen action: {action}  →  {order}")

        return order




