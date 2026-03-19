"""A poke_env Player that picks moves using a saved MaskablePPO policy."""
from poke_env.player import Player
from sb3_contrib import MaskablePPO

from env.action_masking import get_valid_action_mask
from env.singles_env_wrapper import PokemonRLWrapper
from env.states.gen1.battle_state_gen_1 import BattleStateGen1


class PolicyPlayer(Player):
    """A Showdown player driven by a trained MaskablePPO policy.

    :param model_path: Path to the saved ``.zip`` model file.
    :param wrapper: A :class:`~env.singles_env_wrapper.PokemonRLWrapper`
        used as a stateless helper for embedding and action decoding.
    :param deterministic: Use deterministic action selection (default True).
    :param verbose: Print the full BattleState description on every turn.
    """

    def __init__(
        self,
        model_path: str,
        wrapper: PokemonRLWrapper,
        *,
        deterministic: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
            print(BattleStateGen1(battle).describe())

        obs = self._wrapper.embed_battle(battle)
        mask = get_valid_action_mask(battle)

        action, _ = self.model.predict(
            obs,
            deterministic=self._deterministic,
            action_masks=mask,
        )

        order = self._wrapper.action_to_order(action, battle, strict=False)

        if self._verbose:
            print(f"\n→ Chosen action: {action}  →  {order}")

        return order




