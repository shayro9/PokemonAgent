import argparse
from pathlib import Path

import numpy as np
import gymnasium
from gymnasium import spaces

from env.battle_config import BattleConfig


# ---------------------------------------------------------------------------
# Dummy environment
# ---------------------------------------------------------------------------

class _DummyGen1Env(gymnasium.Env):
    """Minimal gymnasium environment with the correct obs/action spaces for Gen 1.

    Only used to satisfy MaskablePPO's constructor — never stepped.
    """

    def __init__(self, config: BattleConfig) -> None:
        super().__init__()
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(config.obs_dim,),
                dtype=np.float32,
            ),
            'action_mask': spaces.Box(
                low=0,
                high=1,
                shape=(config.action_space_size,),
                dtype=np.int64,
            ),
        })
        self.action_space = spaces.Discrete(config.action_space_size)
        self._n_actions = config.action_space_size

    def reset(self, *, seed=None, options=None):
        obs = {
            'observation': self.observation_space['observation'].sample(),
            'action_mask': np.ones(self._n_actions, dtype=np.int64),
        }
        return obs, {}

    def step(self, action):
        obs = {
            'observation': self.observation_space['observation'].sample(),
            'action_mask': np.ones(self._n_actions, dtype=np.int64),
        }
        return obs, 0.0, False, False, {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Supervised warmup (BC + value pretraining) for Gen-1 6v6 policy.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        required=True,
        help='Path to the JSONL file produced by convert_metamon_json.py.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of supervised training epochs (default: 10).',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Mini-batch size (default: 256).',
    )
    parser.add_argument(
        '--lambda-value',
        type=float,
        default=1.0,
        help='Weight for the value MSE loss term (default: 1.0).',
    )
    parser.add_argument(
        '--lambda-bc',
        type=float,
        default=1.0,
        help='Weight for the behavioral cloning CE loss term (default: 1.0).',
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.0,
        metavar='FRAC',
        help=(
            'Fraction of episodes to hold out as a validation set for per-epoch '
            'BC accuracy reporting (e.g. 0.1 = 10%%).  Default: 0.0 (disabled).'
        ),
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help=(
            "Torch device: 'cpu', 'cuda', 'cuda:0', or 'auto' (default). "
            "'auto' uses CUDA if available, otherwise CPU."
        ),
    )
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path('models/supervised_gen1_6v6.zip'),
        help="Path to save the pretrained model (default: models/supervised_gen1_6v6.zip).",
    )
    return parser