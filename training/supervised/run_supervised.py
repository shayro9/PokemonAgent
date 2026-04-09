"""
training/supervised/run_supervised.py
======================================
CLI entry point for supervised warmup (behavioral cloning + value pretraining)
of the Gen-1 6v6 AttentionPointerPolicy from metamon replay data.
"""

from __future__ import annotations

import sys
import torch

from sb3_contrib import MaskablePPO

from env.battle_config import BattleConfig
from policy.policy import AttentionPointerPolicy
from training.config import POLICY_KWARGS
from training.supervised import pretrain_from_human_data, SupervisedWarmupConfig
from training.supervised.utils import build_arg_parser, _DummyGen1Env


def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    if not args.data_path.exists():
        print(f'ERROR: Data file not found: {args.data_path}')
        print('Run data_generation/convert_metamon_json.py first.')
        sys.exit(1)

    if not (0.0 <= args.val_split < 1.0):
        print(f'ERROR: --val-split must be in [0.0, 1.0), got {args.val_split}')
        sys.exit(1)

    config = BattleConfig.gen1()

    # Resolve 'auto' to the actual device so we can log and pass it explicitly.
    if args.device == 'auto':
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        resolved_device = args.device
    device = torch.device(resolved_device)

    print(f'[run_supervised] Gen {config.gen} | obs_dim={config.obs_dim} | '
          f'action_space={config.action_space_size} | device={device}'
          + (f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else ''))

    env   = _DummyGen1Env(config)
    # Use the same architecture kwargs as the RL training pipeline so that a
    # supervised checkpoint can be loaded directly for RL fine-tuning.
    model = MaskablePPO(
        AttentionPointerPolicy,
        env=env,
        device=resolved_device,
        policy_kwargs={**POLICY_KWARGS, 'battle_config': config},
        verbose=0,
    )

    warmup_cfg = SupervisedWarmupConfig(
        data_path=str(args.data_path),
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_value=args.lambda_value,
        lambda_bc=args.lambda_bc,
        val_split=args.val_split,
    )
    model = pretrain_from_human_data(model, warmup_cfg)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_path))
    print(f'[run_supervised] Model saved to {args.output_path}')


if __name__ == '__main__':
    main()

