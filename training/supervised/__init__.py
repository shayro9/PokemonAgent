"""
training/supervised
====================
Standalone supervised (behavioral cloning + value pretraining) module.

Can be run as a standalone CLI::

    python -m training.supervised.run_supervised \\
        --data-path data/supervised/gen1ou.npz \\
        --epochs 10 \\
        --output-path models/supervised_gen1_6v6.zip

Or imported by the RL training pipeline::

    from training.supervised import SupervisedWarmupConfig, pretrain_from_human_data

    warmup_cfg = SupervisedWarmupConfig(data_path="data/supervised/gen1ou.npz", n_epochs=10)
    train_model(..., warmup_config=warmup_cfg)
"""

from .supervised_config import SupervisedWarmupConfig
from .warmup import pretrain_from_human_data

__all__ = ["SupervisedWarmupConfig", "pretrain_from_human_data"]
