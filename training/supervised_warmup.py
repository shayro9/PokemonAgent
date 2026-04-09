"""
training/supervised_warmup.py
==============================
Backward-compatibility shim.

The supervised pretraining logic now lives in the ``training.supervised``
package.  This module re-exports its public API so that existing imports
continue to work without change::

    from training.supervised_warmup import SupervisedWarmupConfig, pretrain_from_human_data
"""

from training.supervised import SupervisedWarmupConfig, pretrain_from_human_data  # noqa: F401

__all__ = ["SupervisedWarmupConfig", "pretrain_from_human_data"]
