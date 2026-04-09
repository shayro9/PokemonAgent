import argparse

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for training and evaluation.
    
    :returns: Configured ``argparse.ArgumentParser`` instance."""
    parser = argparse.ArgumentParser(
        description="Train a MaskablePPO Pokémon bot against a pool and evaluate win rate over an opponent set."
    )

    # Core
    parser.add_argument("--format", type=str, default="gen9customgame")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducible training/evaluation.")
    parser.add_argument("--model-path", default="models/1v1_gen1_500k_steps")
    parser.add_argument(
        "--load-model-path",
        type=str,
        default=None,
        help="Path to an existing saved model (.zip) to load instead of constructing fresh. "
             "Use with --timesteps 0 to evaluate a warmed-up model with no RL training.",
    )

    # Training
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--rounds-per-opponent", type=int, default=2_000)
    parser.add_argument("--eval-every-timesteps", type=int, default=0)

    # Evaluation
    parser.add_argument("--skip-eval", action="store_true", help="Only train and save model, skip evaluation.")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="How many episodes to evaluate against (one battle per opponent).",
    )
    parser.add_argument("--eval-max-steps", type=int, default=500)

    # Generated dataset split (only meaningful if --random-generated)
    parser.add_argument(
        "--agent-data-path",
        type=str,
        default=None,
        help="Path to generated dataset used for the agent team generator. Defaults to data/gen9randombattle_db.json.",
    )
    parser.add_argument(
        "--opponent-data-path",
        type=str,
        default=None,
        help="Path to generated dataset used for the opponent team generator. Defaults to data/gen9randombattle_db.json.",
    )
    parser.add_argument(
        "--matchup-data-path",
        type=str,
        default=None,
        help="Path to a pre-paired matchup database JSON.",
    )
    parser.add_argument(
        "--split-generated-pool",
        action="store_true",
        default=False,
        help="Split generated dataset into disjoint train/eval pools.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of generated dataset used for training when split is enabled.",
    )
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--train-generator-seed", type=int, default=None)
    parser.add_argument("--eval-generator-seed", type=int, default=None)
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use: 'auto' (GPU if available, else CPU), 'cuda' (GPU only), 'cpu'"
    )

    # Parallelism
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environment workers for training. Values > 1 use "
             "SubprocVecEnv for true parallel rollout collection.",
    )

    # Supervised warmup (all optional — omitting --warmup-data-path disables warmup entirely)
    parser.add_argument(
        "--warmup-data-path",
        type=str,
        default=None,
        help="Path to JSONL file for supervised warmup pretraining (produced by "
             "convert_metamon_json.py). Omit to skip warmup.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Number of supervised warmup training epochs (default: 10).",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=3e-4,
        help="Learning rate for supervised warmup (default: 3e-4).",
    )
    parser.add_argument(
        "--warmup-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for supervised warmup (default: 256).",
    )
    parser.add_argument(
        "--eval-after-warmup",
        action="store_true",
        default=False,
        help="Run one evaluation pass immediately after warmup and before RL training. "
             "Produces a baseline win-rate for the warmed-up policy. "
             "Requires --eval-episodes > 0.",
    )

    return parser
