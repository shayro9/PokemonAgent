import argparse

from config.config import TEAM_BY_NAME


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

    # Training
    parser.add_argument("--train-team", default=None, choices=sorted(TEAM_BY_NAME))
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

    # Opponent selection: TRAIN
    train_src = parser.add_mutually_exclusive_group()
    train_src.add_argument("--pool", default=None, help="Comma-separated opponent pool for training.")
    train_src.add_argument("--pool-all", action="store_true", default=False, help="Use all prebuilt solo teams.")
    train_src.add_argument("--random-generated", action="store_true", default=False, help="Use generated opponents.")

    # Opponent selection: EVAL (optional override)
    eval_src = parser.add_mutually_exclusive_group()
    eval_src.add_argument("--eval-pool", default=None, help="Comma-separated opponent pool for evaluation.")
    eval_src.add_argument("--eval-pool-all", action="store_true", default=False, help="Use all prebuilt solo teams.")

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

    return parser
