import argparse

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for training and evaluation.
    
    :returns: Configured ``argparse.ArgumentParser`` instance."""
    parser = argparse.ArgumentParser(
        description="Train a MaskablePPO Pokémon bot against local Pokémon Showdown opponents and evaluate win rate."
    )

    # Core
    parser.add_argument("--format", type=str, default="gen1ou")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducible training/evaluation.")
    parser.add_argument("--model-path", default="models/1v1_gen1_500k_steps")

    # Training
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--rounds-per-opponent", type=int, default=2_000)
    parser.add_argument("--eval-every-timesteps", type=int, default=0)
    parser.add_argument(
        "--curriculum-config",
        type=str,
        default=None,
        help="Optional YAML curriculum that switches opponent Player classes by training timestep.",
    )

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
        help="Path to the generated team dataset used for the agent side. Defaults to "
             "data/meta_pool_teams_6_gen1ou_db.json.",
    )
    parser.add_argument(
        "--opponent-data-path",
        type=str,
        default=None,
        help="Path to the generated team dataset used for the opponent side. Defaults to "
             "data/meta_pool_teams_6_gen1ou_db.json.",
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

    return parser
