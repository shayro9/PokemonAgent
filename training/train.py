import random
import time

from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback

from policy.policy import AttentionPointerPolicy
from .parse import build_arg_parser
from config.config import *
from training.battle_metrics_log import *

from env.env_builder import build_env
from .evaluation import evaluate_model, print_eval_summary, build_fixed_eval_pool

LR = 3e-4
LR_DECAY = 0.9
N_STEPS = 4096
BATCH_SIZE = 256
GAMMA = 0.99
ENT_COEF = 0.03
LOG_FREQ = 500


def train_model(
        model_path: str,
        battle_format: str,
        opponent_generator,
        timesteps: int,
        rounds_per_opponent: int,
        eval_every_timesteps: int,
        eval_kwargs: dict | None = None,
        agent_team_generator=None,
        battle_team_generator=None,
        seed: int = 42,
) -> MaskablePPO:
    """Train a MaskablePPO agent and optionally run periodic evaluation.

    :param model_path: Output path for the saved model.
    :param battle_format: Showdown format used for battles.
    :param opponent_generator: Optional generator of opponent teams.
    :param timesteps: Total training timesteps.
    :param rounds_per_opponent: Battles played before rotating opponents.
    :param eval_every_timesteps: Interval for periodic evaluation.
    :param eval_kwargs: Optional keyword arguments for evaluation calls.
    :param agent_team_generator: Optional generator for agent team rotation.
    :param battle_team_generator: Optional generator yielding both battle teams.
    :param seed: Random seed.
    :returns: The trained ``MaskablePPO`` model."""
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)

    run = wandb.init(
        project="pokemon-rl",
        name=f"{battle_format}__{int(time.time())}",
        config={
            "timesteps": timesteps,
            "rounds_per_opponent": rounds_per_opponent,
            "battle_format": battle_format,
            "learning_rate": LR,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "ent_coef": ENT_COEF,
        },
        sync_tensorboard=False,
        save_code=True,
    )

    # TODO: add from args
    algo = "maskable_ppo"
    use_action_masking = (algo == "maskable_ppo")

    train_env = build_env(
        battle_format,
        opponent_generator,
        rounds_per_opponent,
        agent_team_generator=agent_team_generator,
        battle_team_generator=battle_team_generator,
        use_action_masking=use_action_masking,
    )

    policy_kwargs = dict(
        context_hidden=256,
        move_hidden=128,
        trunk_hidden=256,
        n_attention_heads=4,
    )

    model = MaskablePPO(
        AttentionPointerPolicy,
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=lambda progress: LR * (0.1 + LR_DECAY * progress),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=0.95,
        ent_coef=ENT_COEF,
        clip_range=0.2,
        max_grad_norm=10.0,
        verbose=0,
        seed=seed,
        normalize_advantage=True,
        clip_range_vf=0.2,
        n_epochs=5,
    )

    print(f"rounds_per_opponent={rounds_per_opponent}")
    if eval_every_timesteps > 0 and eval_kwargs:
        trained_steps = 0
        eval_results = []
        metrics_cb = BattleMetricsCallback(env=train_env, log_freq=LOG_FREQ)
        while trained_steps < timesteps:
            step_chunk = min(eval_every_timesteps, timesteps - trained_steps)
            model.learn(
                total_timesteps=step_chunk,
                reset_num_timesteps=False,
                callback=CallbackList([
                    WandbCallback(gradient_save_freq=100, verbose=0),
                    metrics_cb,
                ]))
            trained_steps += step_chunk

            eval_res = evaluate_model(model=model, timestep=trained_steps, **eval_kwargs)
            eval_results.extend(eval_res)
            print(f"\n[Eval @ {trained_steps} training timesteps]")
            print_eval_summary(eval_results)
    else:
        model.learn(
            total_timesteps=timesteps,
            callback=CallbackList([
                WandbCallback(gradient_save_freq=100, verbose=0),
                BattleMetricsCallback(env=train_env, log_freq=LOG_FREQ),
            ])
        )

    model.save(model_path)
    run.finish()
    print(f"Training complete! Model saved as {model_path}.zip")

    return model


def main():
    """Parse arguments, run training, and print evaluation summaries.

    :returns: ``None``."""
    args = build_arg_parser().parse_args()

    battle_format = args.format
    opp = resolve_opponents(args)

    eval_kwargs = {
        "battle_format": battle_format,
        "opponent_generator": opp.eval_gen,
        "eval_episodes": args.eval_episodes,
        "max_steps": args.eval_max_steps,
        "agent_team_generator": opp.eval_agent_gen,
        "battle_team_generator": opp.eval_battle_team_generator,
    }

    model = train_model(
        model_path=args.model_path,
        battle_format=battle_format,
        opponent_generator=opp.train_gen,
        timesteps=args.timesteps,
        rounds_per_opponent=args.rounds_per_opponent,
        eval_every_timesteps=args.eval_every_timesteps,
        eval_kwargs=None if args.skip_eval else eval_kwargs,
        agent_team_generator=opp.train_agent_gen,
        battle_team_generator=opp.train_battle_team_generator,
        seed=args.seed,
    )

    if args.skip_eval:
        return

    eval_results = evaluate_model(model=model, timestep=args.timesteps, **eval_kwargs)
    print_eval_summary(eval_results)


if __name__ == "__main__":
    main()
