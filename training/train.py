import random
import time
import numpy as np
import wandb

from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import set_random_seed

from policy.policy import AttentionPointerPolicy
from config.config import resolve_opponents
from env.env_builder import build_env, build_vec_env
from env.battle_config import BattleConfig
from training.parse import build_arg_parser
from training.device_config import DeviceConfig
from training.battle_metrics_log import BattleMetricsCallback
from training.config import LR, N_STEPS, BATCH_SIZE, GAMMA, ENT_COEF, LR_DECAY, LOG_FREQ, POLICY_KWARGS
from training.supervised import SupervisedWarmupConfig, pretrain_from_human_data
from training.evaluation import evaluate_model, print_eval_summary
from training.validation import validate_args
from wandb.integration.sb3 import WandbCallback


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
        device: str = "auto",
        n_envs: int = 1,
        warmup_kwargs: dict | None = None,
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
    :param device: "auto", "cuda", or "cpu"
    :param n_envs: Number of parallel environment workers. Values > 1 use
        ``SubprocVecEnv`` for true parallelism. Each worker runs its own
        asyncio event loop and Pokémon Showdown connections.
    :param warmup_kwargs: Optional dict of warmup/load settings.
    :returns: The trained ``MaskablePPO`` model."""
    warmup_config     = (warmup_kwargs or {}).get('warmup_config')
    load_model_path   = (warmup_kwargs or {}).get('load_model_path')
    eval_after_warmup = (warmup_kwargs or {}).get('eval_after_warmup', False)

    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    
    # Setup device
    device_config = DeviceConfig(device=device)
    device_config.print_info()

    # TODO: add from args
    algo = "maskable_ppo"

    if n_envs > 1:
        train_env = build_vec_env(
            n_envs=n_envs,
            battle_format=battle_format,
            opponent_generator=opponent_generator,
            rounds_per_opponent=rounds_per_opponent,
            agent_team_generator=agent_team_generator,
            battle_team_generator=battle_team_generator,
        )
        print(f"Using {n_envs} parallel environment workers (SubprocVecEnv).")
    else:
        train_env = build_env(
            battle_format,
            opponent_generator,
            rounds_per_opponent,
            agent_team_generator=agent_team_generator,
            battle_team_generator=battle_team_generator,
        )

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
            "device": str(device_config),
        },
        save_code=True,
    )

    policy_kwargs = {**POLICY_KWARGS, 'battle_config': BattleConfig.gen1()}

    if load_model_path is not None:
        print(f'[train] Loading model from {load_model_path!r} ...')
        model = MaskablePPO.load(load_model_path, env=train_env, device=str(device_config))
    else:
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
            device=str(device_config),
        )

        if warmup_config is not None:
            print(f'[train] Running supervised warmup from {warmup_config.data_path!r} ...')
            pretrain_from_human_data(model, warmup_config)

    if eval_after_warmup:
        if eval_kwargs:
            print('[train] Evaluating post-warmup baseline ...')
            baseline = evaluate_model(model=model, timestep=0, **eval_kwargs)
            print('\n[Eval @ warmup baseline]')
            print_eval_summary(baseline)
        else:
            print('[train] Warning: eval_after_warmup=True but eval is disabled — skipping baseline eval.')

    print(f"rounds_per_opponent={rounds_per_opponent}")
    if timesteps > 0:
        if eval_every_timesteps > 0 and eval_kwargs:
            trained_steps = 0
            eval_results = []
            metrics_cb = BattleMetricsCallback(env=train_env, log_freq=LOG_FREQ)
            wandb_cb = WandbCallback(verbose=0)
            while trained_steps < timesteps:
                step_chunk = min(eval_every_timesteps, timesteps - trained_steps)
                model.learn(
                    total_timesteps=step_chunk,
                    reset_num_timesteps=False,
                    callback=[metrics_cb, wandb_cb])
                trained_steps += step_chunk

                eval_res = evaluate_model(model=model, timestep=trained_steps, **eval_kwargs)
                eval_results.extend(eval_res)
                print(f"\n[Eval @ {trained_steps} training timesteps]")
                print_eval_summary(eval_results)
        else:
            model.learn(
                total_timesteps=timesteps,
                callback=[BattleMetricsCallback(env=train_env, log_freq=LOG_FREQ), WandbCallback(verbose=0)],
            )
    else:
        print('[train] timesteps=0 — skipping RL training.')

    model.save(model_path)
    run.finish()
    print(f"Training complete! Model saved as {model_path}.zip")

    return model


def main():
    """Parse arguments, run training, and print evaluation summaries.

    :returns: ``None``."""
    args = build_arg_parser().parse_args()

    validate_args(args)
    opp = resolve_opponents(args)

    battle_format = args.format
    eval_kwargs = {
        "battle_format": battle_format,
        "opponent_generator": opp.eval_gen,
        "eval_episodes": args.eval_episodes,
        "max_steps": args.eval_max_steps,
        "agent_team_generator": opp.eval_agent_gen,
        "battle_team_generator": opp.eval_battle_team_generator,
    }

    warmup_kwargs = {
            'warmup_config': SupervisedWarmupConfig(
                data_path=args.warmup_data_path,
                n_epochs=args.warmup_epochs,
                lr=args.warmup_lr,
                batch_size=args.warmup_batch_size,
            ) if args.warmup_data_path else None,
            'load_model_path': args.load_model_path,
            'eval_after_warmup': args.eval_after_warmup,
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
        device=args.device,
        n_envs=args.n_envs,
        warmup_kwargs=warmup_kwargs,
    )

    if args.skip_eval:
        return

    if args.timesteps == 0 and args.eval_after_warmup:
        return

    eval_results = evaluate_model(model=model, timestep=args.timesteps, **eval_kwargs)
    print_eval_summary(eval_results)


if __name__ == "__main__":
    main()
