import random
import time
import numpy as np
import wandb

from curriculum.runtime import Curriculum
from curriculum.yaml_loader import load_curriculum_from_yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import set_random_seed

from policy.policy import AttentionPointerPolicy
from config.config import resolve_opponents
from env.env_builder import build_env, build_vec_env
from training.parse import build_arg_parser
from training.device_config import DeviceConfig
from wandb.integration.sb3 import WandbCallback
from training.battle_metrics_log import BattleMetricsCallback
from training.config import LR, N_STEPS, BATCH_SIZE, GAMMA, ENT_COEF, LR_DECAY, LOG_FREQ
from training.curriculum_callback import CurriculumCallback
from training.evaluation import evaluate_model, print_eval_summary


def _print_curriculum_summary(curriculum: Curriculum) -> None:
    """Print a one-time summary of the loaded curriculum."""
    stages = getattr(curriculum, "stages", None)
    if stages:
        print("Curriculum stages:")
        for stage in stages:
            end_label = "end" if stage.end_timestep is None else stage.end_timestep
            print(
                f"  - {stage.name}: [{stage.start_timestep}, {end_label}) "
                f"-> {stage.opponent_player.identifier}"
            )
        return

    initial_stage = curriculum.stage_for_timesteps(0)
    end_label = "end" if initial_stage.end_timestep is None else initial_stage.end_timestep
    print(
        "Curriculum loaded: "
        f"{initial_stage.name} [{initial_stage.start_timestep}, {end_label}) "
        f"-> {initial_stage.opponent_player.identifier}"
    )


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
        curriculum: Curriculum | None = None,
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
    :param curriculum: Optional staged opponent-player curriculum.
    :returns: The trained ``MaskablePPO`` model."""
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    
    # Setup device
    device_config = DeviceConfig(device=device)
    device_config.print_info()

    # TODO: add from args
    algo = "maskable_ppo"
    initial_opponent_player_spec = (
        curriculum.stage_for_timesteps(0).opponent_player
        if curriculum is not None else None
    )

    if n_envs > 1:
        train_env = build_vec_env(
            n_envs=n_envs,
            battle_format=battle_format,
            opponent_generator=opponent_generator,
            rounds_per_opponent=rounds_per_opponent,
            opponent_player_spec=initial_opponent_player_spec,
            agent_team_generator=agent_team_generator,
            battle_team_generator=battle_team_generator,
        )
        print(f"Using {n_envs} parallel environment workers (SubprocVecEnv).")
    else:
        train_env = build_env(
            battle_format,
            opponent_generator,
            rounds_per_opponent,
            opponent_player_spec=initial_opponent_player_spec,
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
        device=str(device_config),
    )

    print(f"rounds_per_opponent={rounds_per_opponent}")
    callbacks = [
        BattleMetricsCallback(env=train_env, log_freq=LOG_FREQ),
        WandbCallback(verbose=0),
    ]
    if curriculum is not None:
        _print_curriculum_summary(curriculum)
        callbacks.insert(0, CurriculumCallback(curriculum=curriculum))

    if eval_every_timesteps > 0 and eval_kwargs:
        trained_steps = 0
        eval_results = []
        while trained_steps < timesteps:
            step_chunk = min(eval_every_timesteps, timesteps - trained_steps)
            model.learn(
                total_timesteps=step_chunk,
                reset_num_timesteps=False,
                callback=callbacks)
            trained_steps += step_chunk

            eval_res = evaluate_model(model=model, timestep=trained_steps, **eval_kwargs)
            eval_results.extend(eval_res)
            print(f"\n[Eval @ {trained_steps} training timesteps]")
            print_eval_summary(eval_results)
    else:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
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
    curriculum = (
        load_curriculum_from_yaml(args.curriculum_config)
        if args.curriculum_config else None
    )

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
        device=args.device,
        n_envs=args.n_envs,
        curriculum=curriculum,
    )

    if args.skip_eval:
        return

    eval_results = evaluate_model(model=model, timestep=args.timesteps, **eval_kwargs)
    print_eval_summary(eval_results)


if __name__ == "__main__":
    main()
