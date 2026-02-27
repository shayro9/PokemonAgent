import time

from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback

from .parse import build_arg_parser
from config.config import *
from .logs import *

from env.env_builder import build_env
from .evaluation import evaluate_model, print_eval_summary


def train_model(
        model_path: str,
        battle_format: str,
        train_team: str,
        opponent_names: list[str],
        opponent_generator,
        timesteps: int,
        rounds_per_opponent: int,
        eval_every_timesteps: int,
        eval_kwargs: dict | None = None,
        agent_team_generator=None,
        battle_team_generator=None,
        seed: int = 42,
) -> MaskablePPO:
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
            "learning_rate": 3e-4,
            "n_steps": 4096,
            "batch_size": 32,
            "gamma": 0.999,
            "ent_coef": 0.1,
        },
        sync_tensorboard=False,
        save_code=True,
    )

    # TODO: add from args
    algo = "maskable_ppo"
    use_action_masking = (algo == "maskable_ppo")

    train_env = build_env(
        train_team,
        battle_format,
        opponent_names,
        opponent_generator,
        rounds_per_opponent,
        agent_team_generator=agent_team_generator,
        battle_team_generator=battle_team_generator,
        use_action_masking=use_action_masking,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=32,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.1,
        clip_range=0.2,
        max_grad_norm=10.0,
        verbose=0,
        seed=seed,
    )

    team_label = (
        next(name for name, team in TEAM_BY_NAME.items() if team == train_team)
        if train_team is not None
        else "generated"
    )
    print(
        f"Starting training: team={team_label} | pool={opponent_names} | "
        f"rounds_per_opponent={rounds_per_opponent}"
    )
    if eval_every_timesteps > 0 and eval_kwargs:
        trained_steps = 0
        eval_results = []
        while trained_steps < timesteps:
            step_chunk = min(eval_every_timesteps, timesteps - trained_steps)
            model.learn(
                total_timesteps=step_chunk,
                reset_num_timesteps=False,
                callback=CallbackList([
                    WandbCallback(gradient_save_freq=100, verbose=0),
                    BattleMetricsCallback(env=train_env, log_freq=100),
                ]))
            trained_steps += step_chunk

            eval_results.extend(evaluate_model(model=model, timestep=trained_steps, **eval_kwargs))
            print(f"\n[Eval @ {trained_steps} training timesteps]")
            print_eval_summary(eval_results)
    else:
        model.learn(
            total_timesteps=timesteps,
            callback=CallbackList([
                WandbCallback(gradient_save_freq=100, verbose=0),
                BattleMetricsCallback(env=train_env, log_freq=100),
            ])
        )

    model.save(model_path)
    run.finish()
    print(f"Training complete! Model saved as {model_path}.zip")

    return model


def main():
    args = build_arg_parser().parse_args()

    battle_format = args.format
    train_team = TEAM_BY_NAME.get(args.train_team) if args.train_team else None

    opp = resolve_opponents(args)

    eval_kwargs = {
        "battle_format": battle_format,
        "train_team": train_team,
        "opponent_names": opp.eval_names,
        "opponent_generator": opp.eval_gen,
        "eval_episodes": args.eval_episodes,
        "max_steps": args.eval_max_steps,
        "agent_team_generator": opp.eval_agent_gen,
    }

    model = train_model(
        model_path=args.model_path,
        battle_format=battle_format,
        train_team=train_team,
        opponent_names=opp.train_names,
        opponent_generator=opp.train_gen,
        timesteps=args.timesteps,
        rounds_per_opponent=args.rounds_per_opponent,
        eval_every_timesteps=args.eval_every_timesteps,
        eval_kwargs=None if args.skip_eval else eval_kwargs,
        agent_team_generator=opp.train_agent_gen,
        seed=args.seed,
    )

    if args.skip_eval:
        return

    eval_results = evaluate_model(model=model, timestep=args.timesteps, **eval_kwargs)
    print_eval_summary(eval_results)


if __name__ == "__main__":
    main()
