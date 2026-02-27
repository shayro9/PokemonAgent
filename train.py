import argparse
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from poke_env import LocalhostServerConfiguration, AccountConfiguration, MaxBasePowerPlayer, RandomPlayer
from poke_env.environment import SingleAgentWrapper
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback

from env_wrapper import PokemonRLWrapper
from teams.single_teams import ALL_SOLO_TEAMS, STEELIX_TEAM
from teams.team_generators import load_pokemon_pool, single_simple_team_generator, split_pokemon_pool

TEAM_BY_NAME = {name: team for name, team in ALL_SOLO_TEAMS}
DEFAULT_DATA_PATH = "data/gen9randombattle_db.json"


def _wrap_action_masker(env, *, enabled: bool):
    """
    If enabled, attach ActionMasker so MaskablePPO (and others) can use masks.
    If disabled, return env unchanged.
    """
    if not enabled:
        return env

    def mask_fn(e):
        base = e
        if hasattr(base, "unwrapped"):
            base = base.unwrapped

        if hasattr(base, "env") and hasattr(base.env, "unwrapped"):
            base = base.env.unwrapped

        return base.action_masks()

    return ActionMasker(env, mask_fn)


@dataclass
class EvalResult:
    timestep: int
    wins: int
    losses: int
    draws: int

    @property
    def episodes(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.wins / self.episodes


def _parse_pool(raw_pool: str | None, pool_all: bool) -> list[str]:
    if pool_all:
        return [name for name, _ in ALL_SOLO_TEAMS]

    if not raw_pool:
        return []

    names = [name.strip() for name in raw_pool.split(",") if name.strip()]
    unknown = [name for name in names if name not in TEAM_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown pokemon(s) in pool: {', '.join(unknown)}. "
            f"Valid names: {', '.join(sorted(TEAM_BY_NAME))}"
        )
    return names


def _resolve_generated_pools(
        data_path: str,
        train_split: float,
        split_seed: int,
) -> tuple[list[dict], list[dict]]:
    pokemon_pool = load_pokemon_pool(data_path)

    train_pool, eval_pool = split_pokemon_pool(
        pokemon_pool=pokemon_pool,
        train_fraction=train_split,
        seed=split_seed,
    )

    print(
        f"Generated pool split: train={len(train_pool)} samples, eval={len(eval_pool)} samples (seed={split_seed}, train_split={train_split})"
    )
    return train_pool, eval_pool


def _build_train_env(
        agent_team: str,
        battle_format: str,
        opponent_names: list[str],
        opponent_generator,
        rounds_per_opponent: int,
        opponent_pool: list[str] = None,
        agent_team_generator=None,
        battle_team_generator=None,
        use_action_masking: bool = False,
) -> SingleAgentWrapper:
    if not opponent_pool:
        opponent_pool = [TEAM_BY_NAME[name] for name in opponent_names]

    unique_id = int(time.time() * 1000) % 100000

    opponent_policy = RandomPlayer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    agent = PokemonRLWrapper(
        battle_format=battle_format,
        team=agent_team,
        opponent_teams=opponent_pool,
        battle_team_generator=battle_team_generator,
        agent_team_generator=agent_team_generator,
        opponent_team_generator=opponent_generator,
        rounds_per_opponents=rounds_per_opponent,
        server_configuration=LocalhostServerConfiguration,
        account_configuration1=AccountConfiguration(f"Player_{unique_id}", None),
        account_configuration2=AccountConfiguration(f"Opponent_{unique_id}", None),
        strict=True,
    )

    env = SingleAgentWrapper(agent, opponent_policy)
    env = _wrap_action_masker(env, enabled=use_action_masking)
    return env


class BattleMetricsCallback(BaseCallback):
    def __init__(self, env: SingleAgentWrapper, log_freq: int = 100, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.log_freq = log_freq
        self._episode_rewards = []
        self._episode_lengths = []
        self._wins = 0
        self._losses = 0
        self._draws = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self._episode_rewards.append(ep_reward)
                self._episode_lengths.append(ep_length)

                if ep_reward > 0:
                    self._wins += 1
                elif ep_reward < 0:
                    self._losses += 1
                else:
                    self._draws += 1

        if self.n_calls % self.log_freq == 0 and self._episode_rewards:
            total = self._wins + self._losses + self._draws
            win_rate = self._wins / total if total > 0 else 0.0
            wandb.log({
                "win_rate": win_rate,
                "mean_episode_reward": np.mean(self._episode_rewards[-50:]),
                "mean_episode_length": np.mean(self._episode_lengths[-50:]),
            }, step=self.num_timesteps)

        return True


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

    train_env = _build_train_env(
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


def _play_episode(eval_env: SingleAgentWrapper, model: MaskablePPO, max_steps: int) -> tuple[float, bool]:
    obs, _ = eval_env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(eval_env))
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    if not (terminated or truncated):
        truncated = True

    return total_reward, truncated


def _generate_eval_pool(pool_size: int, opponent_generator) -> list[str]:
    if opponent_generator is None:
        raise ValueError("Cannot generate eval pool without an opponent team generator.")

    return [next(opponent_generator) for _ in range(pool_size)]


def evaluate_model(
        model: MaskablePPO,
        timestep: int,
        train_team: str,
        battle_format: str,
        opponent_names: list[str],
        opponent_generator,
        eval_episodes: int,
        max_steps: int,
        agent_team_generator=None,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    # TODO: add from args
    algo = "maskable_ppo"
    use_action_masking = (algo == "maskable_ppo")

    if not opponent_names:
        opponent_pool = _generate_eval_pool(eval_episodes, opponent_generator)
    else:
        if eval_episodes > 0:
            sampled_names = random.sample(opponent_names, k=min(eval_episodes, len(opponent_names)))
        else:
            sampled_names = opponent_names
        opponent_pool = [
            TEAM_BY_NAME[opponent_name]
            for opponent_name in sampled_names
        ]

    eval_env = _build_train_env(
        agent_team=train_team,
        battle_format=battle_format,
        opponent_names=[],
        opponent_generator=None,
        rounds_per_opponent=1,
        opponent_pool=opponent_pool,
        agent_team_generator=agent_team_generator,
        use_action_masking=use_action_masking,
    )

    wins = 0
    losses = 0
    draws = 0

    for _ in range(eval_episodes):

        total_reward, truncated = _play_episode(eval_env, model, max_steps=max_steps)
        if truncated:
            draws += 1
        elif total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1

    results.append(EvalResult(timestep=timestep, wins=wins, losses=losses, draws=draws))

    return results


def print_eval_summary(results: list[EvalResult]) -> None:
    print("\nEvaluation results (deterministic policy):")
    print("Timestep Wins  Losses  Draws  Win rate")
    print("-" * 44)
    total_wins = 0
    total_losses = 0
    total_draws = 0
    for result in results:
        total_wins += result.wins
        total_losses += result.losses
        total_draws += result.draws
        print(
            f"{result.timestep:12} {result.wins:4d} {result.losses:7d} "
            f"{result.draws:6d} {100 * result.win_rate:7.2f}%"
        )

    total_episodes = total_wins + total_losses + total_draws
    overall_win_rate = (100 * total_wins / total_episodes) if total_episodes else 0.0
    print("-" * 44)
    print(
        f"Overall        {total_wins:4d} {total_losses:7d} {total_draws:6d} {overall_win_rate:7.2f}%"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a MaskablePPO Pokémon bot against a pool and evaluate win rate over an opponent set."
    )

    # Core
    parser.add_argument("--format", type=str, default="gen9customgame")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducible training/evaluation.")
    parser.add_argument("--model-path", default="data/1v1")

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
        default=100,
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


def _resolve_seed(explicit: Optional[int], fallback: int) -> int:
    return fallback if explicit is None else explicit


@dataclass(frozen=True)
class OpponentsResolved:
    train_names: list[str]
    eval_names: list[str]
    train_gen: Optional[Iterable]
    eval_gen: Optional[Iterable]
    train_agent_gen: Optional[Iterable]
    eval_agent_gen: Optional[Iterable]


def _resolve_opponents(args) -> OpponentsResolved:
    """
    Single source of truth for:
      - train opponent names or generator
      - eval opponent names or generator
    """

    # ---- TRAIN opponents ----
    train_names: list[str] = []
    train_gen = None
    train_agent_gen = None
    eval_agent_gen = None

    if args.random_generated:
        # build generators (possibly split)
        train_seed = _resolve_seed(args.train_generator_seed, args.seed)
        eval_seed = _resolve_seed(args.eval_generator_seed, args.seed)

        agent_data_path = args.agent_data_path or DEFAULT_DATA_PATH
        opponent_data_path = args.opponent_data_path or DEFAULT_DATA_PATH

        if args.split_generated_pool:
            if agent_data_path == opponent_data_path:
                train_pool, eval_pool = _resolve_generated_pools(
                    data_path=agent_data_path,
                    train_split=args.train_split,
                    split_seed=args.split_seed,
                )
                train_agent_pool = train_pool
                eval_agent_pool = eval_pool
                train_opponent_pool = train_pool
                eval_opponent_pool = eval_pool
            else:
                train_agent_pool, eval_agent_pool = _resolve_generated_pools(
                    data_path=agent_data_path,
                    train_split=args.train_split,
                    split_seed=args.split_seed,
                )
                train_opponent_pool, eval_opponent_pool = _resolve_generated_pools(
                    data_path=opponent_data_path,
                    train_split=args.train_split,
                    split_seed=args.split_seed,
                )
        else:
            shared_agent_pool = load_pokemon_pool(agent_data_path)
            shared_opponent_pool = (
                shared_agent_pool
                if agent_data_path == opponent_data_path
                else load_pokemon_pool(opponent_data_path)
            )
            train_agent_pool = eval_agent_pool = shared_agent_pool
            train_opponent_pool = eval_opponent_pool = shared_opponent_pool

        train_gen = single_simple_team_generator(pokemon_pool=train_opponent_pool, seed=train_seed)
        eval_gen = single_simple_team_generator(pokemon_pool=eval_opponent_pool, seed=eval_seed)

        if args.train_team is None:
            train_agent_gen = train_gen
            eval_agent_gen = eval_gen
    else:
        # name-based pools
        train_names = _parse_pool(args.pool, args.pool_all)
        eval_gen = None

    # ---- EVAL opponents ----
    if args.eval_pool is not None or args.eval_pool_all:
        eval_names = _parse_pool(args.eval_pool, args.eval_pool_all)
    else:
        # Otherwise:
        # - if training used names => eval uses same names
        # - if training used generated => eval will use generator (handled above)
        eval_names = train_names

    # if not generated, eval_gen must be None
    if not args.random_generated:
        eval_gen = None

    return OpponentsResolved(
        train_names=train_names,
        eval_names=eval_names,
        train_gen=train_gen,
        eval_gen=eval_gen,
        train_agent_gen=train_agent_gen,
        eval_agent_gen=eval_agent_gen,
    )


def main():
    args = build_arg_parser().parse_args()

    battle_format = args.format
    train_team = TEAM_BY_NAME.get(args.train_team) if args.train_team else None

    opp = _resolve_opponents(args)

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
