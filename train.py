import argparse
from dataclasses import dataclass

from stable_baselines3 import DQN
from poke_env import LocalhostServerConfiguration, RandomPlayer, MaxBasePowerPlayer
from poke_env.environment import SingleAgentWrapper

from env_wrapper import PokemonRLWrapper
from teams.single_teams import ALL_SOLO_TEAMS, STEELIX_TEAM
from teams.team_generators import load_pokemon_pool, single_simple_team_generator, split_pokemon_pool

TEAM_BY_NAME = {name: team for name, team in ALL_SOLO_TEAMS}


@dataclass
class EvalResult:
    opponent: str
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
) -> SingleAgentWrapper:
    if not opponent_pool:
        opponent_pool = [TEAM_BY_NAME[name] for name in opponent_names]

    opponent_policy = MaxBasePowerPlayer(
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
        strict=False,
    )

    return SingleAgentWrapper(agent, opponent_policy)


def train_model(
        model_path: str,
        battle_format: str,
        train_team: str,
        opponent_names: list[str],
        opponent_generator,
        timesteps: int,
        rounds_per_opponent: int,
        agent_team_generator=None,
        battle_team_generator=None,
) -> DQN:
    train_env = _build_train_env(
        train_team,
        battle_format,
        opponent_names,
        opponent_generator,
        rounds_per_opponent,
        agent_team_generator=agent_team_generator,
        battle_team_generator=battle_team_generator,
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=64,
        tau=1.0,
        gamma=0.999,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.7,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
        verbose=0,
    )

    print(
        f"Starting training: team={next(name for name, team in TEAM_BY_NAME.items() if team == train_team)} | pool={opponent_names} | "
        f"rounds_per_opponent={rounds_per_opponent}"
    )
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    print(f"Training complete! Model saved as {model_path}.zip")

    return model


def _play_episode(eval_env: SingleAgentWrapper, model: DQN, max_steps: int) -> tuple[float, bool]:
    obs, _ = eval_env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
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

    pool = []
    for _ in range(pool_size):
        pool.append(next(opponent_generator))

    return pool


def evaluate_model(
        model: DQN,
        train_team: str,
        battle_format: str,
        opponent_names: list[str],
        opponent_generator,
        episodes_per_opponent: int,
        max_steps: int,
        eval_pool: int,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    eval_opponents: list[tuple[str, str]] = []

    if not opponent_names:
        opponent_pool = _generate_eval_pool(eval_pool, opponent_generator)
        opponent_names = [name.split("|")[0] for name in opponent_pool]
        eval_opponents = [
            (f"{name}", opponent_team)
            for (i, opponent_team), name in zip(enumerate(opponent_pool), opponent_names)
        ]
    else:
        eval_opponents = [
            (opponent_name, TEAM_BY_NAME[opponent_name])
            for opponent_name in opponent_names
        ]

    for opponent_name, opponent_team in eval_opponents:
        eval_env = _build_train_env(
            agent_team=train_team,
            battle_format=battle_format,
            opponent_names=[],
            opponent_generator=None,
            rounds_per_opponent=episodes_per_opponent,
            opponent_pool=[opponent_team],
        )

        wins = 0
        losses = 0
        draws = 0

        for _ in range(episodes_per_opponent):
            total_reward, truncated = _play_episode(eval_env, model, max_steps=max_steps)
            if truncated:
                draws += 1
            elif total_reward > 0:
                wins += 1
            elif total_reward < 0:
                losses += 1
            else:
                draws += 1

        results.append(EvalResult(opponent=opponent_name, wins=wins, losses=losses, draws=draws))

    return results


def print_eval_summary(results: list[EvalResult]) -> None:
    print("\nEvaluation results (deterministic policy):")
    print("Opponent       Wins  Losses  Draws  Win rate")
    print("-" * 44)
    for result in results:
        print(
            f"{result.opponent:12} {result.wins:4d} {result.losses:7d} "
            f"{result.draws:6d} {100 * result.win_rate:7.2f}%"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DQN Pokémon bot against a pool and run per-opponent evaluation."
    )
    parser.add_argument("--train-team", default="steelix", choices=sorted(TEAM_BY_NAME))
    parser.add_argument("--pool", default=None, help="Comma-separated opponent pool (default: empty).")
    parser.add_argument("--pool-all", action="store_true", help="Insert all pre built solo teams to the pool",
                        default=False)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--rounds-per-opponent", type=int, default=2_000)
    parser.add_argument("--model-path", default="data/steelix")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-pool", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument("--eval-pool-all", action="store_true", help="Insert all pre built solo teams to the eval pool",
                        default=False)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only train and save model, skip evaluation.",
    )
    parser.add_argument("--random-generated", action="store_true", default=False)
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
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for generated dataset splitting.",
    )
    parser.add_argument("--format", type=str, default="gen9customgame")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    battle_format = args.format
    opponent_names = _parse_pool(args.pool, args.pool_all)
    train_team = TEAM_BY_NAME.get(args.train_team, STEELIX_TEAM)

    train_opponent_generator = None
    eval_opponent_generator = None
    if args.random_generated:
        data_path = 'data/gen9randombattle_db.json'
        if args.split_generated_pool:
            train_pool, eval_pool = _resolve_generated_pools(
                data_path=data_path,
                train_split=args.train_split,
                split_seed=args.split_seed,
            )
            train_opponent_generator = single_simple_team_generator(pokemon_pool=train_pool)
            eval_opponent_generator = single_simple_team_generator(pokemon_pool=eval_pool)
        else:
            shared_pool = load_pokemon_pool(data_path)
            train_opponent_generator = single_simple_team_generator(pokemon_pool=shared_pool)
            eval_opponent_generator = single_simple_team_generator(pokemon_pool=shared_pool)

    model = train_model(
        model_path=args.model_path,
        battle_format=battle_format,
        train_team=train_team,
        opponent_names=opponent_names,
        opponent_generator=train_opponent_generator,
        timesteps=args.timesteps,
        rounds_per_opponent=args.rounds_per_opponent,
    )

    if args.skip_eval:
        return

    eval_opponent_names = _parse_pool(args.pool, args.eval_pool_all) if not opponent_names else opponent_names
    eval_results = evaluate_model(
        model=model,
        battle_format=battle_format,
        train_team=train_team,
        opponent_names=eval_opponent_names,
        opponent_generator=eval_opponent_generator,
        episodes_per_opponent=args.eval_episodes,
        max_steps=args.eval_max_steps,
        eval_pool=args.eval_pool,
    )
    print_eval_summary(eval_results)


if __name__ == "__main__":
    main()
