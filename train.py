import argparse
from dataclasses import dataclass

from stable_baselines3 import DQN
from poke_env import LocalhostServerConfiguration, RandomPlayer
from poke_env.environment import SingleAgentWrapper

from env_wrapper import PokemonRLWrapper
from teams.single_teams import ALL_SOLO_TEAMS, STEELIX_TEAM, shuffled_team_generator


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


def _parse_pool(raw_pool: str | None) -> list[str]:
    if not raw_pool:
        return [name for name, _ in ALL_SOLO_TEAMS]

    names = [name.strip() for name in raw_pool.split(",") if name.strip()]
    unknown = [name for name in names if name not in TEAM_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown pokemon(s) in pool: {', '.join(unknown)}. "
            f"Valid names: {', '.join(sorted(TEAM_BY_NAME))}"
        )
    return names


def _build_train_env(agent_team: str, opponent_names: list[str], rounds_per_opponent: int) -> SingleAgentWrapper:
    opponent_pool = [TEAM_BY_NAME[name] for name in opponent_names]
    opponent_team_gen = shuffled_team_generator(opponent_pool)

    opponent_policy = RandomPlayer(
        battle_format="gen9nationaldex",
        server_configuration=LocalhostServerConfiguration,
    )

    agent = PokemonRLWrapper(
        battle_format="gen9nationaldex",
        team=agent_team,
        opponent_teams=opponent_pool,
        opponent_team_generator=opponent_team_gen,
        rounds_per_opponents=rounds_per_opponent,
        server_configuration=LocalhostServerConfiguration,
        strict=False,
    )

    return SingleAgentWrapper(agent, opponent_policy)


def train_model(
    model_path: str,
    train_team: str,
    opponent_names: list[str],
    timesteps: int,
    rounds_per_opponent: int,
) -> DQN:
    train_env = _build_train_env(train_team, opponent_names, rounds_per_opponent)

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=20_000,
        learning_starts=2_000,
        batch_size=64,
        tau=1.0,
        gamma=0.999,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
        verbose=1,
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


def evaluate_model(
    model: DQN,
    train_team: str,
    opponent_names: list[str],
    episodes_per_opponent: int,
    max_steps: int,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    for opponent_name in opponent_names:
        eval_env = _build_train_env(
            agent_team=train_team,
            opponent_names=[opponent_name],
            rounds_per_opponent=episodes_per_opponent,
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
    parser.add_argument(
        "--pool",
        default=None,
        help="Comma-separated opponent pool (default: all solo teams).",
    )
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--rounds-per-opponent", type=int, default=2_000)
    parser.add_argument("--model-path", default="steelix")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only train and save model, skip evaluation.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    opponent_names = _parse_pool(args.pool)
    train_team = TEAM_BY_NAME.get(args.train_team, STEELIX_TEAM)

    model = train_model(
        model_path=args.model_path,
        train_team=train_team,
        opponent_names=opponent_names,
        timesteps=args.timesteps,
        rounds_per_opponent=args.rounds_per_opponent,
    )

    if args.skip_eval:
        return

    eval_results = evaluate_model(
        model=model,
        train_team=train_team,
        opponent_names=opponent_names,
        episodes_per_opponent=args.eval_episodes,
        max_steps=args.eval_max_steps,
    )
    print_eval_summary(eval_results)


if __name__ == "__main__":
    main()
