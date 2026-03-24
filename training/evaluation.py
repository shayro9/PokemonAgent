import random
from dataclasses import dataclass

from poke_env.environment import SingleAgentWrapper

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from config.config import TEAM_BY_NAME
from env.env_builder import build_env
from env.singles_env_wrapper import PokemonRLWrapper
from teams.generators import InfinitePoolGenerator


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
        return (self.wins + self.draws / 2) / self.episodes


def _play_episode(eval_env: SingleAgentWrapper, model: MaskablePPO, max_steps: int) -> tuple[float, bool]:
    """Run one evaluation episode.

    :param eval_env: Evaluation environment wrapper.
    :param model: Trained policy model.
    :param max_steps: Maximum number of steps before truncation.
    :returns: A tuple of ``(total_reward, truncated)``."""
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


def build_fixed_eval_pool(
        opponent_names: list[str],
        opponent_generator: InfinitePoolGenerator,
        eval_episodes: int,
) -> list[str]:
    """Build a fixed evaluation pool once, to be reused across all evaluations.

    Resolves teams from named opponents or a generator and locks them in so
    every evaluation checkpoint sees the exact same set of opponents.

    :param opponent_names: Optional predefined opponent names to sample from.
    :param opponent_generator: Optional generator for opponent teams.
    :param eval_episodes: Number of episodes (determines pool size).
    :returns: A fixed list of packed opponent team strings."""
    if opponent_names:
        sampled_names = random.sample(opponent_names, k=min(eval_episodes, len(opponent_names)))
        return [TEAM_BY_NAME[name] for name in sampled_names]
    elif opponent_generator is not None:
        return _generate_eval_pool(eval_episodes, opponent_generator)
    else:
        raise ValueError("Must provide either opponent_names or opponent_generator to build eval pool.")


def _generate_eval_pool(pool_size: int, opponent_generator: InfinitePoolGenerator) -> list[str]:
    """Build an evaluation opponent pool from a generator.

    :param pool_size: Number of teams to sample.
    :param opponent_generator: Generator yielding packed opponent teams.
    :returns: A list of packed opponent team strings."""
    if opponent_generator is None:
        raise ValueError("Cannot generate eval pool without an opponent team generator.")

    return [next(opponent_generator) for _ in range(pool_size)]


def evaluate_model(
        model: MaskablePPO,
        timestep: int,
        train_team: str,
        battle_format: str,
        opponent_names: list[str],
        opponent_generator: InfinitePoolGenerator | None,
        eval_episodes: int,
        max_steps: int,
        agent_team_generator: InfinitePoolGenerator | None=None,
        battle_team_generator: InfinitePoolGenerator | None=None,
        fixed_eval_pool: list[str] | None = None,
) -> list[EvalResult]:
    """Evaluate a trained model against a selected opponent pool.

    :param battle_team_generator: full battle generator
    :param model: Trained model to evaluate.
    :param timestep: Training timestep associated with this evaluation.
    :param train_team: Agent team used for evaluation.
    :param battle_format: Showdown format used for evaluation battles.
    :param opponent_names: Optional predefined opponent names.
    :param opponent_generator: Optional generator for opponent teams.
    :param eval_episodes: Number of episodes to evaluate.
    :param max_steps: Maximum steps per episode.
    :param agent_team_generator: Optional generator for agent team rotation.
    :param fixed_eval_pool: Pre-built pool of opponent teams to reuse across
        evaluations. When provided, ``opponent_names`` and
        ``opponent_generator`` are ignored for pool construction.
    :returns: A list containing one ``EvalResult`` summary."""
    results: list[EvalResult] = []

    # TODO: add from args
    algo = "maskable_ppo"
    use_action_masking = (algo == "maskable_ppo")

    if fixed_eval_pool is not None:
        opponent_pool = fixed_eval_pool
    elif not opponent_names:
        if battle_team_generator:
            opponent_pool = []
            battle_team_generator.reset()
        else:
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

    eval_env = build_env(
        agent_team=train_team,
        battle_format=battle_format,
        opponent_names=[],
        opponent_generator=None,
        rounds_per_opponent=1,
        opponent_pool=opponent_pool,
        agent_team_generator=agent_team_generator,
        use_action_masking=use_action_masking,
        battle_team_generator=battle_team_generator,
    )

    wins = 0
    losses = 0
    draws = 0

    for _ in range(eval_episodes):

        total_reward, truncated = _play_episode(eval_env, model, max_steps=max_steps)
        battle = _get_last_battle(eval_env)
        if battle is not None and battle.finished:
            if battle.won:
                wins += 1
            elif battle.lost:
                losses += 1
            else:
                draws += 1

    results.append(EvalResult(timestep=timestep, wins=wins, losses=losses, draws=draws))

    return results


def print_eval_summary(results: list[EvalResult]) -> None:
    """Print per-timestep and aggregate evaluation statistics.

    :param results: Evaluation summaries to print.
    :returns: ``None``."""
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


def _get_last_battle(env):
    while hasattr(env, "env"):
        env = env.env
    if isinstance(env, PokemonRLWrapper):
        battle = env.get_last_battle()
        return battle
    return None
