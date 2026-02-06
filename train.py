import asyncio
from stable_baselines3 import DQN
from env_wrapper import PokemonRLWrapper
from poke_env import RandomPlayer, AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment import SingleAgentWrapper
from teams.single_teams import *


def main():
    team1 = STEELIX_TEAM
    team2 = CONKELDURR_TEAM

    opponent = RandomPlayer(
        battle_format="gen9nationaldex",
        team=team2,
        server_configuration=LocalhostServerConfiguration,
    )

    agent = PokemonRLWrapper(
        battle_format="gen9nationaldex",
        team=team1,
        opponent_team=team2,
        server_configuration=LocalhostServerConfiguration,
        strict=False,
    )

    train_env = SingleAgentWrapper(agent, opponent)

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

    print("Starting training...")
    model.learn(total_timesteps=20000)
    model.save("steelix")
    print("Training complete! Model saved as steelix_dqn_model.zip")


if __name__ == "__main__":
    main()
