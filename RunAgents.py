import numpy as np
from stable_baselines3 import DQN
from poke_env import LocalhostServerConfiguration

from env_wrapper import PokemonRLWrapper
from teams.single_teams import STEELIX_TEAM, CONKELDURR_TEAM


def run_match(
    model1_path: str,
    model2_path: str,
    team1: str,
    team2: str,
    battle_format: str = "gen9nationaldex",
    deterministic: bool = True,
    max_steps: int = 500,
):
    env = PokemonRLWrapper(
        battle_format=battle_format,
        team=team1,
        opponent_team=team2,
        server_configuration=LocalhostServerConfiguration,
        strict=False,
    )

    model1 = DQN.load(model1_path)
    model2 = DQN.load(model2_path)

    obs, info = env.reset()
    a1 = env.agent1.username
    a2 = env.agent2.username

    print(f"Agents: {a1} (model {model1_path}) vs {a2} (model {model2_path})")

    step_i = 0
    while env.agents and step_i < max_steps:
        o1 = obs[a1]
        o2 = obs[a2]

        act1, _ = model1.predict(o1, deterministic=deterministic)
        act2, _ = model2.predict(o2, deterministic=deterministic)

        act1 = np.int64(act1)
        act2 = np.int64(act2)

        obs, rewards, terms, truncs, infos = env.step({a1: act1, a2: act2})

        env.render()

        print(
            f"\nStep {step_i} | a1={int(act1)} r1={rewards[a1]:+.3f} term1={terms[a1]} trunc1={truncs[a1]} "
            f"| a2={int(act2)} r2={rewards[a2]:+.3f} term2={terms[a2]} trunc2={truncs[a2]}"
        )

        step_i += 1

        if terms[a1] or truncs[a1] or terms[a2] or truncs[a2]:
            break

    env.close()
    print("\nMatch finished.")


if __name__ == "__main__":
    run_match(
        model1_path="steelix.zip",
        model2_path="conkeldurr.zip",
        team1=STEELIX_TEAM,
        team2=CONKELDURR_TEAM,
        deterministic=True,
    )
