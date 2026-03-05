import numpy as np

from env.env_builder import build_env
from env.battle_state import BattleState
from teams.single_teams import STEELIX_TEAM, VIKAVOLT_TEAM
from env.embed import MOVE_EMBED_LEN


def debug_run(n_steps=5):
    env = build_env(
        agent_team=STEELIX_TEAM,
        battle_format="gen9nationaldex",
        opponent_names=["regigigas"],
        opponent_generator=None,
        rounds_per_opponent=1,
        strict=True,
    )

    obs, _ = env.reset()
    for step in range(n_steps):
        action = np.int64(6)
        obs, reward, terminated, truncated, _ = env.step(action)


        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    debug_run()
