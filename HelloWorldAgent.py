import asyncio

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer
from BasicAgents import DebugRLPlayer
from env_wrapper import PokemonRLWrapper
from teams.single_teams import *


async def main():
    env = PokemonRLWrapper(
        battle_format="gen9nationaldex",
        team=STEELIX_TEAM,
        opponent_team=CONKELDURR_TEAM,
        strict=False,
    )

    debug_player = DebugRLPlayer(
        env,
        battle_format="gen9nationaldex",
        team=CONKELDURR_TEAM,
    )

    print("Sending challenge...")
    await debug_player.send_challenges("shayromelech", n_challenges=1)
    print("Challenge sent!")


if __name__ == "__main__":
    asyncio.run(main())
