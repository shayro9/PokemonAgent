import asyncio

from agents.BasicAgents import DebugRLPlayer
from env.singles_env_wrapper import PokemonRLWrapper
from teams.single_teams import *
from teams.team_generators import single_simple_team_generator


async def main():
    """Create a debug player and send a single challenge.
    
    :returns: ``None``."""
    env = PokemonRLWrapper(

    )

    player = DebugRLPlayer(
        battle_format="gen9customgame",
        team=STEELIX_TEAM,
        env=env,
    )

    print("Sending challenge...")
    await player.send_challenges("shayromelech", n_challenges=1)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
