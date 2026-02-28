import asyncio

from agents.BasicAgents import DebugRLPlayer
from env.singles_env_wrapper import PokemonRLWrapper
from teams.single_teams import *
from teams.team_generators import single_simple_team_generator


async def main():
    """Create a debug player and send a single challenge.
    
    :returns: ``None``."""
    env = DebugRLPlayer(
        battle_format="gen9nationaldex",
        team=REGIGIGAS_TEAM,
    )

    print("Sending challenge...")
    await env.send_challenges("shayromelech", n_challenges=1)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
