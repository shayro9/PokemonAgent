import asyncio

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer
from BasicAgents import DebugRLPlayer
from env_wrapper import PokemonRLWrapper
from teams.single_teams import *
from teams.team_generators import single_simple_team_generator


async def main():
    team = next(single_simple_team_generator(data_path='data/gen9nationaldex.json'))

    env = PokemonRLWrapper(
        battle_format="gen9nationaldex",
        team=STEELIX_TEAM,
        opponent_teams=STEELIX_TEAM,
        strict=False,
    )

    debug_player = DebugRLPlayer(
        env,
        battle_format="gen9nationaldex",
        team=STEELIX_TEAM,
    )

    print("Sending challenge...")
    await debug_player.send_challenges("shayromelech", n_challenges=1)
    print("Challenge sent!")


if __name__ == "__main__":
    asyncio.run(main())
