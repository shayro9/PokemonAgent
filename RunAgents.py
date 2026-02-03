import asyncio

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer
from BasicAgents import DebugRLPlayer


async def main():
    player1 = DebugRLPlayer(
        account_configuration=AccountConfiguration("bot_1", None),
        server_configuration=LocalhostServerConfiguration,
    )

    player2 = RandomPlayer(
        account_configuration=AccountConfiguration("bot_2", None),
        server_configuration=LocalhostServerConfiguration,
    )

    print("Sending challenge...")
    await player1.battle_against(player2, n_battles=1)
    print("Challenge sent!")


if __name__ == "__main__":
    asyncio.run(main())
