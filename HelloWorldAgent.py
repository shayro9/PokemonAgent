import asyncio

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer


async def main():
    player = RandomPlayer(
        account_configuration=AccountConfiguration("bot_username", None),
        server_configuration=LocalhostServerConfiguration,
    )

    print("Sending challenge...")
    await player.send_challenges("shayromelech", n_challenges=1)
    print("Challenge sent!")


if __name__ == "__main__":
    asyncio.run(main())
