from env_wrapper import PokemonRLWrapper


def test_pettingzoo_logic():
    # 1. Initialize the environment
    env = PokemonRLWrapper()
    env.reset()

    print(f"Agents detected: {env.agents}")

    # 2. Iterate through turns
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # Test your observation shape
            print(f"Agent {agent} sees state of shape: {observation.shape}")

            # Sample a random action
            action = env.action_space(agent).sample()

        env.step(action)

        print(f"Agent {agent} got reward: {reward}")

        if all(env.terminations.values()):
            break

    print("PettingZoo loop completed successfully!")