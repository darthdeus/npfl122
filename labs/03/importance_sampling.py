#!/usr/bin/env python3
# in team with:
# Jakub Arnold 2894e3d5-bd76-11e7-a937-00505601122b
# Petra Doubravová 7ac09119-b96f-11e7-a937-00505601122b
import numpy as np
import gym


def probab(action):
    if action == 1 or action == 2:
        return 0.5
    else:
        return 0


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    args = parser.parse_args()

    # Create the environment
    env = gym.make("FrozenLake-v0")
    env.seed(42)
    states = env.observation_space.n
    actions = env.action_space.n

    # Behaviour policy is uniformly random.
    # Target policy uniformly chooses either action 1 or 2.
    V = np.zeros(states)
    C = np.zeros(states)

    for _ in range(args.episodes):
        state, done = env.reset(), False

        # Generate episode
        episode = []
        while not done:
            action = np.random.choice(actions)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # TODO: Update V using weighted importance sampling.
        G = 0.0
        W = 1.0
        for state, action, reward in reversed(episode):
            W = W * probab(action) * actions  # target(action|state)/b(action|state)
            G = G + reward
            C[state] = C[state] + W
            if C[state] != 0:
                V[state] = V[state] + (W / C[state]) * (G - V[state])

    # Print the final value function V
    for row in V.reshape(4, 4):
        print(" ".join(["{:5.2f}".format(x) for x in row]))