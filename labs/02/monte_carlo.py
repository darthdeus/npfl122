#!/usr/bin/env python3
import numpy as np
import random

from collections import defaultdict

import cart_pole_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=500, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    # TODO: Implement Monte-Carlo RL algorithm.
    #
    # The overall structure of the code follows.
    training = True

    returns = defaultdict(lambda: [])
    Q = np.zeros([env.states, env.actions], dtype=np.float32)
    # Q.fill(50.)

    C = np.zeros([env.states, env.actions], dtype=np.float32)

    policy = np.random.randint(0, env.actions, env.states, dtype=np.int32)

    eps_diff = (args.epsilon_final - args.epsilon) / float(args.episodes)
    eps_curr = args.epsilon

    episode_num = 0

    while training:
        episode_num += 1
        G = 0.0

        # Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            if random.random() < eps_curr:
                action = random.randint(0, env.actions - 1)
            else:
                action = policy[state].item()

            next_state, reward, done, _ = env.step(action)

            # print("G: {}, action: {}, policy: {}".format(G, action, policy[:5]))

            G = args.gamma * G + reward

            C[state, action] += 1
            # returns[(state, action)].append(G)
            # Q[state, action] = np.mean(returns[(state, action)]).item()
            Q[state, action] += (G - Q[state, action])/C[state, action]

            policy[state] = np.argmax(Q[state, :]).item()

            state = next_state

        eps_curr += eps_diff

        if episode_num % args.render_each == 0:
            print(f"eps curr: {eps_curr}")

            # Evaluation episode
            state, done = env.reset(), False
            while not done:
                env.render()
                action = policy[state].item()
                state, _, done, _ = env.step(action)


    # Perform last 100 evaluation episodes
