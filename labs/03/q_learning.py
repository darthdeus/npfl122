#!/usr/bin/env python3
import numpy as np
import random

import mountain_car_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=250, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment()

    # TODO: Implement Q-learning RL algorithm.
    #
    # The overall structure of the code follows.
    training = True

    Q = np.random.random([env.states, env.actions])
    Q.fill(-300)

    eps_diff = (args.epsilon_final - args.epsilon) / float(args.episodes)
    eps_curr = args.epsilon

    while training:
        # Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            if random.random() < eps_curr:
                action = random.randint(0, env.actions - 1)
            else:
                action = np.argmax(Q[state]).item()

            next_state, reward, done, _ = env.step(action)

            max_qsa = np.max(Q[next_state]).item()
            # TODO: ???????????/
            Q[state, action] += args.alpha * (reward + args.gamma*(max_qsa - Q[state, action]))

            state = next_state

        eps_curr = max(0.05, eps_curr + eps_diff)
        if env.episode % args.render_each == 0:
            print("epsilon: {}".format(eps_curr))

            state, done = env.reset(), False
            while not done:
                env.render()

                action = np.argmax(Q[state]).item()
                state, _, done, _ = env.step(action)

    # Perform last 100 evaluation episodes
