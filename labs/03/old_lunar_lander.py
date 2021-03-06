#!/usr/bin/env python3
import numpy as np

import lunar_lander_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.06, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.06, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")

    # parser.add_argument("--alpha", default=0.02, type=float, help="Learning rate.")
    # parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    # parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
    # parser.add_argument("--epsilon_final", default=0.5, type=float, help="Final exploration factor.")
    # parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    # The environment has env.states states and env.actions actions.

    eps_update = (args.epsilon - args.epsilon_final) / args.episodes

    # TODO: Implement Q-learning RL algorithm.
    epsilon = args.epsilon
    # Q = np.zeros((env.states, env.actions))
    Q = 5*np.ones((env.states, env.actions))

    Q = np.load("Q3.dat.npy")

    # The overall structure of the code follows.
    for i in range(args.episodes):
        # Perform a training episode
        state, done = env.reset(), False
        if i % 100 == 0:
            print(Q.mean())

        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            rand = np.random.rand()
            if rand <= epsilon:
                action = np.random.randint(0, env.actions)
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)

            Q[state, action] = Q[state, action] + args.alpha * (
                    reward + args.gamma * np.max(Q[next_state, :]) - Q[state, action])
            state = next_state

        if i < args.episodes:
            epsilon = epsilon - eps_update

    np.save("Q4.dat", Q)

    env.reset(start_evaluate=True)

    for _ in range(100):

        # Perform an evaluation episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            action = np.argmax(Q[state, :])
            state, reward, done, _ = env.step(action)