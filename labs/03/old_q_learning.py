#!/usr/bin/env python3
# in team with:
# Jakub Arnold 2894e3d5-bd76-11e7-a937-00505601122b
# Petra Doubravová 7ac09119-b96f-11e7-a937-00505601122b
import numpy as np

import mountain_car_evaluator


def epsilon_greedy_action(state):
    rand = np.random.rand()
    if rand <= epsilon:
        return np.random.randint(0, env.actions)
    else:
        return np.argmax(Q[state, :])


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(2)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.45, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.06, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment()

    mean = np.zeros(100)
    i = 0
    training = True

    eps_step_update = (args.epsilon - args.epsilon_final) / args.episodes

    # TODO: Implement Q-learning RL algorithm.
    epsilon = args.epsilon
    Q = np.zeros((env.states, env.actions))

    while training:
        if i > args.episodes:
            break

        state, done = env.reset(), False
        avg_reward = 0

        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            action = epsilon_greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            avg_reward += reward
            Q[state, action] = Q[state, action] + args.alpha * (
                        reward + args.gamma * np.max(Q[next_state, :]) - Q[state, action])
            state = next_state

        i += 1
        mean[i % 100] = avg_reward
        if i >= 100:

            avg = np.average(mean)
            if avg > -50:
                training = False
        if i < args.episodes:
            epsilon = epsilon - eps_step_update

    # Perform last 100 evaluation episodes
    while True:
        state, done = env.reset(True), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            action = np.argmax(Q[state, :])
            next_state, reward, done, _ = env.step(action)
            state = next_state