#!/usr/bin/env python3
import abc
import random

import collections
from collections import defaultdict
from argparse import Namespace
from abc import abstractmethod

import numpy as np


class Runner(abc.ABC):
    args: Namespace

    @abstractmethod
    def __init__(self, args: Namespace) -> None:
        self.args = args

    @abstractmethod
    def pick_action(self) -> int:
        pass

    @abstractmethod
    def update_params(self, action: int, reward: float) -> None:
        pass


class RunnerGreedy(Runner):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.Q = np.array([self.args.initial for _ in range(args.bandits)], dtype=np.float32)
        self.N = np.array([0 for _ in range(args.bandits)])

    def pick_action(self) -> int:
        if np.random.uniform() < self.args.epsilon:
            return np.random.randint(0, len(self.Q))
        else:
            return np.argmax(self.Q).item()

    def update_params(self, action: int, reward: float) -> None:
        self.N[action] += 1

        if self.args.alpha > 0:
            learning_rate = self.args.alpha
        else:
            learning_rate = (1 / self.N[action])

        increment = learning_rate * (reward - self.Q[action])
        self.Q[action] = self.Q[action] + increment


class RunnerUCB(RunnerGreedy):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.N = np.array([1.0/args.bandits for _ in range(args.bandits)])
        self.c = args.c

    def pick_action(self) -> int:
        t = self.N.sum()
        return np.argmax(self.Q + self.c * np.sqrt(np.log(t) / self.N)).item()


class RunnerGradient(Runner):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.alpha = args.alpha
        self.possible_actions = range(args.bandits)
        self.H = np.array([0 for _ in self.possible_actions], dtype=np.float32)

    def policy(self) -> np.ndarray:
        return np.exp(self.H) / np.exp(self.H).sum()

    def pick_action(self) -> int:
        return np.random.choice(self.possible_actions, size=1, p=self.policy())[0]

    def update_params(self, action: int, reward: float) -> None:
        self.H[action] += self.alpha * reward * (1 - self.policy()[action].item())


class MultiArmedBandits():
    def __init__(self, bandits, episode_length):
        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(np.random.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length
        #print("Initialized {}-armed bandit, maximum average reward is {}".format(bandits, np.max(self._bandits)))

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = np.random.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")

    parser.add_argument("--mode", default="greedy", type=str, help="Mode to use -- greedy, ucb and gradient.")
    parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
    parser.add_argument("--c", default=1., type=float, help="Confidence level in ucb.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
    parser.add_argument("--initial", default=0, type=float, help="Initial value function levels.")
    args = parser.parse_args()

    env = MultiArmedBandits(args.bandits, args.episode_length)

    average_rewards = []
    for episode in range(args.episodes):
        env.reset()

        # TODO: Initialize required values (depending on mode).
        if args.mode == "greedy":
            runner = RunnerGreedy(args)
        elif args.mode == "ucb":
            runner = RunnerUCB(args)
        elif args.mode == "gradient":
            runner = RunnerGradient(args)

        average_rewards.append(0)
        done = False
        while not done:
            # TODO: Action selection according to mode
            action = runner.pick_action()

            _, reward, done, _ = env.step(action)
            average_rewards[-1] += reward / args.episode_length

            # TODO: Update parameters
            runner.update_params(action, reward)

    # Print out final score as mean and variance of all obtained rewards.
    print("Final score: {}, variance: {}".format(np.mean(average_rewards), np.var(average_rewards)))

    fname = f"outputs/out.txt"
    with open(fname, "a+") as f:
        f.write(f"{args.mode},{args.alpha},{args.c},{args.epsilon},{args.initial},{np.mean(average_rewards)}\n")
