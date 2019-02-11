#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import cart_pole_pixels_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, num_actions):
        with self.session.graph.as_default():
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            self.actions = tf.placeholder(tf.int32, [None])
            self.returns = tf.placeholder(tf.float32, [None])

            # TODO: Add network running inference.
            #
            # For generality, we assume the result is in `self.predictions`.
            #
            # Only this part of the network will be saved, in order not to save
            # optimizer variables (e.g., estimates of the gradient moments).
            hidden = tf.layers.conv2d(self.states, 8, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
            hidden = tf.layers.max_pooling2d(hidden, pool_size=(3, 3), strides=2)
            hidden = tf.layers.conv2d(self.states, 16, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
            hidden = tf.layers.max_pooling2d(hidden, pool_size=(3, 3), strides=2)
            hidden = tf.layers.conv2d(self.states, 16, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
            hidden = tf.layers.flatten(hidden)

            logits = tf.layers.dense(hidden, num_actions)

            self.predictions = tf.nn.softmax(logits)

            # Saver for the inference network
            self.saver = tf.train.Saver()

            # TODO: Training using operation `self.training`.
            l1 = tf.losses.sparse_softmax_cross_entropy(self.actions, logits, weights=self.returns)
            # l2 = tf.losses.mean_squared_error(self.returns, baseline)

            loss = l1

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")


            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        return self.session.run(self.predictions, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns })

    def save(self, path):
        self.saver.save(self.session, path, write_meta_graph=False, write_state=False)

    def load(self, path):
        self.saver.restore(self.session, path)

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=2000, type=int, help="Number of episodes to train for")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint path.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=10, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_pixels_evaluator.environment()

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    # Load the checkpoint if required
    if args.checkpoint:
        # Try extract it from embedded_data
        try:
            import embedded_data
            embedded_data.extract()
        except:
            pass
        network.load(args.checkpoint)

        # TODO: Evaluation
        while True:
            state, done = env.reset(True), False
            while not done:
                action = network.predict([state])[0].argmax()
                state, reward, done, _ = env.step(action)

    else:
        # TODO: Training
        for _ in range(args.episodes // args.batch_size):
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(args.batch_size):
                # Perform episode
                states, actions, rewards = [], [], []
                state, done = env.reset(), False
                while not done:
                    if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                        env.render()

                    # TODO: Compute action probabilities using `network.predict` and current `state`
                    probabilities = network.predict([state])[0]

                    # TODO: Choose `action` according to `probabilities` distribution (np.random.choice can be used)
                    action = np.random.choice(range(env.actions), p=probabilities)

                    next_state, reward, done, _ = env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state

                # TODO: Compute returns by summing rewards (with discounting)
                returns = []
                for reward in reversed(rewards):
                    if len(returns) == 0:
                        returns.append(reward)
                    else:
                        returns.append(reward + args.gamma * returns[-1])

                returns = reversed(returns)

                # TODO: Add states, actions and returns to the training batch
                batch_states .extend(states)
                batch_actions.extend(actions)
                batch_returns.extend(returns)

            # Train using the generated batch
            network.train(batch_states, batch_actions, batch_returns)

        # Save the trained model
        network.save("cart_pole_pixels/model")

        evaluating = True

        # TODO: Evaluation
        while True:
            state, done = env.reset(True), False
            while not done:
                action = network.predict([state])[0].argmax()
                state, reward, done, _ = env.step(action)
