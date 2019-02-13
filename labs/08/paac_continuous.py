#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import continuous_mountain_car_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, tiles, weights, actions):
        with self.session.graph.as_default():
            self.states = tf.placeholder(tf.int32, [None, tiles])
            self.actions = tf.placeholder(tf.float32, [None, actions])
            self.returns = tf.placeholder(tf.float32, [None])

            # TODO: Because `self.states` are vectors of tile indices, convert them
            # to one-hot encoding and store them as `states`. I.e., for batch
            # example i, state should be a vector of length `weights` with `tiles` ones
            # on indices `self.states[i, 0..`tiles`-1] and the rest being zeros.
            encoded = tf.one_hot(self.states, weights)
            encoded = tf.reduce_sum(encoded, axis=-1)

            assert len(encoded.shape) == 2
            assert int(encoded.shape[1]) == tiles

            # Expert remark: The `states` representation is very sparse, so much better
            # performance can be achieved by converting it to `SparseTensor`. However,
            # the `tf.layers.dense` cannot process `SparseTensor` inputs, so you would have
            # to implement it manually using `tf.sparse_tensor_dense_matmul`.

            # TODO: Compute `self.mus` and `self.sds`, each of shape [batch_size, actions].
            # Compute each independently using `states` as input, adding a fully connected
            # layer with args.hidden_layer units and ReLU activation. Then:
            # - For `self.mus` add a fully connected layer with `actions` outputs.
            #   To avoid `self.mus` moving from the required [-1,1] range, you can apply
            #   `tf.tanh` activation.
            # - For `self.sds` add a fully connected layer with `actions` outputs
            #   and `tf.nn.softplus` activation.
            hidden_mu = tf.layers.dense(encoded, args.hidden_layer, activation=tf.nn.relu)
            hidden_sd = tf.layers.dense(encoded, args.hidden_layer, activation=tf.nn.relu)

            self.mus = tf.layers.dense(hidden_mu, actions, tf.nn.tanh)
            self.sds = tf.layers.dense(hidden_sd, actions, tf.nn.softplus)

            # TODO: Create `action_distribution` using tf.distributions.Normal
            # and computed `self.mus` and `self.sds`.
            mrgauss = tf.distributions.Normal(self.mus, self.sds)

            # TODO(reinforce_with_baseline): Compute `self.values`, starting with `states` and
            # - add a fully connected layer of size args.hidden_layer and ReLU activation
            # - add a fully connected layer with 1 output and no activation
            # - modify the result to have shape `[batch_size]` (you can use for example `[:, 0]`)
            hidden = tf.layers.dense(encoded, args.hidden_layer, activation=tf.nn.relu)
            self.values = tf.layers.dense(hidden, 1, activation=None)[:, 0]

            # TODO: Compute `loss` as a sum of three losses:
            # - negative log probability of the `self.actions` in the `action_distribution`
            #   (using `log_prob` method). You need to sum the log probabilities
            #   of subactions for a single batch example (using tf.reduce_sum with axis=1).
            #   Then weight the resulting vector by (self.returns - tf.stop_gradient(self.values))
            #   and compute its mean.
            # - negative value of the distribution entropy (use `entropy` method of
            #   the `action_distribution`) weighted by `args.entropy_regularization`.
            # - mean square error of the `self.returns` and `self.values`
            l1 = (self.returns - tf.stop_gradient(self.values)) * tf.reduce_sum(mrgauss.log_prob(self.actions), axis=1)
            l2 = - args.entropy_regularization * mrgauss.entropy()
            l3 = tf.losses.mean_squared_error(self.returns, self.values)

            loss = tf.reduce_mean(l1) + tf.reduce_mean(l2) + tf.reduce_mean(l3)

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict_actions(self, states):
        return list(zip(*self.session.run([self.mus, self.sds], {self.states: states})))

    def predict_values(self, states):
        return self.session.run(self.values, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--tiles", default=8, type=int, help="Tiles to use.")
    parser.add_argument("--workers", default=5, type=int, help="Number of parallel workers.")
    args = parser.parse_args()

    # Create the environment
    env = continuous_mountain_car_evaluator.environment(tiles=args.tiles)
    assert len(env.action_shape) == 1
    action_lows, action_highs = env.action_ranges

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, args.tiles, env.weights, env.action_shape[0])

    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)
    while True:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Choose actions using network.predict_actions.
            # using np.random.normal to sample action and np.clip
            # to clip it using action_lows and action_highs,
            preds = network.predict_actions(states)

            actions = [np.clip(np.random.normal(p[0], p[1]), action_lows, action_highs)
                       for p in preds]

            # TODO: Perform steps by env.parallel_step
            xxx = env.parallel_step(actions)

            next_states, rewards, dones, _ = list(zip(*xxx))
            next_values = network.predict_values(np.array(next_states))

            # TODO: Compute return estimates by
            # - extracting next_states from steps
            # - computing value function approximation in next_states
            # - estimating returns by reward + (0 if done else args.gamma * next_state_value)
            returns = []
            for next_value, reward, action, done in zip(next_values, rewards, actions, dones):
                if done:
                    ret = reward
                else:
                    ret = reward  + args.gamma * next_value

                returns.append(ret)
                #     next_state_value = np.zeros_like(next_states, dtype=np.float32)
                # else:
                #     next_state_value = network.predict_values(next_states)

                # TODO: next_state_value only for action?
                # returns.append(reward + args.gamma * next_state_value[action])

            # returns = reward + args.gamma * next_state_value
            states  = np.array(states,   dtype=np.float32)
            actions = np.array(actions, dtype=np.int32)
            returns = np.array(returns, dtype=np.float32)

            # actions = actions[..., 0]
            # print(states.shape, actions.shape, returns.shape)
            # __import__('ipdb').set_trace()

            # TODO: Train network using current states, chosen actions and estimated returns
            network.train(states, actions, returns)

            states = next_states

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                action, _ = network.predict_actions([state])[0]
                state, reward, done, _ = env.step(action)
