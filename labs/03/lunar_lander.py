# #!/usr/bin/env python3
# import numpy as np
#
# import lunar_lander_evaluator
#
# if __name__ == "__main__":
#     # Fix random seed
#     np.random.seed(42)
#
#     # Parse arguments
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
#     parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
#
#     parser.add_argument("--alpha", default=None, type=float, help="Learning rate.")
#     parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
#     parser.add_argument("--epsilon", default=None, type=float, help="Exploration factor.")
#     parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
#     parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
#     args = parser.parse_args()
#
#     # Create the environment
#     env = lunar_lander_evaluator.environment()
#
#     # The environment has env.states states and env.actions actions.
#
#     # TODO: Implement a suitable RL algorithm.
#     #
#     # The overall structure of the code follows.
#     while training:
#
#         # To generate expert trajectory, you can use
#         # state, trajectory = env.expert_trajectory()
#
#         # Perform a training episode
#         state, done = env.reset(), False
#         while not done:
#             if args.render_each and env.episode and env.episode % args.render_each == 0:
#                 env.render()
#
#             next_state, reward, done, _ = env.step(action)
#
#     # Perform last 100 evaluation episodes

#!/usr/bin/env python3
import glob
import numpy as np
import random
import tensorflow as tf
tf.enable_eager_execution()

import lunar_lander_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=100, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    # TODO: Implement Q-learning RL algorithm.
    #
    # The overall structure of the code follows.
    training = True

    Q = np.random.random([env.states, env.actions])
    C = np.zeros_like(Q)
    # Q.fill(200)

    eps_diff = (args.epsilon_final - args.epsilon) / float(args.episodes)
    eps_curr = args.epsilon

    iter = 0

    trajs = []

    def get_trajectory():
        state, trajectory = env.expert_trajectory()

        full_trajectory = []
        for action, reward, next_state in trajectory:
            full_trajectory.append((state, action, reward, next_state))
            state = next_state

        return full_trajectory

    # trajectories = [get_trajectory() for _ in range(10000)]
    # np.save("trajectories.npy", np.array(trajectories))
    # print("generated trajectories")
    trajectories = np.load("trajectories.npy")
    print("loaded trajectories")

    maxname = max([int(name.replace("logs/", "")) for name in glob.glob("logs/*")])

    logdir = "./logs/{}".format(maxname + 1)
    print("Logging into {}".format(logdir))

    global_step = tf.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    def imitation():
        # state, trajectory = env.expert_trajectory()
        # trajectory = get_trajectory()
        # trajs.extend(trajectory)

        # ITER_PER_TRAJECTORY = 1

        # for _ in range(ITER_PER_TRAJECTORY):
        for trajectory in trajectories:
            rev = list(reversed(trajectory))

            G = 0.0

            for state, action, reward, next_state in trajectory:
                G = args.gamma * G + reward

                C[state, action] += 1
                Q[state, action] += (G - Q[state, action])/C[state, action]

                # Q[state, action] = G

                # G = args.gamma * G

                    # Q[state, action] += 1.0 * \
                    #         (reward + 0.98*np.max(Q[next_state]) - Q[state, action])

        print("{}\tepsilon: {}\t{}\t{}".format(iter, eps_curr, Q.mean(), len(trajs)))

        state, done = env.reset(), False
        while not done:
            env.render()

            action = np.argmax(Q[state]).item()
            state, _, done, _ = env.step(action)

    print("Imitation START")

    global_step.assign_add(1)
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.histogram("Q", Q.reshape(-1))

    imitation()

    global_step.assign_add(1)
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.histogram("Q", Q.reshape(-1))

    print("Imitation DONE, continuing with Q-learning")

    while training:
        global_step.assign_add(1)
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            tf.contrib.summary.histogram("Q", Q.reshape(-1))

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

            # max_qsa = np.max(Q[next_state]).item()
            Q[state, action] += 1.0 * \
                    (reward + args.gamma*np.max(Q[next_state]) - Q[state, action])

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
