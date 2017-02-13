import os
import argparse
import gym
import numpy as np
from agents import KerasAgent, LasagneAgent, EnsebleAgent
from actions import flip_actions


parser = argparse.ArgumentParser(description="Run commands",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', type=str, default="MsPacman-v0", help="Environment name.")
parser.add_argument('--n_runs', type=int, default=10, help="Number of runs")
parser.add_argument('--act_rep', type=int, default=1, help="action repeat for uct + sl")

UCT_SL_DIRS = [
    #'models/cent_artem',
    'models/pacman_openai2_color_aug_flip_nfr4_d2_cont',
    #'models/cent_openai2_d2_gray_flip_nf4_augment',
]
A3C_DIRS = [
    #'models/cent_a3cth'
]


def test_env(env, agent, sample=False, act_rep=1, epsilon=0.0, verbose=False):
    step = 0
    total_reward = 0
    terminal = False
    frame = env.reset().copy()
    agent.reset()

    while not terminal:

        if np.random.rand() <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.choose_action(frame.copy(), sample)

        for _ in xrange(act_rep):
            frame, r, terminal, _ = env.step(a)
            total_reward += r
            if terminal:
                break

        step += 1

    if verbose:
        print 'steps: {}, reward {}'.format(step, total_reward)

    return total_reward


def test_n_times(n_times, env, agent, sample=False, act_rep=1, epsilon=0.0, verbose=False):
    rewards = [test_env(env, agent, sample, act_rep, epsilon, verbose) for _ in xrange(n_times)]
    return np.mean(rewards), np.median(rewards)


if __name__ == '__main__':
    args = parser.parse_args()
    env = gym.make(args.env)
    all_results = []

    for d in UCT_SL_DIRS:

        for f_name in os.listdir(d):
            f_path = os.path.join(d, f_name)

            if f_path[-3:] == '.h5':
                if 'color' in f_path:
                    agent1 = KerasAgent(f_path, gray_state=False)
                    agent2 = KerasAgent(f_path, flip_actions(env.get_action_meanings()),  gray_state=False)
                else:
                    agent1 = KerasAgent(f_path, gray_state=True)
                    agent2 = KerasAgent(f_path, flip_actions(env.get_action_meanings()), gray_state=True)

                print '\nmodel {}'.format(f_path)
                for eps in [0, 0.01, 0.05]:
                    mean_reward, median_reward = test_n_times(args.n_runs, env, agent1, False, args.act_rep)
                    report_str = 'singe model: eps {}; mean reward {}; median reward {}'.\
                        format(eps, mean_reward, median_reward)
                    all_results.append((mean_reward, median_reward, f_path, report_str))
                    print report_str

                    for ensemple_mode in ('mean', 'sample_hard'):
                        agent = EnsebleAgent([agent1, agent2], ensemple_mode)
                        mean_reward, median_reward = test_n_times(args.n_runs, env, agent, False, args.act_rep)
                        report_str = 'flip_ensemple {}; eps {}; mean reward {}; median reward {}'.\
                            format(ensemple_mode, eps, mean_reward, median_reward)
                        all_results.append((mean_reward, median_reward, f_path, report_str))
                        print report_str

    for d in A3C_DIRS:
        for f_name in os.listdir(d):
            f_path = os.path.join(d, f_name)

            if f_path[-4:] == '.pkl':
                agent = LasagneAgent(f_path)

                print '\nmodel {}'.format(f_path)
                for eps in [0, 0.01, 0.05]:
                    for sample in (True, False):

                        if sample and eps > 0:
                            continue

                        mean_reward, median_reward = test_n_times(args.n_runs, env, agent, sample, 1)
                        report_str = 'sample {}; eps {}; mean reward {}; median reward {}'.\
                                format(sample, eps, mean_reward, median_reward)
                        all_results.append((mean_reward, median_reward, f_path, report_str))
                        print report_str

    all_results_mean_sort = sorted(all_results, key=lambda x: x[0], reverse=True)
    all_results_median_sort = sorted(all_results, key=lambda x: x[1], reverse=True)

    print 'top 3 mean:'
    for r in all_results_mean_sort[:3]:
        print r[2], r[3]

    print '\ntop 3 median:'
    for r in all_results_median_sort[:3]:
        print r[2], r[3]
