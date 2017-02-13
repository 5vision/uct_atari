import os
import argparse
import gym
from gym.wrappers import Monitor
from find_best_model import test_n_times
import numpy as np
from agents import KerasAgent, LasagneAgent, EnsebleAgent
from actions import flip_actions

parser = argparse.ArgumentParser(description="Run commands",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', type=str, default="MsPacman-v0", help="Environment name.")
parser.add_argument('--model', type=str, help="Path to model")
parser.add_argument('--act_rep', type=int, default=1, help="action repeat for uct + sl")
parser.add_argument('--seed', type=int, default=None, help="action repeat for uct + sl")
parser.add_argument('--n_runs', type=int, default=10, help="Number of runs")
parser.add_argument('--sample', action='store_true', help="Sample from policy")
parser.add_argument('--ensemble_mode', type=str, default=None, help="How to ensemmble flipping. If none no ensempling")
parser.add_argument('--eps', type=float, default=0, help="Epsilon in eps greedy strategy")
args = parser.parse_args()

monitor_fname = 'env_{}_model_{}_actrep_{}_seed_{}_nruns_{}_sample_{}_ensmode_{}_eps_{}'.\
    format(args.env, os.path.basename(args.model), args.act_rep, args.seed,
           args.n_runs, args.sample, args.ensemble_mode, args.eps)
monitor_dir = os.path.join('monitors/', monitor_fname)

env = gym.make(args.env)
env = Monitor(env, monitor_dir)
env.seed(args.seed)
np.random.seed(args.seed)

if args.model[-3:] == '.h5':
    gray_state = not ('color' in args.model)
    agent = KerasAgent(args.model, gray_state=gray_state)
    if args.ensemble_mode:
        agent_flip = KerasAgent(
            args.model, flip_actions(env.env.env.get_action_meanings()), gray_state
        )
        agent = EnsebleAgent([agent, agent_flip], args.ensemble_mode)
elif args.model[-4:] == '.pkl':
    agent = LasagneAgent(args.model)
else:
    raise ValueError('wrong file, model file end with ".h5" or ".pkl"')

agent.seed(args.seed)

print test_n_times(args.n_runs, env, agent, args.sample, args.act_rep, args.eps, True)
env.close()
