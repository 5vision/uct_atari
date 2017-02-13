import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = 'device=cpu'

import argparse
from multiprocessing import Process
import os
from datetime import datetime
import run_uct


parser = argparse.ArgumentParser(description="Run commands",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', type=str, default="pong",
                    help="Environment name.")
parser.add_argument('--version', type=str, default="v0", help="Version of environment.")
parser.add_argument('--act_rep', type=int, default=1, help="How many times repeat choosen action.")
parser.add_argument('--max_steps', type=int, default=10000, help="Maximum number of steps in environment.")
parser.add_argument('--num_workers', default=8, type=int, help="Number of concurent workers.")
parser.add_argument('--runs_per_worker', default=1, type=int, help="Number of runs for each worker.")
parser.add_argument('--save_dir', type=str, default=None, help="Path where to save collected data.")
parser.add_argument('--sim_steps', default=100, type=int,
                    help="Number of simulations for selecting action with rollout policy.")
parser.add_argument('--search_horizont', default=100, type=int, help="Search_horizont for each simulation.")
parser.add_argument('--gamma', type=float, default=1., help="Discount factor for reward.")
parser.add_argument('--exploration', type=float, default=-2,
                    help="Coefficient of exploration part in action selecting during simulation.")
parser.add_argument('--prune_tree', action='store_true',
                    help="After choosing action with uct make tree pruning.\n"
                         "This means save tree and all visits for selecting new action from new state."
                         "Otherwise create new tree for selecting next new action.")
parser.add_argument('--rollout_agent_name', type=str, default=None,
                    help="Name of agent for rollouts: random or keras model filename.")
parser.add_argument('--behavior_agent_name', type=str, default=None,
                    help="Name of agent for behavior: random, keras model filename or 'uct'.")
parser.add_argument('--eps_greedy', type=float, default=0., help="Probability of selecting random action.")
parser.add_argument('--save_freq', type=int, default=50, help="Frequency of saving uct data.")
parser.add_argument('--report_freq', type=int, default=100, help="Frequency of reporting uct progress.")


def collect_data(env_name='pong', version='v0', act_rep=1, max_steps=10000,
                 rollout_agent_name=None, behavior_agent_name=None, eps_greedy=0,
                 sim_steps=20, search_horizont=20, gamma=1., exploration=1.,
                 prune_tree=False, report_freq=100, runs_per_worker=1,
                 num_workers=8, save_dir=None, save_freq=10,):

    if save_dir is None:
        save_dir = os.path.join('data', env_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save experiment info
    fname_experiment = os.path.join(save_dir, 'experiment_info.txt')
    with open(fname_experiment, 'wb') as f:
        f.write('env_name: {}\n'.format(env_name))
        f.write('version: {}\n'.format(version))
        f.write('act_rep: {}\n'.format(act_rep))
        f.write('max_steps: {}\n'.format(max_steps))
        f.write('sim_steps: {}\n'.format(sim_steps))
        f.write('search_horizont: {}\n'.format(search_horizont))
        f.write('gamma: {}\n'.format(gamma))
        f.write('exploration: {}\n'.format(exploration))
        f.write('prune_tree: {}\n'.format(prune_tree))
        f.write('runs_per_worker: {}\n'.format(runs_per_worker))
        f.write('num_workers: {}\n'.format(num_workers))
        f.write('rollout_agent_name: {}\n'.format(rollout_agent_name))
        f.write('behavior_agent_name: {}\n'.format(behavior_agent_name))
        f.write('save_freq: {}\n'.format(save_freq))
        f.write('report_freq: {}\n'.format(report_freq))
        f.write('eps_greedy: {}\n'.format(args.eps_greedy))

    # start workers
    workers = []
    for i in xrange(num_workers):
        w = Process(
            target=run_uct.run,
            args=(env_name, version, act_rep, max_steps,
                  rollout_agent_name, behavior_agent_name,
                  eps_greedy, sim_steps, search_horizont,
                  gamma, exploration, prune_tree, report_freq,
                  runs_per_worker, save_dir, save_freq, i)
        )
        w.daemon = True
        w.start()
        workers.append(w)

    for w in workers:
        w.join()


if __name__ == '__main__':
    args = parser.parse_args()
    collect_data(
        args.env, args.version, args.act_rep, args.max_steps,
        args.rollout_agent_name, args.behavior_agent_name,
        args.eps_greedy, args.sim_steps, args.search_horizont,
        args.gamma, args.exploration, args.prune_tree,
        args.report_freq, args.runs_per_worker, args.num_workers,
        args.save_dir, args.save_freq,
    )
