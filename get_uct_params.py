import numpy as np
from gym.envs.atari import AtariEnv
import random
import argparse

MAX_TIME_STEP = 3000
parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('name', type=str, help="Environment name.")
args = parser.parse_args()


def get_charts(name):
    charts = []
    print ('###############################' + name + '###############################')
    const_best_score = -100000
    chart_const_best = ([0], [0])
    env = AtariEnv(game=name, obs_type='image', frameskip=(2, 5), repeat_action_probability=0.25)#gym.make(name)
    env.reset()
    actionN = env.action_space.n
    policy = []
    max_steps = 30000
    n_episode = 50
    rew = [[0] for _ in range(n_episode)]
    p = lambda na: random.randint(0, na - 1)
    cnt = 0
    for i_episode in range(n_episode):
        ob = env.reset()
        while cnt < max_steps:
            cnt+=1
            action = p(actionN)#env.action_space.sample()
            S_prime, r, done, info = env.step(action)
            rew[i_episode].append(r)
            if done:
                break
        cnt = 0
        rew[i_episode] = np.array(rew[i_episode])
    env.close()
    return rew


def get_average(rewards):
    sm = np.zeros(max([len(rs) for rs in rewards]))
    for i in range(max([len(rs) for rs in rewards])):
        cnt = 0
        for r in rewards:
            if i < len(r):
                cnt+=1
                sm[i] += r[i]
        sm[i] = sm[i] / cnt
    return [np.sum(sm[:i + 1]) for i in range(len(sm))]


def get_rollout_horizon(name):
    points = []
    rewards = get_charts(name)
    y_average = get_average(rewards)
    p = (y_average[0], y_average[len(y_average) // 2], y_average[-1])
    TO_THE_END = -1
    d1 = abs(p[1] - p[0])
    d2 = abs(p[2] - p[1])
    res = MAX_TIME_STEP
    if (d1 != 0):
        if d2 / d1 < 5:
            res = 100
        else:
            res = TO_THE_END
    else:
        res = TO_THE_END

    act_rep = 1
    if res == TO_THE_END:
        act_rep = 2

    res = 'search_horizont {}, act_rep {}'.format(res, act_rep)
    return res


print (get_rollout_horizon(args.name))
