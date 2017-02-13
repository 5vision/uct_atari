import random
import numpy as np
from math import sqrt, log


def rand_argmax(lst):
    max_l = max(lst)
    best_is = [i for i, l in enumerate(lst) if l == max_l]
    i = random.randint(0, len(best_is)-1)
    return best_is[i]


def ucb(visits, value, parent_visits, exploration):
    return value + exploration * sqrt(log(parent_visits)/visits)


def moving_average(v, n, vi):
    return (n * v + vi) / (n + 1)


class Node(object):
    n_actions = 0

    def __init__(self, state, terminal=0, reward=0, parent=None, parent_action=None):
        self.state = state
        self.terminal = terminal
        self.reward = reward  # immediate reward during simulation for calculating parent reward
        self.parent = parent  # parent Node
        self.parent_action = parent_action
        self.childs = {}  # dict of child nodes for each action, key is action, value  is dict of nodes where key is observation
        self.a_visits = [0] * self.n_actions  # list of visits for each action, index is action
        self.a_values = [0.] * self.n_actions  # list of visits for each action, index is action
        self.value = 0.  # current value for node
        self.visits = 0  # current number of visits for node

    def best_action(self):
        return rand_argmax(self.a_visits)

    def selection(self, env, exploration):
        # restore env state
        env.restore_state(self.state)

        # some actions have not been used, explore leaf node
        if len(self.childs) < self.n_actions:
            # it is sure leaf
            leaf = True
            
            # choose action
            a = random.choice([a for a in xrange(self.n_actions) if a not in self.childs])

            f, r, t, _ = env.step(a)
            s = env.clone_state()

            # create new node
            node = Node(s, t, r, self, a)
            
            # add child to parent
            self.childs[a] = ({f: node})

        # all action have been taken, choose according to max q value + exploration
        else:
            # if exploration negative do normalization
            if exploration == -1.:
                a_values = [(a - np.mean(self.a_values)) / (np.std(self.a_values)+1e-8) for a in self.a_values]
                exploration = 1.414  # sqrt(2)
            elif exploration == -2.:
                a_values = np.asarray(self.a_values) - np.min(self.a_values)
                a_values /= (np.max(a_values) + 1e-8)
                exploration = 1.414  # sqrt(2)
            elif exploration >= 0:
                a_values = self.a_values
            else:
                raise ValueError('wrong value fo exploration: {}'.format(exploration))

            # calculate ucb value for each action
            a_vals = [ucb(self.a_visits[i], a_values[i], self.visits, exploration)
                      for i in xrange(self.n_actions)]
            a = rand_argmax(a_vals)

            # here we should make env.step because environment is stochastic
            f, r, t, _ = env.step(a)

            # check if we have been in this state
            if f in self.childs[a]:
                node = self.childs[a][f]
                node.reward = r
                node.terminal = t
                leaf = False
            else:
                # create new node, this is leaf node now
                s = env.clone_state()
                node = Node(s, t, r, self, a)
                self.childs[a][f] = node
                leaf = True

        return node, leaf


def uct_action(env, agent, node, sim_steps, search_horizont, gamma, exploration=1.):
    
    # do simulations
    for _ in xrange(sim_steps):
        sample(env, agent, node, search_horizont, gamma, exploration)

    #choose action
    return node.best_action()


def sample(env, agent, node, search_horizont, gamma, exploration):
    
    depth = 0
    leaf = False

    #while not leaf and depth < search_horizont:
    while not leaf:
        node, leaf = node.selection(env, exploration)
        depth += 1

        # break in terminal
        if node.terminal:
            break

    R = node.value
    if leaf and not node.terminal:
        R = rollout(env, agent, search_horizont, gamma)

    # backup
    update_values(R, node, gamma)


def rollout(env, agent, n_steps, gamma=0.99):
    R = 0.
    g = 1.
    step = 0

    # TODO: add parents buffers
    if hasattr(agent, 'reset_bufs'):
        agent.reset_bufs()

    while True:
        frame = env.env._get_image()
        a = agent.choose_action(frame)
        #a = env.action_space.sample()
        _, r, t, _, = env.step(a)
        R += r*g
        g *= gamma
        step += 1
        if t or 0 < n_steps <= step:
            break

    return R


def update_values(R, node, gamma=0.99):
    
    while node is not None:
        # update this node info
        node.value = moving_average(node.value, node.visits, R)
        node.visits += 1

        # calculate value for parent
        R = node.reward + gamma * R

        # update childs lists with data in parent node
        parent_node = node.parent
        a = node.parent_action
        if parent_node:
            parent_node.a_values[a] = moving_average(parent_node.a_values[a], parent_node.a_visits[a], R)
            parent_node.a_visits[a] += 1

        # make parent current node
        node = parent_node
