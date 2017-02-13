import os
import numpy as np
from gym.envs.atari import AtariEnv
from gym.core import Wrapper
from gym import utils, spaces, error
import atari_py
from ale_python_interface import ALEInterface
from game_config import VERSIONS


class Atari(AtariEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='pong', obs_type='ram', frameskip=(2, 5), repeat_action_probability=0.):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, game, obs_type)
        assert obs_type in ('ram', 'image')

        self.game_path = atari_py.get_game_path(game)
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist'%(game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.ale = ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        self._seed()

        (screen_width, screen_height) = self.ale.getScreenDims()
        self._buffer = np.empty((screen_height, screen_width, 3), dtype=np.uint8)

        self._action_set = self.ale.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width,screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=np.zeros(128), high=np.zeros(128)+255)
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _get_image(self):
        return self.ale.getScreenRGB(self._buffer).copy()


def create_env(game='pong', version='v0', act_rep=1):
    params = VERSIONS[version]
    env = Atari(game, **params)
    return AtariUCT(env, act_rep)


def get_hashable_state(s):
    s.flags.writeable = False
    return s.ravel().data


class AtariUCT(Wrapper):
    def __init__(self, env, act_rep=1):
        super(AtariUCT, self).__init__(env)
        assert act_rep >= 1
        self.act_rep = int(act_rep)

    def _reset(self):
        observation = self.env.reset()
        state = get_hashable_state(observation)
        return state

    def _step(self, action):
        cum_reward = 0
        for _ in xrange(self.act_rep):
            obs, reward, terminal, info = self.env.step(action)
            cum_reward += reward
            if terminal:
                break
        return get_hashable_state(obs), cum_reward, terminal, info


    def clone_state(self):
        #return self.env.ale.cloneSystemState()
        return self.env.ale.cloneState()

    def restore_state(self, game_state):
        # self.env.ale.restoreSystemState(game_state)
        self.env.ale.restoreState(game_state)
