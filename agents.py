import numpy as np
from keras.models import load_model
from lasagne_model import build_model
from utils import make_state
import cv2
from collections import deque
from keras.objectives import categorical_crossentropy
import cPickle


class BaseAgent(object):

    def __init__(self, n_actions, seed=None, not_sample=False):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)
        self.not_sample = not_sample

    def get_probs(self, frame):
        raise NotImplementedError

    def choose_action(self, frame, sample=True):
        probs = self.get_probs(frame)

        if self.not_sample:
            sample = False

        if sample:
            return self.rng.multinomial(1, probs-np.finfo(np.float32).epsneg).argmax()
        else:
            return probs.argmax()

    def seed(self, seed):
        self.rng.seed(seed)

    def reset(self):
        pass


class RandomAgent(BaseAgent):
    """ Agent that chooses action randomly """
    def __init__(self, n_actions):
        super(RandomAgent, self).__init__(n_actions)

    def get_probs(self, frame):
        return np.asarray([1./self.n_actions]*self.n_actions)

    def choose_action(self, frame, sample=True):
        return self.rng.randint(self.n_actions)


class KerasAgent(BaseAgent):
    """Agent that uses keras model to predict action"""
    def __init__(self, model_path, flip_map=None, gray_state=True, **kwargs):
        # load model
        model = load_model(
            model_path,
            custom_objects={'loss_fn': categorical_crossentropy}
        )
        if flip_map is not None:
            assert model.output_shape[1] == len(flip_map)

        super(KerasAgent, self).__init__(n_actions=model.output_shape[1], **kwargs)

        self.gray_state = gray_state

        if len(model.input_shape) == 5:
            self.n_frames = model.input_shape[2]
            self.rnn = True
        else:
            self.n_frames = model.input_shape[1]
            self.rnn = False

        if not gray_state:
            self.n_frames /= 3
        self.height, self.width = model.input_shape[2:]
        self.model = model
        self.flip_map = flip_map
        self.reset()

    def reset(self):
        self.model.reset_states()
        self.buf = deque(maxlen=self.n_frames)

    def get_probs(self, frame):
        if self.flip_map:
            frame = cv2.flip(frame, 1)
        s = make_state(frame, self.buf, self.height, self.width, make_gray=self.gray_state)

        if self.rnn:
            probs = self.model.predict(np.asarray([s]))[0][0]
        else:
            probs = self.model.predict(s)[0]

        if self.flip_map:
            return probs[self.flip_map]
        return probs


class LasagneAgent(BaseAgent):
    def __init__(self, model_path, observation_shape=(4, 84, 84), flip_map=None, **kwargs):

        # read weights
        with open(model_path, 'rb') as f:
            weights = cPickle.load(f)

        # build model and set weights
        n_actions = weights[-1].shape[-1]
        _, prob_fn, _, params = build_model(observation_shape, n_actions)
        for p, w in zip(params, weights):
            p.set_value(w)

        super(LasagneAgent, self).__init__(n_actions=n_actions, **kwargs)

        self.n_frames = observation_shape[0]
        self.height, self.width = observation_shape[-2:]
        self.flip_map = flip_map
        self.probs = prob_fn
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.n_frames)

    def get_probs(self, frame):
        if self.flip_map:
            frame = cv2.flip(frame, 1)
        s = make_state(frame, self.buf, self.height, self.width, make_gray=True, average='cv2')
        probs = self.probs(s)[0]

        if self.flip_map:
            return probs[self.flip_map]
        return probs


class EnsebleAgent(BaseAgent):
    """Agent that sample action randomly from other provided agents"""
    def __init__(self, agents, ensemble_mode='mean', weights=None,
                 **kwargs):
        if weights is None:
            weights = [1./len(agents)]*len(agents)
        assert sum(weights) == 1
        assert ensemble_mode in ('mean', 'sample_soft', 'sample_hard')

        super(EnsebleAgent, self).__init__(agents[0].n_actions, **kwargs)
        self.agents = agents
        self.weights = np.asarray(weights)
        self.ensemble_mode = ensemble_mode

    def get_probs(self, frame):
        probs_all = [agent.get_probs(frame) for agent in self.agents]
        return self.weights.dot(probs_all)

    def choose_action(self, frame, sample=True):
        # first get probs for all agents and then sample/argmax
        if self.ensemble_mode == 'mean':
            probs = self.get_probs(frame)

            if self.not_sample:
                sample = False

            if sample:
                return self.rng.multinomial(1, probs - np.finfo(np.float32).epsneg).argmax()
            else:
                return probs.argmax()
        # first get action for all agents then sample
        else:
            actions = [agent.choose_action(frame, sample) for agent in self.agents]
            if self.ensemble_mode == 'sample_soft':
                a_idx = self.rng.multinomial(len(actions), self.weights).argmax()
                return actions[a_idx]
            elif self.ensemble_mode == 'sample_hard':
                return self.rng.choice(actions)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def seed(self, seed):
        self.rng.seed(seed)
        for agent in self.agents:
            agent.seed(seed)
