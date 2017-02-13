import threading
import numpy as np
import random
import cv2
import os
import tarfile
import cPickle
from collections import deque
from keras.utils.np_utils import to_categorical


def make_state(frame, buf, height=84, width=84,
               downsample=None, make_gray=True,
               average='mean',):
    frame = resize(frame, height, width, downsample)
    if make_gray:
        frame = rgb2gray(frame, average)
    buf.append(frame)
    while len(buf) < buf.maxlen:
        buf.append(frame)

    state = np.array(buf)
    if not make_gray:
        state = make_color_state(state)

    return np.expand_dims(state, 0)


def crop_image(img, h=(50, -20), w=(10, -10)):
    return img[slice(*h), slice(*w)]


def resize(frame, height=84, width=84, downsample=None):
    h, w = frame.shape[:2]
    if downsample is not None:
        width = int(w / downsample)
        height = int(h / downsample)
    else:
        downsample = 0.5 * h / height + 0.5 * w / width

    if downsample > 4:
        frame = cv2.resize(frame, (width*2, height*2))

    frame = cv2.resize(frame, (width, height))
    return frame


def rgb2gray(frame, average='mean'):
    if average == 'cv2':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32)
    elif average == 'mean':
        frame = frame.astype(np.float32)
        frame = frame.mean(axis=2)
    else:
        raise NotImplementedError('wrong average type: {}'.format(average))

    frame *= (1.0 / 255.0)
    frame -= 0.5
    return frame


def load_runs(dirs, height=84, width=84, downsample=None, min_score=None):

    def append_run(run):
        if run['reward'] >= min_score:
            frames = [resize(f, height, width, downsample) for f in run['frames']]
            run['frames'] = frames
            runs.append(run)

    if min_score is None:
        min_score = -np.inf

    runs = []
    for d in dirs:

        if os.path.isdir(d):
            for fname in os.listdir(d):

                if fname[-4:] != '.pkl':
                    continue

                f_path = os.path.join(d, fname)
                if os.path.exists(f_path):
                    print 'reading file', f_path
                    with open(f_path, 'rb') as f:
                        run = cPickle.load(f)

                    append_run(run)
                else:
                    print 'deleted file  {}'.format(f_path)
        else:
            tar = tarfile.open(d, "r:gz")

            for fname in tar.getnames():

                if fname[-4:] != '.pkl':
                    continue

                print 'reading file', fname
                run = cPickle.load(tar.extractfile(tar.getmember(fname)))
                append_run(run)

    return runs


def make_color_state(frame):
    frame = np.rollaxis(frame, 3, 1)
    frame = frame.reshape([-1] + list(frame.shape[-2:]))
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    #frame -= 0.5
    return frame


def process_run(run, n_frames=4, flip_map=None, make_gray=True):
    frames = []
    actions = run['actions']
    action_values = run['action_values']
    rgb_frames = run['frames']

    buf = deque(maxlen=n_frames)

    frame0 = rgb_frames[0]
    if flip_map is not None:
        frame0 = cv2.flip(frame0, 1)

    if make_gray:
        frame0 = rgb2gray(frame0)

    # fill buffer with first frame
    for i in xrange(n_frames):
        buf.append(frame0)

    # process all frames
    for frame in rgb_frames:
        if flip_map is not None:
            frame = cv2.flip(frame, 1)

        if make_gray:
            frame = rgb2gray(frame)

        buf.append(frame)

        # stack frames
        frame = np.array(buf)

        # make number of channels equal to 3*n_frames
        if not make_gray:
            frame = make_color_state(frame)

        frames.append(frame)

    if flip_map is not None:
        actions = [flip_map[a] for a in actions]
        action_values = [[av[a] for a in flip_map] for av in action_values]

    data = zip(frames, actions, action_values)

    return data


def augment_image(image):

    # move channel to the last axis
    image = np.rollaxis(image, 0, 3)
    h, w, ch = image.shape[:3]

    # brightness
    brightness = random.uniform(-0.1, 0.1)

    # rotation and scaling
    rot = 1
    scale = 0.01
    Mrot = cv2.getRotationMatrix2D((h / 2, w / 2), random.uniform(-rot, rot), random.uniform(1.0 - scale, 1.0 + scale))

    # affine transform and shifts
    pts1 = np.float32([[0, 0], [w, 0], [w, h]])
    a = 1
    shift = 1
    shiftx = random.randint(-shift, shift)
    shifty = random.randint(-shift, shift)
    pts2 = np.float32([[
        0 + random.randint(-a, a) + shiftx,
        0 + random.randint(-a, a) + shifty
    ], [
        w + random.randint(-a, a) + shiftx,
        0 + random.randint(-a, a) + shifty
    ], [
        w + random.randint(-a, a) + shiftx,
        h + random.randint(-a, a) + shifty
    ]])
    M = cv2.getAffineTransform(pts1, pts2)

    def _augment(image):
        image = np.add(image, brightness)

        augmented = cv2.warpAffine(
            cv2.warpAffine(
                image
                , Mrot, (w, h)
            )
            , M, (w, h)
        )

        if augmented.ndim < 3:
            augmented = np.expand_dims(augmented, 2)

        return augmented

    # make same transform for each channel, splitting image by four channels
    image_lst = [image[..., i:i+4] for i in xrange(0, ch, 4)]
    augmented_lst = map(_augment, image_lst)
    augmented = np.concatenate(augmented_lst, axis=-1)

    # roll channel axis back when returning
    augmented = np.rollaxis(augmented, 2, 0)

    return augmented


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def batch_generator(runs, n_actions, loss, augment=True, n_frames=4,
                    flip_map=None, make_gray=True, weight_runs=True,
                    batch_size=100):
    assert loss in ('cross_entropy', 'policy_loss', 'value_softmax')

    runs = runs[0](runs[1])
    total_frames = sum(len(r['actions']) for r in runs)
    print 'total training examples', total_frames

    max_score = max(r['reward']for r in runs)
    min_score = min(r['reward'] for r in runs)
    delta_score = max_score - min_score

    while 1:
        random.shuffle(runs)

        batch_frames = []
        batch_targets = []
        batch_weights = []
        for run in runs:
            data = process_run(run, n_frames, make_gray=make_gray)
            if flip_map is not None:
                data += process_run(run, n_frames, flip_map, make_gray)
            random.shuffle(data)

            run_score = run['reward']
            if delta_score > 0:
                run_weight = 0.5 + (run_score - min_score) / 2. / delta_score
            else:
                run_weight = 1.

            for d in data:
                frame, action, action_vals = d

                if augment:
                    frame = augment_image(frame)

                batch_frames.append(frame)

                if weight_runs:
                    batch_weights.append(run_weight)

                if loss == 'cross_entropy':
                    batch_targets.append(action)
                else:
                    batch_targets.append(action_vals)

                if len(batch_frames) == batch_size:
                    bf = np.asarray(batch_frames)

                    if loss == 'cross_entropy':
                        bt = to_categorical(batch_targets, n_actions).astype('float32')
                    else:
                        bt = np.asarray(batch_targets).astype('float32')

                    batch = (bf, bt)
                    if weight_runs:
                        bw = np.asarray(batch_weights).astype('float32')
                        batch = (bf, bt, bw)

                    yield batch

                    del batch_frames[:]
                    del batch_targets[:]
                    del batch_weights[:]
