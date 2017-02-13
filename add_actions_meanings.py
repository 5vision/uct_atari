import os
import cPickle
from envs import create_env

env = create_env('skiing')

dirs = [
    '/media/storage2/uct_data/skiing_20170115-061728',
    '/media/storage2/uct_data/skiing1',
    '/media/storage2/uct_data/skiing2',
    '/media/storage2/uct_data/skiing_20170118-054226',
    '/media/storage2/uct_data/skiing_20170117-201459',
    '/media/storage2/uct_data/skiing_20170120-232547',
    '/media/storage2/uct_data/skiing_20170121-222543',
]

for d in dirs:
    for fname in os.listdir(d):

        if fname[-4:] != '.pkl':
            continue

        print 'reading file', fname
        with open(os.path.join(d, fname), 'rb') as f:
            run = cPickle.load(f)

        if 'action_meanings' not in run or True:
            run['action_meanings'] = env.env.get_action_meanings()
            with open(os.path.join(d, fname), 'wb') as f:
                cPickle.dump(run, f, -1)

env.close()
