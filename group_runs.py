import os
import cPickle
from os import listdir
from os.path import isdir
import re
import argparse

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--load_dir', type=str, default=".", help="Directory with runs data to group.")
parser.add_argument('--save_dir', type=str, default="", help="Directory where to save grouped data.")
args = parser.parse_args()

dir_path = args.load_dir
save_dir = args.save_dir

def extract_file_name_data(file_name):
    parsed = re.split("[,_.]+", file_name)
    if (not parsed[-1] == 'pkl'):
        return None
    data = {'process':int(parsed[2]), 'run':int(parsed[4]),\
            'steps':int(parsed[6]), 'exp':parsed[-1]}
    return data

def get_all_files_pathes_and_names(path):
    pathes = [path + "/" + f for f in listdir(dir_path) if not isdir(f)]
    names = [f for f in listdir(dir_path) if not isdir(f)]
    return pathes, names

pathes, names = get_all_files_pathes_and_names(dir_path)
extracted_names = [extract_file_name_data(f_name) for f_name in names]
#files can be unsorted, or some other mistakes
max_process_id = max([file_name['process'] for file_name in extracted_names if file_name])
max_run_id = max([file_name['run'] for file_name in extracted_names if file_name])
group_by = [[ [] for i in range(max_run_id + 1)] for j in range(max_process_id + 1)]

for f_path, f_name, file_name_data in zip(pathes, names, extracted_names):
    if os.path.exists(f_path) and file_name_data:
        with open(f_path, 'rb') as f:
            run = cPickle.load(f)
            group_by[file_name_data['process']][file_name_data['run']].append((run, file_name_data['steps']))

def process_data(ordered_data, process, n_run):
    frames = []
    actions = []
    rewards = []
    action_values = []
    action_visits = []
    action_meanings = ordered_data[0][0]['action_meanings']
    reward = ordered_data[-1][0]['reward']
    for i in ordered_data:
        frames.extend(i[0]['frames'])
        actions.extend(i[0]['actions'])
        rewards.extend(i[0]['rewards'])
        action_values.extend(i[0]['action_values'])
        action_visits.extend(i[0]['action_visits'])
    res = {
        'frames': frames,
        'actions': actions,
        'reward': reward,
        'action_visits': action_visits,
        'action_values': action_values,
        'rewards': rewards,
        'action_meanings': action_meanings,
    }
    if not os.path.exists(save_dir) and not save_dir == "":
        os.makedirs(save_dir)
    fname = os.path.join(save_dir, 'run_process_{}_run_{}.pkl'.format(process, n_run))
    with open(fname, 'wb') as f:
        cPickle.dump(res, f, -1)

for i, row in enumerate(group_by):
    for j, group in enumerate(row):
        orderd_to_concat = []
        if len(group) > 0:
            orderd_to_concat=sorted(group, key=lambda x: x[1])
            process_data(orderd_to_concat, i, j)
