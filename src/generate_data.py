import argparse
import pickle
import os
import cv2
import numpy as np

from environment import Environment
from random_env import get_random_env


def resize(img, scale):
    return cv2.resize(img, dsize=(img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_AREA)


parser = argparse.ArgumentParser()

parser.add_argument('-n', "--num_envs", type=int, default=10000)
parser.add_argument('-l', "--trajectory_length", type=int, default=10)
parser.add_argument('-s', "--step", type=int, default=2)
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument('dirname')

args = parser.parse_args()
np.random.seed(args.random_seed)

for i in range(args.num_envs):
    env, _, _ = get_random_env(args.scale)
    states, actions = env.generate_trajectory(args.trajectory_length, max_step=args.scale * args.step)
    os.makedirs(os.path.join(args.dirname, str(i)))
    with open(os.path.join(args.dirname, str(i), "empty.bin"), 'wb') as fp:
        pickle.dump(resize(env._env, args.scale), fp)
    with open(os.path.join(args.dirname, str(i), "actions.bin"), 'wb') as fp:
        pickle.dump(actions, fp)
    for j, state in enumerate(states):
        with open(os.path.join(args.dirname, str(i), f"state_{j}.bin"), 'wb') as fp:
            img = env.plot_enviroment(state)
            img = resize(img, args.scale)
            pickle.dump(img, fp)
   
