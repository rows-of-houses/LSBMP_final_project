import argparse
import pickle
import os
import cv2
import numpy as np

from environment import Environment
from random_env import get_random_env
from utils import set_random_seed


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
set_random_seed(args.random_seed)

for i in range(args.num_envs):
    while True:  # Guarantee that we have the correct length
        env, _, _ = get_random_env(args.scale)
        states, actions = env.generate_trajectory(args.trajectory_length, max_step=args.scale * args.step)
        if len(actions) == args.trajectory_length:
            break
        print(f"Failed to build trajectory for index {i}, retrying")
    scaled_actions = []
    for action in actions:
        scaled_actions.append((action[0] / args.scale, action[1] / args.scale))
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
   
