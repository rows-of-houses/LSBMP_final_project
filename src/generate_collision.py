import argparse
import pickle
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from environment import Environment
from random_env import get_random_env
from model import AutoEncoder_Dynamics
from utils import set_random_seed


def resize(img, scale):
    return cv2.resize(img, dsize=(img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_AREA)


parser = argparse.ArgumentParser()

parser.add_argument('-n', "--num_envs", type=int, default=25000)
parser.add_argument('-l', "--num_pairs", type=int, default=10)
parser.add_argument('-s', "--step", type=int, default=8)
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("model_path")
parser.add_argument('dirname')

args = parser.parse_args()
set_random_seed(args.random_seed)
os.makedirs(args.dirname)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc_dyn_net = AutoEncoder_Dynamics()
enc_dyn_net.load_state_dict(torch.load(args.model_path, weights_only=True))
enc_dyn_net.eval()
enc_dyn_net.to(device)

labels = []
latents1 = []
latents2 = []
for i in tqdm(range(args.num_envs)):
    env, _, _ = get_random_env(args.scale)
    states1, states2 = [], []
    labels_inner = []
    for j in range(args.num_pairs):
        state1, state2, label = env.generate_collision(args.scale * args.step)
        states1.append(resize(env.plot_enviroment(state1), args.scale))
        states2.append(resize(env.plot_enviroment(state2), args.scale))
        labels_inner.append(label)
    states1 = torch.tensor(np.stack(states1), dtype=torch.float32).to(device)
    states2 = torch.tensor(np.stack(states2), dtype=torch.float32).to(device)
    with torch.no_grad():
        z_full = enc_dyn_net.encode(states1, states2)[1].cpu().numpy()
    z1, z2 = z_full[:states1.shape[0]], z_full[states1.shape[0]:]
    latents1.append(z1)
    latents2.append(z2)
    labels.append(labels_inner)
    with open(os.path.join(args.dirname, f"empty_{i}.bin"), 'wb') as fp:
        pickle.dump(resize(env._env, args.scale), fp)

mean, std = np.stack(latents1 + latents2).mean(), np.stack(latents1 + latents2).std()
latents1 = (np.concatenate(latents1, axis=0) - mean) / std
latents2 = (np.concatenate(latents2, axis=0) - mean) / std
print(f"latent mean: {mean}, std: {std}")

with open(os.path.join(args.dirname, "latents1.bin"), 'wb') as fp:
    pickle.dump(latents1, fp)
with open(os.path.join(args.dirname, "latents2.bin"), 'wb') as fp:
    pickle.dump(latents1, fp)
with open(os.path.join(args.dirname, "labels.bin"), 'wb') as fp:
    pickle.dump(labels, fp)
with open(os.path.join(args.dirname, "stats.bin"), 'wb') as fp:
    pickle.dump({"mean": mean, "std": std}, fp)

