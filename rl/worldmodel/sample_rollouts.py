import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from skimage import img_as_ubyte
import imageio
import gym
import numpy as np
from tqdm import tqdm
from random import choice, random, randint
import argparse
import torch

from envs import Assault

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--iters', type=int, default=10000, required=False) 
    parser.add_argument('--show', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--steps', type=int, default=1000, required=False)
    return parser 

def get_action_randomly(env, steps, obs):
    action = env.sample_action()
    return action

def sample_rollouts(env, iters, steps):
    os.makedirs('generated/assault/rollouts', exist_ok=True) 
    os.system('rm -rf generated/assault/rollouts/*')

    pbar = tqdm(range(iters))
    cnt_iter = 0
    for episode in pbar:
        pbar.set_description('Episode [{}/{}]'.format(cnt_iter + 1, iters))
        if cnt_iter > iters:
            break

        obs = env.reset()
        for t in range(steps):
            action = get_action_randomly(env, t, obs)
            obs, reward, done, _, _ = env.step(action)
            if args.show:
                env.render()
            if done:
                cnt_iter += t
                episode += t
                pbar.update(t)
                break

            i = ('0000' + str(t))[-4:]
            imageio.imwrite(f'generated/assault/rollouts/{episode}_{i}.jpg', img_as_ubyte(obs[0]))


if __name__ == '__main__':
    args = create_parser().parse_args()
    env = Assault(23)
    device = torch.device(args.device)

    sample_rollouts(env, args.iters, args.steps)
    env.close()
