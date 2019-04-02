import gym

from baselines import deepq
from baselines import logger
import time

from gym_unity.envs.unity_env import UnityEnv

import subprocess as sp
import os

env = UnityEnv("../unity_envs/kais_banana", 0, use_visual=True, uint8_visual=True, flatten_branched=True)

act = deepq.learn(env, network='cnn', total_timesteps=0, load_path="logs_backup/model")#"unity_model.pkl")

#Visualizing
#TODO Maybe slow down the simulation by inserting some delays here.
while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(act(obs[None])[0])
        episode_rew += rew
        time.sleep(0.05)
    print("Episode reward", episode_rew)
