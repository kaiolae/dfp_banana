##Compares performance between 1) Evolved Goal Vectors (loaded from pickled individual),
##2) Static Goal Vector(s), 3) Any hardcoded rules.

from display_trained_agent_behavior import evaluate_a_goal_vector, mask_unused_gpus

import tensorflow as tf
from keras import backend as K
from gym_unity.envs.unity_env import UnityEnv
from dfp import DFPAgent
from networks import Networks
import numpy as np
import neat
import os
import pickle
from collections import Counter
import pandas as pd
import sys

env = UnityEnv("../unity_envs/kais_banana_with_battery_consumable_balanced", worker_id=9, use_visual=True,
               flatten_branched=True)
BATTERY_REFILL_AMOUNT = 100
BATTERY_CAPACITY = 100

NUM_TESTS_PER_SETUP = 25

HARDCODED_PREFERENCE_VECTOR = [1,0,1]

def make_result_summary_dataframe(store_to_filepath, dfp_net, goal_vector=[0,0,0], goal_net=None):
    #When using evolved goals, fill in goal-net. Otherwise, fill in goal-vector.

    results_summary = []
    for i in range(NUM_TESTS_PER_SETUP):
        eval_result = evaluate_a_goal_vector(goal_vector, env, dfp_net, goal_producing_network=goal_net, display=False,
                                             battery_refill_amount=BATTERY_REFILL_AMOUNT,
                                             battery_capacity=BATTERY_CAPACITY)
        results_summary.append(eval_result)
        print("Test ", i, " done")

    result_frame = pd.DataFrame(results_summary)

    result_frame.to_csv(store_to_filepath, index=None, header=True)



if __name__ == "__main__":
    ##Some parameters

    winner_filename = sys.argv[1] #Pickled winner indiv
    store_to_folder = sys.argv[2]

    with open(winner_filename, 'rb') as pickle_file:
        winner_genome = pickle.load(pickle_file)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config")

    goal_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    #Configuring and loading the trained DFP-net.
    measurement_size = 3
    timesteps = [1, 2, 4, 8, 16, 32]
    goal_size = measurement_size * len(timesteps)
    img_rows, img_cols = 84, 84
    img_channels = 3  # KOE: If I want to change this, I have to also edit the frame stacking when forming s_t
    state_size = (img_rows, img_cols, img_channels)


    action_size = env.action_space.n
    print("Env has ", action_size, " actions.")

    global dfp_net
    dfp_net = DFPAgent(state_size, measurement_size, action_size, timesteps)
    dfp_net.model = Networks.dfp_network(state_size, measurement_size, goal_size, action_size, len(timesteps), dfp_net.learning_rate)

    loaded_model = "june27_battery_balanced1_agnostic_battery_limit_on/model/dfp.h5" #KOE: This was trained with real battery consequences. It seemed to learn battery seeking behavior better.
    dfp_net.load_model(loaded_model)
    dfp_net.epsilon = dfp_net.final_epsilon

    make_result_summary_dataframe(store_to_folder+"/evolved_result_summary.csv", dfp_net, goal_net=goal_net)
    make_result_summary_dataframe(store_to_folder+"/hardcoded_result_summary.csv",dfp_net, goal_vector=[1,0,1])



