##Stores the objective values for all objectives throughout a full run. Useful to see how it changes as we go.
#TODO :Should also store measurements!

from display_trained_agent_behavior import evaluate_a_goal_vector, mask_unused_gpus

import tensorflow as tf
from keras import backend as K
from gym_unity.envs import UnityEnv
from dfp import DFPAgent
from networks import Networks
import numpy as np
import neat
import os
import pickle
from collections import Counter
import pandas as pd
import sys
import Utilities


BATTERY_REFILL_AMOUNT = 100
BATTERY_CAPACITY = 100
PENALTY_FOR_PICKING = 100
NUM_TIMESTEPS = 900


def make_measurements_and_objectives_dataframe(store_to_filepath, dfp_net, goal_net, env):

    evaluate_a_goal_vector([0,0,0], env, dfp_net, goal_producing_network=goal_net, display=False,
                                         battery_refill_amount=BATTERY_REFILL_AMOUNT,
                                         battery_capacity=BATTERY_CAPACITY, penalty_for_picking=PENALTY_FOR_PICKING, num_timesteps=NUM_TIMESTEPS,
                                         record_meas_and_objs=store_to_filepath)




if __name__ == "__main__":
    ##Some parameters

    winner_filename = sys.argv[1] #Pickled winner indiv
    store_to_folder = sys.argv[2]


    if(len(sys.argv) > 2):
        seed = int(sys.argv[3])
    else:
        seed = 1
    Utilities.store_seed_to_folder(seed, store_to_folder, "StoreObjectiveHistory")

    with open(winner_filename, 'rb') as pickle_file:
        winner_genome = pickle.load(pickle_file)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config")


    env = UnityEnv("../unity_envs/kais_banana_with_explicit_charge_decision_red_battery_900_timesteps", worker_id=29,
                   use_visual=True,
                   flatten_branched=True, seed=seed)


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

    loaded_model = "oct7_explicitly_controlled_battery_charging_agnostic_battery_limit_on/model/dfp.h5" #Learned dfp with infinite battery. june27_battery_balanced1_agnostic_battery_limit_on/model/dfp.h5" #KOE: This was trained with real battery consequences. It seemed to learn battery seeking behavior better.
    dfp_net.load_model(loaded_model)
    dfp_net.epsilon = dfp_net.final_epsilon


    make_measurements_and_objectives_dataframe(store_to_folder+"/measurements_and_objectives_dataframe.csv", dfp_net, goal_net, env)



