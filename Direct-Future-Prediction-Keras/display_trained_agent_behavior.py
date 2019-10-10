#!/usr/bin/env python
from __future__ import print_function

import skimage
import skvideo.io
import numpy as np

from gym_unity.envs import UnityEnv

from keras import backend as K

import itertools as it
from time import sleep
import tensorflow as tf

from networks import Networks

from dfp import DFPAgent

from Utilities import convert_action_id_to_name
import gym
from gym import wrappers
from time import time
#from gym_recording.wrappers import TraceRecordingWrapper
import scipy.misc
import pylab as plt
import pickle
import neat

import matplotlib.ticker as tick
from scipy.interpolate import RegularGridInterpolator

BATTERY_REFILL_AMOUNT = 100
BATTERY_CAPACITY = 100
PENALTY_FOR_PICKING = 100
NUM_TIMESTEPS = 300

def preprocessImg(img, size):
    img = skimage.color.rgb2gray(img)

    return img

def mask_unused_gpus(leave_unmasked=1):
  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

  try:
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]
    print("Available gpus are: ", available_gpus)
    if len(available_gpus) < leave_unmasked: raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus[:leave_unmasked]))
  except Exception as e:
    print('"nvidia-smi" is probably not installed. GPUs are not masked', e)


      #TODO: Consider loading experimental parameters, such as battery capacity, from file.


##PARAMETERS FOR STORED FIGURES
colors = ['blue', 'orange', 'green']
plot_upper_x = 0
plot_x_width = 100
plot_upper_y = 0
plot_y_width = 50

bar_width = 15

def convert_plotted_values_y_axis(plotted_objective_values):
    # Converts the raw values we want to plot to fit inside the image frame.
    converted_values = (1 - plotted_objective_values) + 1  # Since the y-axis in images is "upside down"
    # Values now range fro 0 to 2, where 2 are those that were previously -1, and 0 are those that were previously 1.

    # Stretching out the y-range.
    converted_values = converted_values * plot_y_width
    return converted_values


def convert_plotted_values_x_axis(plotted_objective_values):
    # Stretching out the y-range.
    #First making 0 the minimum, then stretching out.
    converted_values = (1+plotted_objective_values) * plot_x_width
    return converted_values


def regrid(data, out_x, out_y):
    #Upscaling image.
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))

    return interpolating_function((xv, yv))



def superimpose_meas_and_objs_on_image(store_to_image, image, meas, objs):
    #Consider first up-scaling the image.
    meas_names = ["Batt_Level", "Num_Poisons", "Num_Foods"]
    objs_names = ["Battery", "Poison", "Food"]

    fig, ax = plt.subplots()
    ax.imshow(image)
    y_pos = [25, 55, 85] #TODO Generalize
    objcounter = 0
    for obj in objs_names:
        ax.barh(y_pos[objcounter], convert_plotted_values_x_axis(objs[objcounter]), bar_width,
                align='center', color=colors[objcounter],
                ecolor='black')
        objcounter +=1

    ax.text(300,390,str(meas[0]), fontsize=40, color='blue')
    ax.legend(objs_names, loc='upper right', fontsize=15)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(tick.NullLocator())
    plt.gca().yaxis.set_major_locator(tick.NullLocator())
    plt.savefig(store_to_image, bbox_inches="tight", pad_inches=0)
    plt.close(fig)



def evaluate_a_goal_vector(goal_vector, env, dfp_network, num_timesteps = 300, display = True, battery_capacity = 100, timesteps = [1,2,4,8,16,32], goal_producing_network = None, stop_when_batt_empty = True, battery_refill_amount = 100,
                           penalty_for_picking=0, record_meas_and_objs=False):
    #Runs one "game" with a number of timesteps, displaying behavior or returning scores.
    #goal_vector is the goal for a single timestep (e.g. [food, poison, battery]), and automatically repeated for all timesteps.
    #dfp_network is a network initialized with the weights from the trained DFP network
    #If goal-producing network is None, we use the goal_vector. Otherwise, we use the one produced by the network, which is adaptive.
    #Penalty for picking: Simulates time taken to charge, by ticking ahead a number of timesteps
    print("Resetting env")
    initial_observation = env.reset()
    battery=battery_capacity

    prev_battery = battery
    s_t = initial_observation
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    # Number of food pickup as measurement
    food = 0
    # Number of poison pickup as measurement
    poison = 0
    # Initial normalized measurements.
    #KOE: Not sure if I need to normalize...
    #KOE: Original paper normalized by stddev of the value under random exploration.
    m_t = np.array([battery/10.0, poison, food])

    # Goal
    goal = np.array(goal_vector * len(timesteps))
    inference_goal = goal
    battery_picks = 0

    if record_meas_and_objs:
        # Write Rolling Statistics to file
        with open(record_meas_and_objs, "a+") as meas_obs_file:
            meas_obs_file.write("m_batt m_pois m_food o_batt o_pois o_food \n")

    if goal_producing_network:
        all_goal_outputs = []

    t = 0
    charge_count = 0
    dont_charge_count = 0
    while t < num_timesteps:
        t+=1
        if goal_producing_network:
            current_goal_vec = goal_producing_network.activate(m_t)
            all_goal_outputs.append(current_goal_vec)
            goal = np.array(current_goal_vec * len(timesteps))
            inference_goal = goal
        # Epsilon Greedy


        #print("Measurements are: ", m_t)
        action_idx  = dfp_network.get_action(s_t, m_t, goal, inference_goal) #KOE: This is the forward pass through the NN.
        if(",charge" in convert_action_id_to_name(action_idx)):
            charge_count += 1
        else:
            dont_charge_count += 1
        observation, reward, done, info = env.step(action_idx)
        if reward!=0:
            print("Got reward: ", reward)
        battery -= 1  # Always reducing by 1
        if battery == 0 and stop_when_batt_empty:
            done = True


        if (done):
            print("Game done at timestep ", t)
            print ("Episode Finish ")
            print("WARNING: Episode finished at timestep ", t, " of ", num_timesteps)
            break
        else:
            x_t1 = observation

        s_t1 = np.expand_dims(x_t1, axis=0)  # 1x64x64x4


        #Reward value is sometimes slightly inaccurate in conversion from UnityML to Python, Therefore using these ranges.
        if (reward>-1.05 and reward < -0.95): # Pick up Poison
            poison += 1
            print("Picked up. Current poison is ", poison)
        if (reward > 0.95 and reward <1.05): # Pick up food
            food += 1
            print("Picked up. Current food is ", food)
        if (reward>0.05 and reward <0.15):
            print("Picked a battery")
            battery+=battery_refill_amount
            if battery > battery_capacity:
                battery = battery_capacity
            battery_picks+=1
            print("Touched a battery. Picks: ", battery_picks)
            if penalty_for_picking:
                t+=penalty_for_picking
                print("Skipping ahead ", penalty_for_picking, " timesteps.")
                if t> num_timesteps:
                    t=num_timesteps
                print("Battery after pick was ", battery)
        if reward>-0.15 and reward <-0.05:
            print("Touched a battery without picking")

        #print("Timestep ", t, ". battery: ", battery)

        #KOETODO: Think about normalization.
        m_t = np.array([battery/10.0, poison, food]) # Measurement after transition

        if record_meas_and_objs:
            # Write Rolling Statistics to file
            with open(record_meas_and_objs, "a+") as meas_obs_file:
                meas_obs_file.write(str(m_t[0]) + " ")
                meas_obs_file.write(str(m_t[1]) + " ")
                meas_obs_file.write(str(m_t[2]) + ' ')
                meas_obs_file.write(str(current_goal_vec[0]) + ' ')
                meas_obs_file.write(str(current_goal_vec[1]) + ' ')
                meas_obs_file.write(str(current_goal_vec[2]) + '\n')
        s_t = s_t1
        #print("Timestep ", t)
        #if display:
        #    sleep(0.1) #To get real-time (not too fast) video
        if display:
            observation_array = np.array(observation)
            observation_array = regrid(observation_array, 400, 400)
            print("Reshaped observation: ", observation_array.shape)
            superimpose_meas_and_objs_on_image('frames_for_video/'+str(t).zfill(3)+".png", observation_array, [battery, poison, food], current_goal_vec)


    print("Number of charge outputs: ", charge_count)
    print("Number of dont charge outputs: ", dont_charge_count)
    if goal_producing_network:
        return {"battery": battery, "poison": poison, "food": food, "battery_picks": battery_picks, "goal_history":all_goal_outputs, "num_timesteps_elapsed":t}
    else:
        return {"battery": battery, "poison": poison, "food": food, "battery_picks": battery_picks, "goal_history":None, "num_timesteps_elapsed":t}

if __name__ == "__main__":
    mask_unused_gpus()

    evo_winner = "oct9-evolving-agent-with-charge-output_1/winner_network.pickle"

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    #Setting up the env
    #TODO Worker_id can be changed to run in parallell
    #Flatten_branched gives us a onehot encoding of all 54 action combinations.
    print("Opening unity env")
    env = UnityEnv("../unity_envs/kais_banana_with_explicit_charge_decision_red_battery_900_timesteps", worker_id=31, use_visual=True, flatten_branched=True, seed=5) #TODO: Add seed input if I want.
    #env = TraceRecordingWrapper(env)
    #env.recording = "test_openai_recording/"
    #env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True) KOE: I think this wrapper may not be working with Unity ML agents.

    print("env setup done")
    measurement_size = 3
    timesteps = [1, 2, 4, 8, 16, 32]
    goal_size = measurement_size * len(timesteps)
    img_rows, img_cols = 84, 84
    img_channels = 3  # KOE: If I want to change this, I have to also edit the frame stacking when forming s_t
    state_size = (img_rows, img_cols, img_channels)


    action_size = env.action_space.n
    print("Env has ", action_size, " actions.")

    dfp_net = DFPAgent(state_size, measurement_size, action_size, timesteps)
    dfp_net.model = Networks.dfp_network(state_size, measurement_size, goal_size, action_size, len(timesteps), dfp_net.learning_rate)

    loaded_model = "oct8_explicitly_controlled_battery_charging_agnostic/model/dfp.h5"
    dfp_net.load_model(loaded_model)
    dfp_net.epsilon = dfp_net.final_epsilon

    with open(evo_winner, 'rb') as pickle_file:
        winner_genome = pickle.load(pickle_file)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config")

    goal_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)


    evaluate_a_goal_vector([1,-1,1], env, dfp_net, display=True, num_timesteps=NUM_TIMESTEPS, battery_refill_amount=BATTERY_REFILL_AMOUNT, battery_capacity=BATTERY_CAPACITY, penalty_for_picking=PENALTY_FOR_PICKING,
                           goal_producing_network=goal_net)

    env.close()

#1: 117 / 76
#-1: 148 / 152
