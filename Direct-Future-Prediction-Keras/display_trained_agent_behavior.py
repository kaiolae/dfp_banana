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

BATTERY_REFILL_AMOUNT = 300
BATTERY_CAPACITY = 300
PENALTY_FOR_PICKING = 100
NUM_TIMESTEPS = 900

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
    recorded_frames = []
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
        if display:
            recorded_frames.append(observation)
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
        if display:
            sleep(0.1) #To get real-time (not too fast) video
    if display:
        recorded_frames=np.array(recorded_frames)
        recorded_frames=recorded_frames.astype(np.uint8)
        skvideo.io.vwrite("test_video.mp4", recorded_frames)

    print("Number of charge outputs: ", charge_count)
    print("Number of dont charge outputs: ", dont_charge_count)
    if goal_producing_network:
        return {"battery": battery, "poison": poison, "food": food, "battery_picks": battery_picks, "goal_history":all_goal_outputs, "num_timesteps_elapsed":t}
    else:
        return {"battery": battery, "poison": poison, "food": food, "battery_picks": battery_picks, "goal_history":None, "num_timesteps_elapsed":t}

if __name__ == "__main__":
    mask_unused_gpus()

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    #Setting up the env
    #TODO Worker_id can be changed to run in parallell
    #Flatten_branched gives us a onehot encoding of all 54 action combinations.
    print("Opening unity env")
    env = UnityEnv("../unity_envs/kais_banana_with_explicit_charge_decision_red_battery_900_timesteps", worker_id=3, use_visual=True, flatten_branched=True, seed=3) #TODO: Add seed input if I want.


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


    evaluate_a_goal_vector([-1,-1,1], env, dfp_net, display=False, num_timesteps=NUM_TIMESTEPS, battery_refill_amount=BATTERY_REFILL_AMOUNT, battery_capacity=BATTERY_CAPACITY, penalty_for_picking=PENALTY_FOR_PICKING)

    env.close()

#1: 117 / 76
#-1: 148 / 152
