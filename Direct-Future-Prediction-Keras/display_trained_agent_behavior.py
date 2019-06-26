#!/usr/bin/env python
from __future__ import print_function

import skimage
import skvideo.io
import numpy as np

from gym_unity.envs.unity_env import UnityEnv

from keras import backend as K

import itertools as it
from time import sleep
import tensorflow as tf

from networks import Networks

from dfp import DFPAgent

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
    env = UnityEnv("../unity_envs/kais_banana_with_battery_consumable_balanced", worker_id=22, use_visual=True, flatten_branched=True)

    print("Resetting env")
    initial_observation = env.reset()

    battery = 100 # [Health]
    prev_battery = battery
    action_size = env.action_space.n
    print("Env has ", action_size, " actions.")
    measurement_size = 3 # [Battery, posion, food]
    timesteps = [1,2,4,8,16,32]#[1,2,4,8,16,32]
    goal_size = measurement_size * len(timesteps)
    img_rows , img_cols = 84, 84

    img_channels = 3 # KOE: If I want to change this, I have to also edit the frame stacking when forming s_t

    state_size = (img_rows, img_cols, img_channels)
    agent = DFPAgent(state_size, measurement_size, action_size, timesteps)
    agent.model = Networks.dfp_network(state_size, measurement_size, goal_size, action_size, len(timesteps), agent.learning_rate)



    #x_t = game_state.screen_buffer # 480 x 640
    #x_t = preprocessImg(initial_observation, size=(img_rows, img_cols))

    #np.save("input_output_examples/initial_obs.npy", initial_observation)
    #np.save("input_output_examples/preprocessed_obs.npy", x_t)
    #KOE: Preprocessing to get black and white.

    #KOE: Not sure what is going on here. 4 images in a row?
    #s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    #s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    s_t = initial_observation
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    #np.save("input_output_examples/stacked_obs.npy", s_t)
    # Number of food pickup as measurement
    food = 0

    # Number of poison pickup as measurement
    poison = 0

    # Initial normalized measurements.
    #KOE: Not sure if I need to normalize...
    #KOE: Original paper normalized by stddev of the value under random exploration.
    m_t = np.array([battery/100.0, poison, food])

    # Goal
    # KOE: Battery, poison, food. No way to affect battery so far except standing still. Maybe that will happen?
    #KOETODO: Rewarding both food and poison as an initial test. If that works, but the other not,
    #maybe the color vision is a problem?
    #TODO: How to sync loaded weights with agent goal?
    goal = np.array([1, 0.1, 0.1] * len(timesteps))
    #for both: goal = np.array([0.0, 1.0, -1.0] * len(timesteps))
    #TODO Make input argument
    loaded_model = "june26_battery_balanced_2_agnostic/model/dfp.h5" #To see things working, swich to kais banana 2 with june13_goal_agnostic_1 as input.
    agent.load_model(loaded_model)
    agent.epsilon = agent.final_epsilon #After training, we want to visualize without randomness.
    inference_goal = goal
    done = False

    num_timesteps = 300

    recorded_frames = []


    battery_picks = 0
    for t in range(num_timesteps):

        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx  = agent.get_action(s_t, m_t, goal, inference_goal) #KOE: This is the forward pass through the NN.
        observation, reward, done, info = env.step(action_idx)
        recorded_frames.append(observation)
        if reward!=0:
            print("Got reward: ", reward)
            print("Taking action ", action_idx)
        meas = info['brain_info'].vector_observations
        if (done):
            print("Game done at timestep ", t)
            print ("Episode Finish ")
            battery = 100
            x_t1 = env.reset()
        else:
            x_t1 = observation
            battery = battery

        #Img to black/white
        #x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
        #x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        #s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #KOE: What is this? Some sequence of images?
        s_t1 = np.expand_dims(x_t1, axis=0)  # 1x64x64x4

        if (reward==-1): # Pick up Poison
            poison += 1
            print("Picked up. Current poison is ", poison)
        if (reward==1): # Pick up food
            food += 1
            print("Picked up. Current food is ", food)
        if reward != -1 and reward != 1 and reward!=0:
            battery+=100
            battery_picks+=1
            print("Touched a battery. Picks: ", battery_picks)
        # Update the cache
        prev_battery = battery

        #KOETODO: Think about normalization.
        m_t = np.array([battery/100.0, poison, food]) # Measurement after transition
        s_t = s_t1
        sleep(0.1) #To get real-time (not too fast) video
    env.close()

    recorded_frames=np.array(recorded_frames)
    recorded_frames=recorded_frames.astype(np.uint8)
    skvideo.io.vwrite("test_video.mp4", recorded_frames)


#KOE: Made it to the end. Now test running, print out, debug, etc.
