#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

from gym_unity.envs.unity_env import UnityEnv

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K

import itertools as it
from time import sleep
import tensorflow as tf

from networks import Networks

#KOE: making my edits directly here. The original one is still in dfp_original.py
def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)
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


class DFPAgent:

    def __init__(self, state_size, measurement_size, action_size, timesteps):

        # get size of state, measurement, action, and timestep
        self.state_size = state_size
        self.measurement_size = measurement_size
        self.action_size = action_size
        self.timesteps = timesteps

        # these is hyper parameters for the DFP
        self.gamma = 0.99
        self.learning_rate = 0.00001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 2000
        self.explore = 50000 
        self.frame_per_action = 4
        #KOETODO: If this means how much to observe btw each train, then this model seems very
        #inefficient?
        self.timestep_per_train = 5 # Number of timesteps between training interval

        # experience replay buffer
        self.memory = deque()
        self.max_memory = 20000

        # create model
        self.model = None

        # Performance Statistics
        self.stats_window_size= 50 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time

    def get_action(self, state, measurement, goal, inference_goal):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            #print("----------Random Action----------")
            action_idx = random.randrange(self.action_size)
        else:
            measurement = np.expand_dims(measurement, axis=0)
            goal = np.expand_dims(goal, axis=0)
            f = self.model.predict([state, measurement, goal]) # [1x6, 1x6, 1x6]
            f_pred = np.vstack(f) # 3x6
            obj = np.sum(np.multiply(f_pred, inference_goal), axis=1) # num_action
            #KOE: Double-check the NN output a bit.
            action_idx = np.argmax(obj)
        return action_idx

    # Save trajectory sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, m_t, is_terminated):
        self.memory.append((s_t, action_idx, r_t, s_t1, m_t, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

    # Pick samples randomly from replay memory (with batch_size)
    def train_minibatch_replay(self, goal):
        """
        Train on a single minibatch
        """
        batch_size = min(self.batch_size, len(self.memory))
        rand_indices = np.random.choice(len(self.memory)-(self.timesteps[-1]+1), self.batch_size)

        state_input = np.zeros(((batch_size,) + self.state_size)) # Shape batch_size, img_rows, img_cols, 4
        measurement_input = np.zeros((batch_size, self.measurement_size)) 
        goal_input = np.tile(goal, (batch_size, 1))
        f_action_target = np.zeros((batch_size, (self.measurement_size * len(self.timesteps)))) 
        action = []

        for i, idx in enumerate(rand_indices):
            future_measurements = []
            last_offset = 0
            done = False
            for j in range(self.timesteps[-1]+1):
                if not self.memory[idx+j][5]: # if episode is not finished
                    if j in self.timesteps: # 1,2,4,8,16,32
                        if not done:
                            future_measurements += list( (self.memory[idx+j][4] - self.memory[idx][4]) )
                            last_offset = j
                        else:
                            future_measurements += list( (self.memory[idx+last_offset][4] - self.memory[idx][4]) )
                else:
                    done = True
                    if j in self.timesteps: # 1,2,4,8,16,32
                        future_measurements += list( (self.memory[idx+last_offset][4] - self.memory[idx][4]) )
            f_action_target[i,:] = np.array(future_measurements)
            state_input[i,:,:,:] = self.memory[idx][0]
            measurement_input[i,:] = self.memory[idx][4]
            action.append(self.memory[idx][1])

        f_target = self.model.predict([state_input, measurement_input, goal_input]) # Shape [32x18,32x18,32x18]

        for i in range(self.batch_size):
            f_target[action[i]][i,:] = f_action_target[i]

        loss = self.model.train_on_batch([state_input, measurement_input, goal_input], f_target)

        return loss

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


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
    env = UnityEnv("../unity_envs/kais_banana", worker_id=0, use_visual=True, uint8_visual=True, flatten_branched=True)
    '''game = DoomGame()
    game.load_config("../../scenarios/health_gathering.cfg")
    game.set_sound_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()'''

    initial_observation = env.reset()
    #KOETODO This would have to be manually configured for each environment.
    #KOE: What is this misc??


    #misc = game_state.game_variables  # [Health]
    #prev_misc = misc
    #KOE: I think this should be the same as my battery measure.
    misc = 100 # [Health]
    prev_misc = misc

    # game.get_available_buttons_size() # [Turn Left, Turn Right, Move Forward]

    action_size = env.action_space.n
    print("Env has ", action_size, " actions.")
    measurement_size = 3 # [Battery, posion, food]
    timesteps = [1,2,4,8,16,32]
    goal_size = measurement_size * len(timesteps)

    img_rows , img_cols = 84, 84 #KOE: Think this is still correct.
    # Convert image into Black and white
    img_channels = 3 # KOE: Think this is correct.

    state_size = (img_rows, img_cols, img_channels)
    agent = DFPAgent(state_size, measurement_size, action_size, timesteps)

    agent.model = Networks.dfp_network(state_size, measurement_size, goal_size, action_size, len(timesteps), agent.learning_rate)

    #x_t = game_state.screen_buffer # 480 x 640
    x_t = preprocessImg(initial_observation, size=(img_rows, img_cols))
    #KOE: Preprocessing to get black and white.

    #KOE: Not sure what is going on here. 4 images in a row?
    s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    # Number of food pickup as measurement
    food = 0

    # Number of poison pickup as measurement
    poison = 0

    # Initial normalized measurements.
    #KOE: Not sure if I need to normalize...
    #KOE: Original paper normalized by stddev of the value under random exploration.
    m_t = np.array([misc/100.0, food, poison])

    # Goal
    # KOE: Battery, poison, food. No way to affect battery so far except standing still. Maybe that will happen?
    goal = np.array([0.5, -1.0, 1.0] * len(timesteps))

    #TODOKOE: Need to implement randomized goals.
    #KOE: Now we're talking! This is the one to allow evolving goals!
    # Goal for Inference (Can change during test-time)
    inference_goal = goal

    #KOE: env.step returns observation, reward, done, info. done should be the one we need to check.
    #is_terminated = game.is_episode_finished()
    done = False

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_reward = 0 # Maximum episode life (Proxy for agent performance) #KOE: Remove?
    #life = 0

    # Buffer to compute rolling statistics 
    reward_buffer = []

    while not done:

        loss = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx  = agent.get_action(s_t, m_t, goal, inference_goal) #KOE: This is the forward pass through the NN.

        '''
        KOE: Here, we take the action, observe rewards, done and skip ahead.
        game.set_action(a_t.tolist())
        skiprate = agent.frame_per_action
        game.advance_action(skiprate) #Repeats the action skiprate times and returns state after that.

        game_state = game.get_state()  # Observe again after we take the action
        is_terminated = game.is_episode_finished()

        r_t = game.get_last_reward() 
        '''

        #The vector space in Unity has 4 branches, with multiple actions i each! Those can also be combined!
        #I need the ANN output to be able to select all combinations.
        #TODO Believe step just wants the index of the action.
        print("Taking action ", action_idx)
        observation, reward, done, info = env.step(action_idx)
        print("Got reward: ", reward)
        #TODO How to step ahead multiple steps? - I asked github- check what they suggest.

        #Observation is the image. vector_observations are the measurements.
        #battery, eaten_poison, eaten_food
        meas = info['brain_info'].vector_observations
        if (done):
            if ((food-poison) > max_reward):
                max_reward = (food-poison)
            GAME += 1
            reward_buffer.append(food-poison)
            print ("Episode Finish ")
            #game.new_episode()
            misc = 100 #KOE: Not sure what's the point of this. Maybe remove?
            #misc = game_state.game_variables
            #x_t1 = game_state.screen_buffer
            #reset returns the initial scren buffer.
            x_t1 = env.reset()
        else:
            x_t1 = observation
            misc = meas[0]

        #Img to black/white
        x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #KOE: What is this? Some sequence of images?


        #KOETODO: Need to think about if I should give the banana signal only exactly when picking, or
        #also a few seconds after (as I do now).


        #TODO: Meas will now have accumulated foods/poisons. I could also give them as 0/1. Not sure
        #what is best. Original code gave accumulated values, but it shouldn't matter since the DIFF
        #is what is used in predictions.
        if (reward==-1): # Pick up Poison
            poison += 1
        if (reward==1): # Pick up food
            food += 1

        #KOE: Remove?
        '''if (done):
            life = 0
        else:
            life += 1'''

        # Update the cache
        prev_misc = misc

        #KOETODO: Storing here m_t, but we want to predict m_t+1. How is that trained?
        # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, r_t, s_t1, m_t, done)

        #KOETODO: Think about normalization.
        m_t = np.array([meas[0]/100.0, food, poison]) # Measurement after transition

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            loss = agent.train_minibatch_replay(goal)
            
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            agent.model.save_weights("models/dfp.h5", overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if (done):
            print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                  "/ Food", food, "/ Poison", poison, "/ LOSS", loss)

            food = 0
            poison = 0

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe: 
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(reward_buffer)))
                agent.var_score.append(np.var(np.array(reward_buffer)))

                # Reset rolling stats buffer
                reward_buffer = []

                # Write Rolling Statistics to file
                with open("statistics/dfp_stats.txt", "w") as stats_file:
                    stats_file.write('Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score: ' + str(max_reward) + '\n')
                    stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                    stats_file.write('var_score: ' + str(agent.var_score) + '\n')

#KOE: Made it to the end. Now test running, print out, debug, etc.
