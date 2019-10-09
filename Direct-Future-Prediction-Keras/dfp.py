#!/usr/bin/env python
from __future__ import print_function

import skimage
import random
from random import choice
import numpy as np
from collections import deque
import time
import os
import sys
from gym_unity.envs import UnityEnv
import getopt

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K

import itertools as it
from time import sleep
import tensorflow as tf

from networks import Networks
import pandas as pd

from helper_code import total_size

#If true, we try to predict the number of picked batteries rather than the battery level - to make this prediction more similar to that of
#foods and poisons- The reason is learned battery seeking performs worse than food seeking.
MEASURE_NUM_BATTERIES_INSTEAD_OF_BATTERY_LEVEL = True


#KOE: Get rid of image preprocessing and stacking.. Deepq works fine without it.
#KOE: making my edits directly here. The original one is still in dfp_original.py
def preprocessImg(img, size):

    #KOE: I think my Unity pictures are alreay the right shape - no need to do this reshaping.
    #img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    #img = skimage.transform.resize(img,size)
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

BATTERY_CAPACITY =100

class DFPAgent:

    def __init__(self, state_size, measurement_size, action_size, timesteps):

        # get size of state, measurement, action, and timestep
        self.state_size = state_size
        self.measurement_size = measurement_size
        self.action_size = action_size
        self.timesteps = timesteps

        # these is hyper parameters for the DFP
        self.gamma = 0.99
        self.learning_rate = 0.00001 #KOE: Original value 0.00001. Trying ten times less after problems may 2019

        #Epsilon-greedy with chance of random action decreasing gradually.
        #Currently, it seems to be halved around game 100 (50% chance of random), and zero around game
        #200 (full exploit).
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32 # KOE: Increased from 32 to 256 to see if I can utilize gpu more. No effect.
        self.observe = 2000
        self.explore = 50000 
        self.frame_per_action = 4
        #KOETODO: If this means how much to observe btw each train, then this model seems very
        #inefficient?
        #KOE: Note: The standard q-learning in openaiGym uses value 1 here: Train every step!
        #However, the one I'm using has set that parameter to 5.
        self.timestep_per_train = 5 #KOE: Increased from 5 to see if I can improve GPU utilization # Number of timesteps between training interval

        # experience replay buffer
        self.memory = deque()
        self.max_memory = 50000 #was 20000 #NOTE: My deepq implementation has 50000.

        # create model
        self.model = None

        # Performance Statistics
        self.stats_window_size= 10 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time

        self.next_mem_id = 0

    def get_predicted_effects(self, state, measurement, goal):
        """
        Like get_action, but stops before calculating the action, returning the predicted effects instead.
        """

        measurement = np.expand_dims(measurement, axis=0)
        goal = np.expand_dims(goal, axis=0)
        f = self.model.predict([state, measurement, goal]) # [1x6, 1x6, 1x6]
        f_pred = np.vstack(f) # 3x6
        return f_pred


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

        #print("_____________SIZE OF AN ELEM IN BUFFER: ___________________", total_size(self.memory[-1]))
        #print("_____________SIZE OF ENTIRE BUFFER: ___________________", total_size(self.memory))



    # Pick samples randomly from replay memory (with batch_size)
    #KOETODO: Could it be a problem that the rewards are much sparser than the energy-feedback?
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

        #KOE: Doing some checks on how sparse the measurements-streams are.
        futures_with_no_meas_change = 0
        futures_with_meas_change = 0
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
                        if future_measurements[-1]!=0 or future_measurements[-2] != 0:
                            futures_with_meas_change += 1
                        else:
                            futures_with_no_meas_change +=1
                else:
                    done = True
                    if j in self.timesteps: # 1,2,4,8,16,32
                        future_measurements += list( (self.memory[idx+last_offset][4] - self.memory[idx][4]) )
            #print("Target array shape: ", f_action_target.shape)
            #print("Future meas is ", future_measurements)
            #print("Input array shape: ", np.array(future_measurements).shape)
            f_action_target[i,:] = np.array(future_measurements) #TODO: KOE modified due to error
            state_input[i,:,:,:] = self.memory[idx][0]
            measurement_input[i,:] = self.memory[idx][4]
            action.append(self.memory[idx][1])

        #KOETODO: Why is this running in parallell with agent simulations??
        #print("Done creating training vector. Futures without meas change: ", futures_with_no_meas_change, " and with: ", futures_with_meas_change)
        f_target = self.model.predict([state_input, measurement_input, goal_input]) # Shape [32x18,32x18,32x18]

        for i in range(self.batch_size):
            f_target[action[i]][i,:] = f_action_target[i]

        #print("Training on batch")
        #KOETODO: What is in the loss? Seems it may have 1 value per action?
        #KOETODO: Storing the history of inputs/outputs to check that it is sensible.

        #print("Dumping Input/output pair to file in input_output_examples/")
        #np.save("input_output_examples/state_input", state_input)
        #np.save("input_output_examples/meaurement_input", measurement_input)
        #np.save("input_output_examples/goal_input", goal_input)
        #np.save("input_output_examples/f_target", f_target)


        #KOETODO: Seems the target here is a value I have already predicted. Strange...
        loss = self.model.train_on_batch([state_input, measurement_input, goal_input], f_target)

        #Storing training images for testing. TODO: Remove.
        if not os.path.exists('images_for_testing.out.npy'):
            np.save('images_for_testing.out', state_input)
            print("STORED TRAINING IMAGES TO NUMPY")


        #print("Model is ", self.model)
        #print("Loss was ", loss)

        return loss

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":


    #Parsing arguments
    loaded_model = '' #if empty, we start from scratch. TODO: Loading is an experiment. May not work, since we lose the replay buffer.
    goal_agnostic = True #Goal-agnostic training was found to be essential to generalize to new goals in the original DFP paper.
    battery_limited = False #If true, agent stops and episode ends if battery runs out.
    argv = sys.argv[1:]
    SAVE_TO_FOLDER = "oct8_explicitly_controlled_battery_charging"
    seed = 1

    try:
        opts, args = getopt.getopt(argv, "l:g:b:s:d", ["loaded_model=", "goal_agnostic_off", "battery_limit_on", "save_to", "seed"])
    except getopt.GetoptError:
        print('dfp.py --loaded_model optional_loaded_model.h5')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-l", "--loaded_model"):
            loaded_model = arg
        if opt in ("-g", "--goal_agnostic_off"):
            goal_agnostic = False
        if opt in ("-b", "--battery_limit_on"):
            battery_limited = True
        if opt in ("-s", "--save_to"):
            SAVE_TO_FOLDER = arg
        if opt in ("-d", "--seed"):
            seed = int(arg)




    print("Training is goal agnostic? ", goal_agnostic)



    if goal_agnostic:
        SAVE_TO_FOLDER+="_agnostic"

    if battery_limited:
        SAVE_TO_FOLDER+="_battery_limit_on"

    if not os.path.exists(SAVE_TO_FOLDER):
        os.makedirs(SAVE_TO_FOLDER)
        os.makedirs(SAVE_TO_FOLDER+"/model")

    Utilities.store_seed_to_folder(seed, SAVE_TO_FOLDER, "dfp")

    start=time.time()
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
    env = UnityEnv("../unity_envs/kais_banana_with_explicit_charge_decision_red_battery_300_timesteps", worker_id=22, use_visual=True,  flatten_branched=True, seed=seed) #KOE: Note: If I accept images as uint8_visual=True, I have to convert to float later.

    print("Resetting env")
    initial_observation = env.reset()
    #KOETODO This would have to be manually configured for each environment.

    battery = 100 # [Health]
    prev_battery = battery

    # game.get_available_buttons_size() # [Turn Left, Turn Right, Move Forward]
    print("Action space is: ", env.action_space)
    action_size = env.action_space.n
    print("Env has ", action_size, " actions.")
    measurement_size = 3 # [Battery, posion, food]
    timesteps = [1,2,4,8,16,32] # For long horizon: [4,8,16,32,64,128]
    goal_size = measurement_size * len(timesteps)

    img_rows , img_cols = 84, 84 #KOE: Think this is still correct.
    # Convert image into Black and white

    #KOETODO Not quite sure what happens here - I'm making images black/white, so what is the point?
    img_channels = 3 # KOE: If I want to change this, I have to also edit the frame stacking when forming s_t

    state_size = (img_rows, img_cols, img_channels)
    agent = DFPAgent(state_size, measurement_size, action_size, timesteps)
    agent.model = Networks.dfp_network(state_size, measurement_size, goal_size, action_size, len(timesteps),
                                       agent.learning_rate)

    if loaded_model:
        print("Loading stored model from ", loaded_model)
        agent.load_model(loaded_model)
        agent.epsilon = agent.final_epsilon #After training, we want to visualize without randomness.
    else:
        print("Starting training from scratch. Not loading model.")

    #x_t = game_state.screen_buffer # 480 x 640
    #x_t = preprocessImg(initial_observation, size=(img_rows, img_cols))

    #np.save("input_output_examples/initial_obs.npy", initial_observation)
    #np.save("input_output_examples/preprocessed_obs.npy", x_t)
    #KOE: Preprocessing to get black and white.

    #KOE: Not sure what is going on here. 4 images in a row?
    #s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = initial_observation
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    #np.save("input_output_examples/stacked_obs.npy", s_t)
    # Number of food pickup as measurement
    food = 0

    # Number of poison pickup as measurement
    poison = 0
    num_batteries = 0

    if MEASURE_NUM_BATTERIES_INSTEAD_OF_BATTERY_LEVEL:
        m_t = np.array([num_batteries, poison, food])
    else:
        #KOE: Normalizing battery by dividing by 10, so battery ranges from 0->10, the two others around 0->20/30. Should be even enough?
        m_t = np.array([battery/10.0, poison, food])

    # Goal
    # KOE: Battery, poison, food. No way to affect battery so far except standing still. Maybe that will happen?
    #KOETODO: Rewarding both food and poison as an initial test. If that works, but the other not,
    #maybe the color vision is a problem?
    if goal_agnostic:
        goal_vector= [random.uniform(-1,1) for i in range(3)]
    else:
        goal_vector = [-1.0, -1.0, 1.0]

    goal = np.array(goal_vector * len(timesteps))
    print("Initial goal vector is ", goal_vector)
    print("Initial goal is ", goal)

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
    max_reward = 0 # Maximum episode life (Proxy for agent performance) #KOE: Remove?
    #life = 0

    # Buffer to compute rolling statistics 
    reward_buffer = []
    food_buffer = []
    num_batteries_buffer = []
    poison_buffer = []
    loss_buffer = []
    battery_buffer = []

    #TODO Maybe set up some adaptive number of training episodes?
    timesteps_per_game = 300 #KOE: I double checked that there are in fact exactly 300 steps per episode. NOTE: These are the steps in which we act. We act every 5th step, meaning the episode lasts 1500 steps.
    total_training_timesteps = timesteps_per_game*10000 #Was 4000

    with open(SAVE_TO_FOLDER+"/dfp_stats.txt", "a+") as stats_file:
        stats_file.write('GAME_NUMBER ')
        stats_file.write('Max_Score ')
        stats_file.write('mavg_score ')
        stats_file.write('mavg_loss ')
        stats_file.write('var_score ')
        stats_file.write('mavg_battery ')
        stats_file.write('mavg_num_batteries ')
        stats_file.write('mavg_food ')
        stats_file.write('mavg_poison \n')

    for t in range(total_training_timesteps):
        loss = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx  = agent.get_action(s_t, m_t, goal, inference_goal) #KOE: This is the forward pass through the NN.


        #KOEComment: My unity agent also skips 5 frames between actions, controlled in the Unity interface.
        #The vector space in Unity has 4 branches, with multiple actions i each! Those can also be combined!
        #I need the ANN output to be able to select all combinations.
        #TODO Believe step just wants the index of the action.

        observation, reward, done, info = env.step(action_idx)

        if battery_limited and battery<0:
            done=True
            print("Battery empty. Stopping.")

        if (done):
            print("Game done at timestep ", t)
            if ((food-poison) > max_reward):
                max_reward = (food-poison)
            GAME += 1
            reward_buffer.append(food-poison)
            food_buffer.append(food)
            poison_buffer.append(poison)
            battery_buffer.append(battery)
            num_batteries_buffer.append(num_batteries)
            print ("Episode Finish ")
            #game.new_episode()
            battery = 100 #KOE: Not sure what's the point of this. Maybe remove?
            #battery = game_state.game_variables
            #x_t1 = game_state.screen_buffer
            #reset returns the initial scren buffer.
            x_t1 = env.reset()
            food=0
            poison=0
            num_batteries = 0

            if goal_agnostic:
                #If goal agnostic, we randomize goal between episodes
                #TODO Original DFP paper had two goal agnostic modes: [-1,1] and[0,1], Just trying the former here.
                goal_vector= [random.uniform(-1,1) for i in range(3)]
                goal = np.array(goal_vector * len(timesteps))
                inference_goal=goal
                print("Goal agnostic training. Randomizing goal: ", goal)

        else:
            x_t1 = observation
            battery-=1 #Always reducing by 1

        #Img to black/white
        #x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
        #x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        #s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #KOE: What is this? Some sequence of images?

        s_t1 = np.expand_dims(x_t1, axis=0) # 1x64x64x4


        #KOETODO: Need to think about if I should give the banana signal only exactly when picking, or
        #also a few seconds after (as I do now).


        #TODO: Meas will now have accumulated foods/poisons. I could also give them as 0/1. Not sure
        #what is best. Original code gave accumulated values, but it shouldn't matter since the DIFF
        #is what is used in predictions.
        if (reward>-1.05 and reward<-0.95): # Pick up Poison
            poison += 1
            print("Picked up. Current poison is ", poison)
        if (reward > 0.95 and reward <1.05): # Pick up food
            food += 1
            print("Picked up. Current food is ", food)
        if reward > 0.05 and reward < 0.15:
            print("Touched a battery!! Battery restored!")
            print("Reward was: ", reward)
            battery+=100
            num_batteries += 1
            if battery > BATTERY_CAPACITY:
                battery = BATTERY_CAPACITY
        if reward > -0.15 and reward <-0.05:
            print("Touched a battery without charging!!")
            print("Reward was: ", reward)


        # Update the cache
        prev_battery = battery

        #KOETODO: Storing here m_t, but we want to predict m_t+1. How is that trained?
        # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        #KOE: This seems to be handled by future_measurements in the training code.


        if MEASURE_NUM_BATTERIES_INSTEAD_OF_BATTERY_LEVEL:
            m_t = np.array([num_batteries, poison, food])  # Measurement after transition
        else:
            m_t = np.array([battery/10.0, poison, food]) # Measurement after transition

        #KOE: Moved this to under m_t oct2 2019 - strange to have it over.
        #print("Storing into memory. s_t is ", s_t.shape)
        #TODO: Analyze a bit what s_t is, is it 4 images in a row??
        agent.replay_memory(s_t, action_idx, r_t, s_t1, m_t, done)

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            loss = agent.train_minibatch_replay(goal)
            #print("Loss size ", len(loss))
            loss_buffer.append(np.mean(np.array(loss))) # Loss has 1 element for each action (?)
            
        s_t = s_t1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            agent.model.save_weights(SAVE_TO_FOLDER+"/model/dfp.h5", overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if (done):
            #print("DONE: loss size", len(loss) )
            if len(loss_buffer) >= 1:
                print("TIME", t, "/ GAME", GAME, "/ STATE", state, "/ EPSILON", agent.epsilon, \
                      "/ Food", food, "/ Poison", poison, "/ Avg Batt", np.mean(np.array(battery_buffer)),
                      "/ Avg Num Batt", np.mean(np.array(num_batteries_buffer)), "/ LOSS", loss_buffer[-1])
            # "/ ACTION", action_idx, "/ REWARD", r_t, \ KOE: Don't see point in printing CURRENT action and reward.


            # Save Agent's Performance Statistics
            #TODO Could consider simplifying and just saving after every game.
            if GAME % agent.stats_window_size == 0 and t > agent.observe: 
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(reward_buffer)))
                agent.var_score.append(np.var(np.array(reward_buffer)))
                #KOETODO: I think I can remove these. storing straight to file instead.

                mavg_score = np.mean(np.array(reward_buffer))
                var_score = np.var(np.array(reward_buffer))
                mavg_food = np.mean(np.array(food_buffer)) #TODO The moving average food here is a strange measure.
                mavg_battery = np.mean(np.array(battery_buffer))
                mavg_poison = np.mean(np.array(poison_buffer))
                mavg_loss = np.mean(loss_buffer)
                mavg_num_batteries = np.mean(np.array(num_batteries_buffer))
                food_buffer = []
                battery_buffer = []
                poison_buffer = []
                reward_buffer = []
                loss_buffer = []
                num_batteries_buffer = []

                # Write Rolling Statistics to file
                with open(SAVE_TO_FOLDER+"/dfp_stats.txt", "a+") as stats_file:
                    stats_file.write(str(GAME) + " ")
                    stats_file.write(str(max_reward) + " ")
                    stats_file.write(str(mavg_score) + ' ')
                    stats_file.write(str(mavg_loss) + ' ')
                    stats_file.write(str(var_score) + ' ')
                    stats_file.write(str(mavg_battery) + ' ')
                    stats_file.write(str(mavg_num_batteries) + ' ')
                    stats_file.write(str(mavg_food) + ' ')
                    stats_file.write(str(mavg_poison) + '\n')

    env.close()
    end=time.time()
    time_elapsed = end-start
    with open(SAVE_TO_FOLDER+"/timing_info.txt", "w") as text_file:
        print("Time Elapsed: {}".format(time_elapsed), file=text_file)


#KOE: Made it to the end. Now test running, print out, debug, etc.
