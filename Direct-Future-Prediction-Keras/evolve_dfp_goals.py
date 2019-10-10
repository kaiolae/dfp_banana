#
# Takes a trained network, and runs it with evolving goals.
#

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
import sys
import Utilities

#Some parameters
BATTERY_REFILL_AMOUNT = BATTERY_CAPACITY = 100
FITNESS_EVALS_PER_INDIVIDUAL = 5 #Multiple fitness evals per individual to counter effect of randomness.
PENALTY_FOR_PICKING_BATTERY = 100
NUM_TIMESTEPS = 900
#This is especially important, since the objective of seeking food directly gives very high variance, works great sometimes -
#whereas seeking battery AND food gives stable and good behavior.

# Reporter helps store things on evolution-events such as generation end.
class StoreStatsReporter(neat.reporting.BaseReporter):

    def __init__(self):
        global store_to_folder
        self.generation = 0
        self.fitness_file_name = store_to_folder+"/fitness_summary.csv"
        fitness_file = open(self.fitness_file_name, "w+")
        fitness_file.write("Generation Best_Fitness Median_Fitness StdDev MedianTimeToBattEmpty MedianBattPicks MedianPoisPicks MedianFoodPicks\n")
        fitness_file.close()

        self.outputs_file_name = store_to_folder+"/nn_outputs_summary.csv"
        outputs_file = open(self.outputs_file_name, "w+")
        outputs_file.write("Generation Battery Poison Food\n")
        outputs_file.close()

        self.avg_outputs_storage_size = 0

    def start_generation(self, generation):
        self.generation = generation

    def end_generation(self, config, population, species_set):
        global stats
        global additional_statistics_storage
        # Storing fitness
        fitness_file = open(self.fitness_file_name, "a") #TODO: Fix! best_genome gest the historically best one - a lucky individual. Find instead the current best.
        fitness_file.write(str(self.generation) + " " + str(additional_statistics_storage["best_fitness"]) + " " +
                           str(stats.get_fitness_median()[-1]) + " "+ str(stats.get_fitness_stdev()[-1])+" " +
                           str(np.mean(np.array(additional_statistics_storage["battery_time"]), axis=0)) + " " +
                           str(np.mean(np.array(additional_statistics_storage["battery_picks"]), axis=0)) + " " +
                           str(np.mean(np.array(additional_statistics_storage["pois_picks"]), axis=0)) + " " +
                           str(np.mean(np.array(additional_statistics_storage["food_picks"]), axis=0)) + "\n")


        additional_statistics_storage["battery_time"] = []
        additional_statistics_storage["battery_picks"] = []
        additional_statistics_storage["pois_picks"] = []
        additional_statistics_storage["food_picks"] = []
        additional_statistics_storage["best_fitness"] = -1000

        # Storing NN outputs
        avg_nn_outputs_this_generation = avg_outputs_storage[self.avg_outputs_storage_size:]
        outputs_file = open(self.outputs_file_name, "a")
        for individual_outputs in avg_nn_outputs_this_generation:
            outputs_file.write(str(self.generation))
            for outputs in individual_outputs:
                outputs_file.write(" "+str(outputs))
            outputs_file.write("\n")
        self.avg_outputs_storage_size = len(avg_outputs_storage)

#EA code based on NEAT XOR example
def eval_genomes(genomes, config):
    #TODO: Set up for N evaluations of each individual, to counter randomness.
    global avg_outputs_storage
    global env
    global dfp_net
    global additional_statistics_storage
    current_most_fit= None
    current_highest_fitness = -1000
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #Running the net N times, collecting averages.
        all_eval_results = []
        for i in range(FITNESS_EVALS_PER_INDIVIDUAL):
            eval_result = evaluate_a_goal_vector([0,0,0], env, dfp_net , num_timesteps = NUM_TIMESTEPS, goal_producing_network=net, display=False, battery_refill_amount=BATTERY_REFILL_AMOUNT, battery_capacity=BATTERY_CAPACITY, penalty_for_picking=PENALTY_FOR_PICKING_BATTERY)
            avg_nn_outputs = np.mean(np.array(eval_result["goal_history"]), axis=0)
            eval_result["goal_history"] = avg_nn_outputs #Replacing the full history with the AVERAGE GOAL - that's all I'm using.
            all_eval_results.append(eval_result)
            print("Eval result is ", eval_result)


        #Goal_history has to be taken care of separately, since it has more than 1 scalar.
        all_goal_history = []
        for eval in all_eval_results:
            gh = eval["goal_history"]
            all_goal_history.append(gh)
            del eval["goal_history"] #Removing the item from dict.
        all_goal_history=np.array(all_goal_history)
        average_goal_history=np.mean(all_goal_history, axis=0)
        print("Average goal history is ", average_goal_history)


        eval_sums = Counter()
        counters = Counter()
        for itemset in all_eval_results:
            eval_sums.update(itemset)
            counters.update(itemset.keys())
        eval_averages = {x: float(eval_sums[x])/counters[x] for x in eval_sums.keys()}
        print("***************Eval averages is ", eval_averages, " *******************************")

        #We want to store the average outputs generated by the NN to see how the coeffs evolve over generations.
        #TODO Later, I may want to store all this rather than average. Maybe I can see interesting trends of how different objectives are prioritized at different points in agent's life?

        print("Avg nn outputs was: ", average_goal_history)
        avg_outputs_storage.append(average_goal_history)
        additional_statistics_storage["battery_time"].append(eval_averages["num_timesteps_elapsed"])
        additional_statistics_storage["battery_picks"].append(eval_averages["battery_picks"])
        additional_statistics_storage["pois_picks"].append(eval_averages["poison"])
        additional_statistics_storage["food_picks"].append(eval_averages["food"])

        #KOE: Here, I can choose what to optimize for: Battery, food, poison, etc.
        #How about food minus poison? Battery could take care of itself if we set up the "dead if no battery" rule?
        genome.fitness = eval_averages["food"]-eval_averages["poison"]
        if genome.fitness>current_highest_fitness:
            current_highest_fitness = genome.fitness
            current_most_fit=genome

        #print("Eval result was ", eval_result)
        print("fitness was: ", genome.fitness)
    #Dumping best indiv after each generation.
    additional_statistics_storage["best_fitness"] = current_highest_fitness
    #TODO: Set up dumping of each generation to different file?
    pickle.dump(current_most_fit, open(store_to_folder + "/winner_network.pickle", "wb"))  # Stored winner network. Load later and test.

def run_neat(config_file="config"):
    global stats
    global store_to_folder
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(15,None))
    p.add_reporter(
        StoreStatsReporter())  # KOE: My own reporter, that dumps progress to file as we go, allowing statistics on the fly.

    # 50 indivs seems to give 15 min per generation. 100 gen in 1 day, 250 in a weekend run.
    winner = p.run(eval_genomes, 50)  #TODO Seems around 25-50 was enough.

    # Display the winning genome.
    #print('\nBest genome:\n{!s}'.format(winner))
    #pickle.dump(winner,
    #            open(store_to_folder + "/winner_network.pickle", "wb"))  # Stored winner network. Load later and test.

    # Show output of the most fit genome against training data.
    #print('\nOutput:')
    #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    stats.save_genome_fitness()
    #visualize.plot_stats(stats, ylog=False, view=False)
    # visualize.plot_species(stats, view=True)

if __name__ == "__main__":
    ##Some parameters



    global store_to_folder
    store_to_folder = "oct9-evolving-agent-with-charge-output_2/"
    print("***************Storing results to folder ", store_to_folder, "*********************************")
    if not os.path.exists(store_to_folder):
        os.makedirs(store_to_folder)




    if(len(sys.argv) > 0):
        seed = int(sys.argv[1])
    else:
        seed = 1
    Utilities.store_seed_to_folder(seed, store_to_folder, "evolve_dfp_goals")
    # Stores average NN outputs. TODO Here we should also somehow store the generation number. Look into if NEAT allows that.
    global avg_outputs_storage
    avg_outputs_storage = []

    global additional_statistics_storage
    additional_statistics_storage = {}
    additional_statistics_storage["battery_time"] = []
    additional_statistics_storage["battery_picks"] = []
    additional_statistics_storage["pois_picks"] = []
    additional_statistics_storage["food_picks"] = []
    additional_statistics_storage["best_fitness"] = -1000

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
    global env
    env = UnityEnv("../unity_envs/kais_banana_with_explicit_charge_decision_red_battery_900_timesteps", worker_id=15, use_visual=True, flatten_branched=True, seed=seed)

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

    loaded_model = "oct8_explicitly_controlled_battery_charging_agnostic/model/dfp.h5"
    dfp_net.load_model(loaded_model)
    dfp_net.epsilon = dfp_net.final_epsilon

    global stats
    stats = neat.StatisticsReporter()

    # Configuring NEAT
    run_neat("config")

    avg_outputs_storage = np.array(avg_outputs_storage)
    np.savetxt(store_to_folder + "/evolving_nn_outputs.csv", avg_outputs_storage, delimiter=" ")

    env.close()
