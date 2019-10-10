#For doing analysis on the best evolved network
import pickle
import neat
import sys
import numpy as np
import time
import Utilities

#TODO: Waitimng a bith with this. When setting it up, I need a single method to set up interfaces to fitness func.
#def store_individual_fitness(genome, doom_config_file):
#    net = neat.nn.FeedForwardNetwork.create(genome, config)
#    experiment_interface = set_up_nn_experiment(doom_config_file)
#    fitness = experiment_interface.test_new_individual(net)
#    print("Fitness was ", fitness)
#    avg_reward_vector = []
#    avg_reward_vector.append(fitness)
#    avg_reward_vector = np.array(avg_reward_vector)
#    f1 = open('reward_stats_with_evolved_nn.csv', 'a+')
#    np.savetxt(f1, avg_reward_vector, delimiter=" ")

def store_individual_behavior(genome):

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    num_steps_per_dimension = 20
    measures_range = {"battery":(0,10),"foods" : (0,30), "poisons":(0,30)} #Battery is normalized. Normally, it ranges from 0 to 10, representing 0 to 100%. If battery is larger, e.g. size 30, range will change accordingly.

    measures_to_objectives_matrix = [] #Vector of vector where each element has form [m1, m2, m3, o1, o2, o3]
    col_headers = ["m_battery", "m_poisons", "m_food", "o_battery", "o_poisons", "o_foods"]

    for batt in np.linspace(start=measures_range["battery"][0], stop=measures_range["battery"][1], num=num_steps_per_dimension):
        for foods in np.linspace(start=measures_range["foods"][0], stop=measures_range["foods"][1], num=num_steps_per_dimension):
            for pois in np.linspace(start=measures_range["poisons"][0], stop=measures_range["poisons"][1],
                                      num=num_steps_per_dimension):
                print(batt, ", ", pois, ", ", foods)
                nn_output = net.activate([batt, pois, foods])
                print("Act output: ", nn_output)
                measures_to_objectives_matrix.append([batt, pois, foods,*nn_output])

    measures_to_objectives_matrix=np.array(measures_to_objectives_matrix)
    f1=open("nn_behavior_measures_to_objectives.csv", 'w+')
    for item in col_headers:
        f1.write(item+" ")
    f1.write("\n")
    np.savetxt(f1, measures_to_objectives_matrix, delimiter=" ")

if __name__ == '__main__':

    analysis_mode = sys.argv[1]

    winner_filename = sys.argv[2] #Pickled winner indiv


    if(len(sys.argv) > 3):
        seed = int(sys.argv[3])
    else:
        seed = 1
    Utilities.store_seed_to_folder(seed, ".", "post_evaluate_winner")


    with open(winner_filename, 'rb') as pickle_file:
        winner_genome = pickle.load(pickle_file)

    config_file = "/home/kaiolae/code/dfp_banana/Direct-Future-Prediction-Keras/config"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    if analysis_mode=="fitness":
        store_individual_fitness(winner_genome)#, doom_config_file)
    elif analysis_mode=="behavior":
        store_individual_behavior(winner_genome)
    elif analysis_mode == "video":
        net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        #experiment_interface = set_up_nn_experiment(doom_config_file, num_simulators=1)
        experiment_interface.test_new_individual(net, store_to_video=True)
    else:
        print("Analysis mode ", analysis_mode, " is not supported.")

