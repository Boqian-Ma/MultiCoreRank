import argparse
import math
from resource import error
import time
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

from helpers import create_plot, create_plots, get_influence_node_ranking

def main():
    '''
    Main function for finding heatmap and layer-wise pearson correlation
    '''
    parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks')
    parser.add_argument('d', help='dataset')
    parser.add_argument('m', help='method: "i"=iterative influence calculation, "o"=once off influence calculation', choices=["i", "o"])
    parser.add_argument('p', help='percentage of node removal', type=float, choices=np.arange(0.0, 1.0, 0.1))
    parser.add_argument('c', help='total columns displayed', type=int, choices=range(1, 6))
    args = parser.parse_args()

    # e.g python main.py example i 0.9 5
    
    data_set = args.d
    type = args.m
    percentage = args.p
    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage
    total_columns = args.c

    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    print_file = PrintFile(data_set)

    # Total removing nodes
    total_num_remove_nodes = math.floor(percentage * multilayer_graph.number_of_nodes)

    print("dataset loading time: " + str(time.time()-start_time))

    # Create base plot
    fig, axs = plt.subplots(2, total_columns, figsize=(10 * total_columns, 20))
    
    # Experiment loop

    if total_columns == 1:
        # Full network 
        fig.suptitle('Dataset: {}'.format(data_set), fontsize=16)
        fig.suptitle('Dataset: {}, # of nodes: {}, # of layers: {} \n'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers), fontsize=16)
        create_plot(multilayer_graph, axs)
        # plt.savefig("figures/{}_{}.png".format(data_set, total_columns), format="png")
        print_file.print_figure(plt, total_columns, percentage, iterative=False)


    elif type == "i":
        # Plotting iterative node removal

        axs[0, 0].set_title("Full network")
        err = create_plots(multilayer_graph, 0, axs)

        # First plot        
        # -1 because the first column if the entire graph, then the next 5 are results of node removal
        remove_nodes_per_iteration = int(math.ceil(total_num_remove_nodes / (total_columns - 1)))

        fig.suptitle('Dataset: {}, # of nodes: {}, # of layers: {} \nTotal Node removal percentage: {}%\nTotal removing nodes: {}, Per iteration # of node removal: {}'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers ,percentage * 100 ,total_num_remove_nodes, remove_nodes_per_iteration), fontsize=16)
        print("First node removal iteration")
        
        for col in range(1, total_columns):
            # find influence
            influence, _ = bfs(multilayer_graph, print_file, False)
            print("iteration {} done....".format(col))
            nodes_to_remove = [pair[0] for pair in influence[:remove_nodes_per_iteration]]
            # remove nodes
            multilayer_graph.remove_nodes(nodes_to_remove)
            # find new plots
            err = create_plots(multilayer_graph, col, axs)
            if err:
                break
            axs[0, col].set_title("Iteration {}, Remaining nodes: {}".format(col, multilayer_graph.modified_number_of_nodes))

        print_file.print_figure(plt, total_columns, percentage, iterative=True)

        #plt.savefig("figures/{}_{}_{}_iterative.png".format(data_set, total_columns, percentage), format="png")

    elif type == "o":
        print("type = o")

        influence_ranking = get_influence_node_ranking(multilayer_graph, print_file)
        
        axs[0, 0].set_title("Full network")
        err = create_plots(multilayer_graph, 0, axs)

        remove_nodes_per_iteration = int(math.ceil(total_num_remove_nodes / (total_columns - 1)))

        fig.suptitle('Dataset: {}, # of nodes: {}, # of layers: {} \nTotal Node removal percentage: {}%\nTotal removing nodes: {}, Per iteration # of node removal: {}'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers ,percentage * 100 ,total_num_remove_nodes, remove_nodes_per_iteration), fontsize=16)
       
        print("First node removal iteration")

        for col in range(1, total_columns):
            # find influence
            print("iteration {} done....".format(col))

            start = (col - 1) * remove_nodes_per_iteration
            finish = col * remove_nodes_per_iteration

            nodes_to_remove = influence_ranking[start : finish]
            # remove nodes
            multilayer_graph.remove_nodes(nodes_to_remove)
            # find new plots
            err = create_plots(multilayer_graph, col, axs)
            if err:
                break
            axs[0, col].set_title("Iteration {}, Remaining nodes: {}".format(col, multilayer_graph.modified_number_of_nodes))

        # Parse influence file
        print_file.print_figure(plt, total_columns, percentage, iterative=False)

    print(time.time() - start_time)

if __name__ == "__main__":
    main()