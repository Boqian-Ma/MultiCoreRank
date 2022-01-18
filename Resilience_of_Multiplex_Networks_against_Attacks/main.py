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


def correlation_mean(list, num_layers):
    mean_diag = sum(list) / float(len(list))

    try:
        mean_no_diag = (sum(list) -  num_layers) / float(len(list) -  num_layers)
    except:
        mean_no_diag = float("NAN")
    
    # Print mean values
    return mean_diag, mean_no_diag

def create_plots(multilayer_graph, plot_col, axs):
    '''
    Create density distribution plot and heatmap
    Add to the existing set of plots
    '''
    # Calculate pearson coefficients
    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()

    # flatten list for density plots, not including nan values
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist if not math.isnan(item)]

    # heat map
    im = axs[0, plot_col].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)
    plt.colorbar(im, ax=axs[0, plot_col])

    # Calculate mean
    mean_diag, mean_no_diag = correlation_mean(pearson_flat_list, multilayer_graph.number_of_layers)
    
    # Print mean values
    at = AnchoredText("Mean including diag: {:.2f}\nMean excluding diag: {:.2f}".format(mean_diag, mean_no_diag), prop=dict(size=15), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[0, plot_col].add_artist(at)

    # normalised histogram
    weights = np.ones_like(pearson_flat_list) / float(len(pearson_flat_list))   # calculate weights
    hist = axs[1, plot_col].hist(pearson_flat_list, bins=10, weights=weights)
    axs[1, plot_col].set_xlim(-1, 1.5)
    axs[1, plot_col].set_ylabel('Density')
    axs[1, plot_col].set_xlabel('Pearson Correlation Coefficient')
    axs[1, plot_col].axvline(mean_no_diag, color='k', linestyle='dashed', linewidth=3)

    return None




def create_plot(multilayer_graph, axs):

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist]
    
    # heat map
    im = axs[0].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)    

    plt.colorbar(im, ax=axs[0])

    mean_diag, mean_no_diag = correlation_mean(pearson_flat_list, multilayer_graph.number_of_layers)

    # Print mean values
    at = AnchoredText("Mean including diag: {:.2f}\nMean excluding diag: {:.2f}".format(mean_diag, mean_no_diag), prop=dict(size=15), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[0].add_artist(at)

    # Normalised histogram
    weights = np.ones_like(pearson_flat_list) / float(len(pearson_flat_list))   # calculate weights
    hist = axs[1].hist(pearson_flat_list, bins=10, weights=weights)

    axs[1].set_xlim(-1, 1)
    axs[1].set_ylabel('Density')
    axs[1].set_xlabel('Pearson Correlation Coefficient')
    axs[1].axvline(mean_no_diag, color='k', linestyle='dashed', linewidth=3)

def save_influence_ranking(multilayer_graph, data_set):
    '''
    Save influence nodes to file
    '''
    influence = bfs(multilayer_graph, PrintFile(data_set), False, data_set)
    # Extract node and retain order
    influence = [x[0] for x in influence]

    print(influence)

    string = " ".join(map(str, influence))

    with open("influence/{}_influence_ranking.txt".format(data_set), 'w+') as f:
        f.write(string)


def read_influence_nodes_ranking(multilayer_graph, data_set):
    '''
    Read influence ranking file
    '''
    with open("influence/{}_influence_ranking.txt".format(data_set), 'r') as f:
        inf = f.readline()

    influence = inf.strip().split(" ")

    print(influence)

    if len(influence) != multilayer_graph.number_of_nodes:
        raise ValueError("influence ranking file is incomplete: length of given file is different from length of graph nodes")

    return map(int, influence)

def main():

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
        plt.savefig("figures/{}_{}.png".format(data_set, total_columns), format="png")

    elif type == "i":
        # Plotting iterative node removal

        axs[0, 0].set_title("Full network")
        err = create_plots(multilayer_graph, 0, axs)

        # First plot        
        # -1 because the first column if the entire graph, then the next 5 are results of node removal
        remove_nodes_per_iteration = int(math.ceil(total_num_remove_nodes / (total_columns - 1)))

        fig.suptitle('Dataset: {}, # of nodes: {}, # of layers: {} \nTotal Node removal percentage: {}%\nTotal removing nodes: {}, Per iteration # of node removal: {}'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers ,percentage * 100 ,total_num_remove_nodes, remove_nodes_per_iteration), fontsize=16)
        print("First iteration")
        
        for col in range(1, total_columns):
            # find influence
            influence = bfs(multilayer_graph, PrintFile(data_set), False, data_set)
            print("iteration {} done....".format(col))
            nodes_to_remove = [pair[0] for pair in influence[:remove_nodes_per_iteration]]
            # remove nodes
            multilayer_graph.remove_nodes(nodes_to_remove)
            # find new plots
            err = create_plots(multilayer_graph, col, axs)
            if err:
                break
            axs[0, col].set_title("Iteration {}, Remaining nodes: {}".format(col, multilayer_graph.modified_number_of_nodes))

        plt.savefig("figures/{}_{}_{}_iterative.png".format(data_set, total_columns, percentage), format="png")

    elif type == "o":

        # Locate influence file
        if not os.path.isfile('influence/{}_influence_ranking.txt'.format(data_set)):
            # calculate influence and put in 
            save_influence_ranking(multilayer_graph, data_set)
        
        # Load influence ranking
        try:
            influence_ranking = read_influence_nodes_ranking(multilayer_graph, data_set)
        except ValueError or IOError:
            save_influence_ranking(multilayer_graph, data_set)
            influence_ranking = read_influence_nodes_ranking(multilayer_graph, data_set)


        axs[0, 0].set_title("Full network")
        err = create_plots(multilayer_graph, 0, axs)

        remove_nodes_per_iteration = int(math.ceil(total_num_remove_nodes / (total_columns - 1)))

        fig.suptitle('Dataset: {}, # of nodes: {}, # of layers: {} \nTotal Node removal percentage: {}%\nTotal removing nodes: {}, Per iteration # of node removal: {}'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers ,percentage * 100 ,total_num_remove_nodes, remove_nodes_per_iteration), fontsize=16)
        print("First iteration")

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
        plt.savefig("figures/{}_{}_{}_once.png".format(data_set, total_columns, percentage), format="png")

    print(time.time()-start_time)

if __name__ == "__main__":
    main()