'''
remove nodes and find ranking correlation
'''

import argparse
import math
from resource import error
import time
import os

import copy

import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

from helpers import create_plot, create_plots, get_influence_node_ranking, get_influence_node_tuples, get_influence_node_tuples_new


from scipy import stats


def main(data_set):
    '''
    Main function for finding heatmap and layer-wise pearson correlation
    '''
    type = "i"
    percentage = 0.01
    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage
    total_columns = 3

    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    print_file = PrintFile(data_set)

    # Total removing nodes
    remove_nodes_per_iteration = math.floor(percentage * multilayer_graph.number_of_nodes)
    
    if remove_nodes_per_iteration == 0:
        remove_nodes_per_iteration = 1

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
        # remove_nodes_per_iteration = int(math.ceil(total_num_remove_nodes / (total_columns - 1)))

        # remove_nodes_per_iteration = 

        axs[0, 0].set_title("Full network")
        
        print("First node removal iteration")

        rhos = []

        # Get influences
        # influences = get_influence_node_tuples(multilayer_graph, print_file)
        influences = get_influence_node_tuples_new(multilayer_graph, print_file)
        curr_nodes = influences

        influences_sorted = sorted(influences, key=lambda x: (-x[1], x[0]))


        assert len(multilayer_graph.get_nodes()) == multilayer_graph.modified_number_of_nodes


        # get influuence 


        # iteration 2

        for _ in range(1, total_columns):
            # reset cache
            cache_nodes = curr_nodes
            # remove nodes
            nodes_to_remove = [pair[0] for pair in influences_sorted[:remove_nodes_per_iteration]]

            # print(nodes_to_remove)
            # print("before removal")
            # print(multilayer_graph.modified_number_of_nodes)
            # print(len(cache_nodes))

            multilayer_graph.remove_nodes(nodes_to_remove)

            print("after removal")
            print(multilayer_graph.modified_number_of_nodes)

            # calculate new ranking 
            influences, _ ,_ = bfs(multilayer_graph, print_file, False)
            # full graph influences
            influences_sorted = sorted(influences.items(), key=lambda x: (-x[1], x[0]))

            # print(influences)

            curr_nodes = influences.items()

            # new ranking has fewer nodes
            # remove nodes from cache_nodes

            # find intersection of node set, cache and influences, some nodes are removed because no longer active
            # remove nodes that are no longer active in any layers

            # cache has all active nodes in previous layer
            # influences has all nodes active in current layer
            node_difference = find_iso_nodes(cache_nodes, influences.items())
            # remove isolated nodes
            # find union of nodes to remove and isolated nodes
            # remove nodes from cache
            print(node_difference)
            print(cache_nodes)

            cache_nodes = remove_items_with_keys(node_difference, cache_nodes)
            

            # print(cache_nodes)
            # print(influences.items())

            assert len(cache_nodes) == len(influences)

            rho = calculate_spareman_correlation(cache_nodes, curr_nodes)
            rhos.append(rho)
            
            # print("number of nodes left = {}".format(multilayer_graph.modified_number_of_nodes))
            print(rho)

        return rhos

def find_iso_nodes(cache_nodes, influences):
    '''
    find union of the first element of two lists of tuples
    '''

    cache_nodes_set = set([pair[0] for pair in cache_nodes])
    curr_nodes_set = set([pair[0] for pair in influences])
    iso_nodes = cache_nodes_set - curr_nodes_set

    return list(iso_nodes)



def calculate_spareman_correlation(cache, curr):

    cache_inf =  [pair[1] for pair in cache]
    curr_inf = [pair[1] for pair in curr]

    rho, _ = stats.spearmanr(cache_inf, curr_inf)

    return rho


def remove_items_with_keys(nodes_to_remove, full_list):
    '''
    Remove a list of tuples from a full list
    '''

    # TODO: optimise
    for node in nodes_to_remove:
        for n in full_list:
            if node == n[0]:
                full_list.remove(n)

    return full_list

def create_influence_distribution_plots(node_influences, axs, col, multilayer_graph):
    df = pd.DataFrame(node_influences)
    count = np.isinf(df).values.sum()
    # print(count)
    # bin = len(node_influences)

    df.plot.hist(alpha=0.5, bins=20, grid=True, legend=None, ax=axs[0][col])  # Pandas helper function to plot a hist. Uses matplotlib under the hood.
    
    df_exp = df.apply(np.log)   # pd.DataFrame.apply accepts a function to apply to each column of the data
    df_exp.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_exp.dropna(inplace=True)

    count = np.isinf(df_exp).values.sum()
    # print(count)

    
    df_exp.plot.hist(alpha=0.5, bins=20, grid=True, legend=None, ax=axs[1][col])
    
    # plt.xlabel("Feature value")

    axs[0, col].set_title("Influence Histogram {}".format(multilayer_graph.dataset_file))
    
    axs[0, col].set_ylabel('Frequency')
    axs[0, col].set_xlabel('Influence')
    # axs[0].set_xlim(int(df.min()) - 1, int(df.max()) + 1)

    axs[1, col].set_ylabel('Frequency')
    axs[1, col].set_xlabel('Log(Influence)')
    axs[0, col].set_title("Iteration {}, Remaining nodes: {}".format(col, multilayer_graph.modified_number_of_nodes))


if __name__ == "__main__":
    # datasets = ["northamerica", "oceania", "asia", "europe", "southamerica"]

    # datasets = ["oceania"]
    datasets = ["aarhus"]
    for dataset in datasets:
        rhos = main(dataset)

        print(rhos)

    