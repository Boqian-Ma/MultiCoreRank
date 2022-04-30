'''
Iteratively remove nodes and find influence distributions

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

from helpers import create_plot, create_plots, get_influence_node_ranking


from scipy import stats


def main(data_set):
    '''
    Main function for finding heatmap and layer-wise pearson correlation
    '''
    # parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks')
    # parser.add_argument('d', help='dataset')
    # parser.add_argument('m', help='method: "i"=iterative influence calculation, "o"=once off influence calculation', choices=["i"])
    # parser.add_argument('p', help='percentage of node removal', type=float, choices=np.arange(0.0, 1.0, 0.01))
    # parser.add_argument('c', help='total columns displayed', type=int, choices=range(1, 6))
    # args = parser.parse_args()

    # e.g python main.py aps_0_1_6_8_9 o 0.1 5
    
    # data_set = ""
    type = "i"
    percentage = 0.2
    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage
    total_columns = 5

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
        remove_nodes_per_iteration = int(math.ceil(total_num_remove_nodes / (total_columns - 1)))

        axs[0, 0].set_title("Full network")
        # err = create_plots(multilayer_graph, 0, axs)

        # First plot        
        # -1 because the first column if the entire graph, then the next 5 are results of node removal
        fig.suptitle('Dataset: {}, # of nodes: {}, # of layers: {} \nTotal Node removal percentage: {}%\nTotal removing nodes: {}, Per iteration # of node removal: {}'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers ,percentage * 100 ,total_num_remove_nodes, remove_nodes_per_iteration), fontsize=16)
        print("First node removal iteration")


        influences = bfs(multilayer_graph, print_file, False)

        # full graph influences
        influences = sorted(influences.items(), key=lambda x: (-x[1], x[0]))

        # full graph
        # node_influences = [pair[1] for pair in influences]
        # create_influence_distribution_plots(node_influences, axs, 0, multilayer_graph)

        curr_node_rank = influences

        rhos = []

        nodes_to_remove = [pair[0] for pair in influences[:remove_nodes_per_iteration]]
        multilayer_graph.remove_nodes(nodes_to_remove)





        # for col in range(1, total_columns):
        #     # cache to calculate spearman
        #     cache_node_rank = copy.copy(curr_node_rank)

        #     # find influence
        #     if col > 1:
        #         influences = bfs(multilayer_graph, print_file, False)
        #         influences = sorted(influences.items(), key=lambda x: (-x[1], x[0]))

        #     # nodes to remove from graph
            


        #     # print(nodes_to_remove)
        #     print("cache node before removal")
        #     print(cache_node_rank, len(cache_node_rank))


        #     # should equal to two
        #     print("influences before removal")
        #     print(influences, len(influences))
        #     print("nodes to remove {}".format(nodes_to_remove))


        #     cache_node_rank = remove_items_with_keys(nodes_to_remove, cache_node_rank)
        #     curr_node_rank = remove_items_with_keys(nodes_to_remove, influences)

        #     print("cache node after removal")
        #     print(cache_node_rank, len(cache_node_rank))
        #     print(curr_node_rank, len(curr_node_rank))
        #     print(influences, len(influences))

        #     assert len(cache_node_rank) == len(curr_node_rank)

        #     rho = calculate_spareman_correlation(cache_node_rank, curr_node_rank)

        #     rhos.append(rho)
        #     print(rho)

        #     print("iteration {} done....".format(col))

        #     nodes_to_remove = [pair[0] for pair in influences[:remove_nodes_per_iteration]]
        #     multilayer_graph.remove_nodes(nodes_to_remove)
            
            # Remove top nodes
            
            # remove nodes from graph

            # print("yeet")

            # print(curr_node_rank)
            # print(cache_node_rank)

            # node_influences = [pair[1] for pair in influences[remove_nodes_per_iteration:]]
            # create_influence_distribution_plots(node_influences, axs, col, multilayer_graph)

        # print_file.print_figure(plt, total_columns, percentage, iterative=True)
        # print_file.print_influence_distribution(plt, "{}_influence_remove_{}_{}".format(multilayer_graph.dataset_file, percentage, total_columns))
        #plt.savefig("figures/{}_{}_{}_iterative.png".format(data_set, total_columns, percentage), format="png")
        

        return rhos
def calculate_spareman_correlation(cache, curr):

    cache_inf =  [pair[1] for pair in cache]
    curr_inf = [pair[1] for pair in curr]

    rho, _ = stats.spearmanr(cache_inf, curr_inf)

    return rho


def remove_items_with_keys(nodes_to_remove, full_list):
    '''
    Remove a list of tuples from a full list
    '''
    for node in full_list:
        if node[0] in nodes_to_remove:
            full_list.remove(node)

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

    datasets = ["example"]
    for dataset in datasets:
        rhos = main(dataset)

    