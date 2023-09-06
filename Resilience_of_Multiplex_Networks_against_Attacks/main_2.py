import argparse
import math
from resource import error
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

from helpers import create_plot, create_plots, get_influence_node_ranking, get_influence, get_influence_node_tuples, get_influence_node_tuples_new

import pandas as pd

def main(data_set):
    '''
    Influence distribution plots
    '''
    # data_set = "oceania"
    total_columns = 2
    percentage = 0.1

    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)
    print("dataset loading time: " + str(time.time()-start_time))

    print_file = PrintFile(data_set)

    # Get number of nodes to remove
    remove_nodes_per_iteration = int(math.floor(percentage * multilayer_graph.number_of_nodes))
    if remove_nodes_per_iteration == 0:
        remove_nodes_per_iteration = 1


    # Create base plot
    fig, axs = plt.subplots(2, total_columns, figsize=(10 * total_columns, 20))

    # Iteration 1
    influences = get_influence_node_tuples_new(multilayer_graph, print_file)
    node_influences = [pair[1] for pair in influences]

    df = pd.DataFrame(node_influences)
    plot_influence_distribution(data_set, multilayer_graph.modified_number_of_nodes, axs, 0, df, print_file)

    # Remove nodes
    influences_sorted = sorted(influences, key=lambda x: (-x[1], x[0]))        
    nodes_to_remove = [pair[0] for pair in influences_sorted[:remove_nodes_per_iteration]]
    multilayer_graph.remove_nodes(nodes_to_remove)

    # Iteration 2
    influences = bfs(multilayer_graph, print_file, False)
    node_influences = [pair[1] for pair in influences.items()]
    df = pd.DataFrame(node_influences)
    plot_influence_distribution(data_set, multilayer_graph.modified_number_of_nodes, axs, 1, df, print_file)
    
    print_file.print_influence_distribution(plt, "{}_influence_distribution_log_{}".format(multilayer_graph.dataset_file, percentage))

    print(time.time() - start_time)

def plot_influence_distribution(data_set, remaining_nodes, axs, col, df, print_file):

    # count = np.isinf(df).values.sum()

    print("min = {}".format(df.min()))
    df.plot.hist(alpha=0.5, bins=10, grid=True, legend=None, ax=axs[0, col])  # Pandas helper function to plot a hist. Uses matplotlib under the hood.
    # Bottom plot

    # x_vals = axs[0, col].get_xticks()
    # axs[0, col].set_xticklabels(['{:3.0f}%'.format(x * 100) for x in x_vals])
    # plt.show()

    df_exp = df.apply(np.log)   # pd.DataFrame.apply accepts a function to apply to each column of the data
    df_exp.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_exp.dropna(inplace=True)
    # Count number of inf values
    # count = np.isinf(df_exp).values.sum()
    df_exp.plot.hist(alpha=0.5, bins=10, grid=True, legend=None, ax=axs[1, col], log=True)

    axs[0, col].set_title("Influence Histogram {}".format(data_set))
    axs[0, col].set_ylabel('Frequency')
    axs[0, col].set_xlabel('Influence')
    # axs[0].set_xlim(int(df.min()) - 1, int(df.max()) + 1)

    axs[1, col].set_ylabel('Frequency')
    axs[1, col].set_xlabel('Log(Influence)')
    axs[1, col].set_title("Iteration {}, Remaining nodes: {}".format(col, remaining_nodes))



if __name__ == "__main__":
    # datasets = ["celegans", "biogrid", "europe", "northamerica_0_2_13_14_11", "southamerica_9_17_21_52"]
    # datasets = ["celegans", "europe", "northamerica_0_2_13_14_11", "southamerica_9_17_21_52"]
    # datasets = ["homo"]
    # datasets = ["northamerica_0_2_13_14_11", "southamerica_9_17_21_52"]
    datasets = ["oceania"]

    for dataset in datasets:
        main(dataset)