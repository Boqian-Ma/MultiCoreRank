import argparse
from audioop import mul
import math
from resource import error
import time
import os

import numpy as np
import matplotlib
# from sklearn.datasets import make_gaussian_quantiles

from helpers import correlation_mean
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

def create_plot_layers(multilayer_graph, axs):

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist]

    print(pearson_flat_list)

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


def main(dataset, layers_to_keep=None):
    # e.g python main.py example i 0.9 5
    data_set = dataset

    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage
    total_columns = 1

    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    if layers_to_keep:
        multilayer_graph.keep_layers(layers_to_keep)

    print_file = PrintFile(data_set)

    # Total removing nodes

    print("dataset loading time: " + str(time.time()-start_time))
    # Create base plot
    fig, axs = plt.subplots(2, total_columns, figsize=(10 * total_columns, 20))
    # Experiment loop
    if total_columns == 1:
        # Full network 
        fig.suptitle('Dataset: {}'.format(data_set), fontsize=16)

        if layers_to_keep:
            fig.suptitle('Dataset: {}, # of nodes: {}, Total # of layers: {} \nDisplayed layers: {}'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers, layers_to_keep), fontsize=16)
        else: 
            fig.suptitle('Dataset: {}, # of nodes: {}, Total # of layers: {}'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers), fontsize=16)

        
        create_plot_layers(multilayer_graph, axs)
        # plt.savefig("figures/{}_{}.png".format(data_set, total_columns), format="png")
        print_file.print_subfigure(plt, layers_to_keep)

    print(time.time() - start_time)

    print("\n\n\n {} \n\n\n".format(multilayer_graph.get_number_of_edges()))

if __name__ == "__main__":
    # datasets = ["biogrid", "celegans", "homo", "oceania", "sacchcere", "aps", "northamerica", "higgs","southamerica", "friendfeedtwitter", "friendfeed", "europe", "dblp", "asia", "amazon"]
    # datasets = ["southamerica"]

    # # south america
    # layers_to_keep = [17, 9, 21, 51]

    # # north america
    # #layers_to_keep = [80,59,26,66,73]

    # datasets = ["northamerica_26_59_66_73_80", "southamerica_9_21_17_50"]
    # datasets = ["southamerica_9_17_21_52"]
    # datasets = ["northamerica_0_10_26_59_66_73_80"]
    # datasets = ["northamerica_0_2_13_14_11_9"]
    # datasets = ["northamerica_0_2_14_11_9"]

    datasets = ["southamerica_2_9_17_21_52"]


    for dataset in datasets:
        main(dataset, layers_to_keep=None)