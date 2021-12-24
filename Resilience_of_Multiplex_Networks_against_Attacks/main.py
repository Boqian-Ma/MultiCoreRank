import math
from os import remove
from multilayer_graph.multilayer_graph import MultilayerGraph
import numpy as np
import pytest
import time
import pandas as pd


import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib.ticker import PercentFormatter


import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kde
from sklearn.neighbors import KernelDensity

from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_console import print_dataset_name, print_dataset_info, print_dataset_source
from utilities.print_file import PrintFile 

def create_plots(multilayer_graph, plot_col, axs, density_y_lim=None):
    '''
    Create density distribution plot and heatmap
    Add to the existing set of plots
    '''

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist]
    
    # density distribution
    pearson_density = kde.gaussian_kde(pearson_flat_list)
    pearson_x = np.linspace(-2,10,300)
    pearson_y = pearson_density(pearson_x)

    # heat map
    im = axs[0, plot_col].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)
    plt.colorbar(im, ax=axs[0, plot_col])

    # pearson density function x axis limit
    axs[1, plot_col].set_xlim(-1, 1)

    if density_y_lim is not None:
        axs[1, plot_col].set_ylim(density_y_lim[0], density_y_lim[1])
        sns.distplot(pearson_flat_list, axlabel="Pearson Correlation Coefficient", kde=True, ax=axs[1, plot_col])

        #axs[1, plot_col].plot(pearson_x, pearson_y)
    else:
        # axs[1, plot_col].plot(pearson_x, pearson_y)
        sns.distplot(pearson_flat_list, axlabel="Pearson Correlation Coefficient", kde=True, ax=axs[1, plot_col])

    # histogram
    axs[2, plot_col].set_xlim(-1, 1)
    axs[2, plot_col].hist(pearson_flat_list)


def create_plot(multilayer_graph, axs):

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist]
    
    # print(len(pearson_flat_list))    

    # heat map
    im = axs[0].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)
    plt.colorbar(im, ax=axs[0])

    
    # sns.barplot(ax=axs[1], x=charmander.index, y=charmander.values)
    # sns.distplot(pearson_flat_list, bins=0.5, axlabel="Pearson Correlation Coefficient", norm_hist=True, hist=False, kde=True, ax=axs[1])
    sns.kdeplot(pearson_flat_list, shade=True, ax=axs[1])
    

    ###########
    # pearson density function
    # density distribution
    # pearson_density = kde.gaussian_kde(pearson_flat_list)
    # pearson_x = np.linspace(-1, 1, 300)
    # pearson_y = pearson_density(pearson_x)
    # weights = np.ones_like(np.array(pearson_y))/float(len(np.array(pearson_y)))
    # axs[1].plot(pearson_x, pearson_y, weights=weights)
    ###########

    axs[1].set_xlim(-1, 1)
    axs[1].set_ylim(0, 3)


    # Normalised histogram
    weights = np.ones_like(pearson_flat_list) / float(len(pearson_flat_list))   # calculate weights
    hist = axs[2].hist(pearson_flat_list, bins=100, weights=weights)
    axs[2].set_xlim(-1, 1)

    density, bins, patches = hist
    widths = bins[1:] - bins[:-1]
    #assert((density * widths).sum() == 1.0)
    print((density * widths).sum())
    
def new_axis(arr):
    res = []
    for i in arr:
        res.append([i])
    return np.array(res)

def main():
    start_time = time.time()
    data_set = "example"
    percentage = 0.1
    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage
    total_columns = 1

    multilayer_graph = MultilayerGraph(data_set)

    print("loading time: " + str(time.time()-start_time))

    # find out how many graphs

    fig, axs = plt.subplots(3, total_columns, figsize=(20, 20))

    #fig, axs = plt.subplots(2, total_columns, figsize=(20, 12))

    # Plotting multiple
    if total_columns > 1:
        # first column

        axs[0, 0].set_title("Full network")
        create_plots(multilayer_graph, 0, axs)

        # First plot
        density_y_lim = axs[1, 0].get_ylim()
        
        print(density_y_lim)

        # -1 because the first column if the entire graph, then the next 5 are results of node removal
        total_num_remove_nodes = math.floor(percentage * multilayer_graph.number_of_nodes)
        remove_nodes_per_iteration = int(math.ceil(total_num_remove_nodes / (total_columns - 1)))

        fig.suptitle('Dataset: {}, Total nodes: {} \nRemoval percentage: {} Total remove nodes: {}, Per iteration removal: {}'.format(data_set, multilayer_graph.number_of_nodes, percentage ,total_num_remove_nodes, remove_nodes_per_iteration), fontsize=16)
        print("First iteration")
        
        for col in range(1, total_columns):
            # find influence
            influence = bfs(multilayer_graph, PrintFile(data_set), False, data_set)

            print("iteration {} done....".format(col))

            nodes_to_remove = [pair[0] for pair in influence[:remove_nodes_per_iteration]]

            # remove nodes
            multilayer_graph.remove_nodes(nodes_to_remove)

            # find new plots
            create_plots(multilayer_graph, col, axs, density_y_lim=density_y_lim)
            axs[0, col].set_title("Iteration {}, Remaining nodes: {}".format(col, multilayer_graph.modified_number_of_nodes))

    else:
        # ploting only 1 column
        fig.suptitle('Dataset: {}'.format(data_set), fontsize=16)

        create_plot(multilayer_graph, axs)

    plt.savefig("figures/" + data_set + "_" + str(total_columns) + "_" + str(percentage) +".png", format="png")

    print(time.time()-start_time)
    quit()


if __name__ == "__main__":
    main()