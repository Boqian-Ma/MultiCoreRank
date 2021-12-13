import math
from os import remove
from multilayer_graph.multilayer_graph import MultilayerGraph
import numpy as np
import pytest
import time

import matplotlib.pyplot as plt

from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_console import print_dataset_name, print_dataset_info, print_dataset_source
from utilities.print_file import PrintFile 


DATA_SET = "example"

def create_plots(multilayer_graph, plot_col, axs):

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist]

    print(pearson_flat_list)
    
    pearson_density = kde.gaussian_kde(pearson_flat_list)
    pearson_x = np.linspace(-2,10,300)
    pearson_y = pearson_density(pearson_x)

    # heat map
    im = axs[0, plot_col].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)
    plt.colorbar(im, ax=axs[0, plot_col])

    # pearson density function
    axs[1, plot_col].set_xlim(-1, 1)
    axs[1, plot_col].plot(pearson_x, pearson_y)



from scipy.stats import gaussian_kde, kde

def main():
    start_time = time.time()
    data_set = "aps"
    percentage = 0.1
    total_columns = 2

    multilayer_graph = MultilayerGraph(data_set)


    # find out how many graphs
    total_num_remove_nodes = percentage * multilayer_graph.number_of_nodes

    # -1 because the first column if the entire graph, then the next 5 are results of node removal
    remove_nodes_per_iteration = int(math.ceil(total_num_remove_nodes / (total_columns - 1)))
    print(remove_nodes_per_iteration)
    fig, axs = plt.subplots(2, total_columns)
    
    # Origional plot
    create_plots(multilayer_graph, 0, axs)

    
    for col in range(1, total_columns):

        # find influence
        influence = bfs(multilayer_graph, PrintFile(data_set), False, data_set)
        nodes_to_remove = [pair[0] for pair in influence[:remove_nodes_per_iteration]]

        # print(nodes_to_remove)
        # If no more nodes can be removed
        # if len(nodes_to_remove) < multilayer_graph.number_of_nodes:
        #     break

        # remove nodes
        multilayer_graph.remove_nodes(nodes_to_remove)

        # find new plots
        create_plots(multilayer_graph, col, axs)


    plt.show()

    print(time.time()-start_time)
    quit()
    # Create subplot



    

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist]
    
    pearson_density = kde.gaussian_kde(pearson_flat_list)
    pearson_x = np.linspace(-2,10,300)
    pearson_y = pearson_density(pearson_x)

    fig, axs = plt.subplots(2)
    fig.suptitle(data_set + " correlation heatmap and density")

    # heat map
    im = axs[0].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)
    plt.colorbar(im, ax=axs[0])

    # pearson density function
    axs[1].set_xlim(-1, 1)
    axs[1].plot(pearson_x, pearson_y)

    plt.show()


    # plt.hist(flat_list, range=[-1, 1])
    # plt.show()

    # quit()
    # # set number of nodes removed in each iteration
    # remove_num_nodes = int(math.ceil(multilayer_graph.number_of_nodes * percentage))
    # plt.imshow(coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    # plt.clim(-1, 1)
    # plt.colorbar()
    # plt.show() 

    quit()
    influence = bfs(multilayer_graph, PrintFile(data_set), False, data_set)

    remove_num_nodes = int(math.ceil(len(influence) * percentage))
    # print(influence)
    # print(len(influence) * percentage)
    # print(remove_num_nodes)

    nodes_to_remove = [pair[0] for pair in influence[:remove_num_nodes]]
    print(nodes_to_remove)
    #assert(multilayer_graph.pearson_correlation_coefficient(1, 1) == 1)



if __name__ == "__main__":
    main()