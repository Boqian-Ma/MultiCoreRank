import math
from os import remove
from multilayer_graph.multilayer_graph import MultilayerGraph
import numpy as np
import pytest

import matplotlib.pyplot as plt

from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_console import print_dataset_name, print_dataset_info, print_dataset_source
from utilities.print_file import PrintFile 


DATA_SET = "example"

def create_plots():
    pass


from scipy.stats import gaussian_kde, kde

def main():

    data_set = "aps"
    percentage = 0.1

    multilayer_graph = MultilayerGraph(data_set)


    # find out how many graphs
    num_graphs = percentage * multilayer_graph.number_of_nodes
    num_rows = math.ceil(math.sqrt(num_graphs))

    '''
    fig, axs = plt.subplots(num_rows, num_rows)

    
    # Whole graph
    coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    axs.imshow(coe_matrix, cmap='Greens', origin='lower', interpolation='none')

    for i in range(num_rows):

        for j in range(num_rows):
            # pearson Correlation matrix
            coe_matrix = multilayer_graph.pearson_correlation_coefficient()
            axs.imshow(coe_matrix, cmap='Greens', origin='lower', interpolation='none')

            pass
    
    '''
    # Create subplot

    coe_matrix = multilayer_graph.pearson_correlation_coefficient()

    flat_list = [item for sublist in coe_matrix for item in sublist]
    
    density = kde.gaussian_kde(flat_list)
    x = np.linspace(-2,10,300)
    y = density(x)

    plt.plot(x, y)
    plt.title("Density Plot of the data")
    plt.show()


    # plt.hist(flat_list, range=[-1, 1])
    # plt.show()

    quit()
    # set number of nodes removed in each iteration
    remove_num_nodes = int(math.ceil(multilayer_graph.number_of_nodes * percentage))
    plt.imshow(coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    plt.clim(-1, 1)
    plt.colorbar()
    plt.show() 

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