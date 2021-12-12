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

#############################
# multilayer_graph.py tests #
#############################
def test_remove_node():
    multilayer_graph = MultilayerGraph(DATA_SET)

    multilayer_graph.remove_node(1)
    multilayer_graph.remove_node(2)

def test_get_connected_nodes():
    pass

def test_get_nodes():
    multilayer_graph = MultilayerGraph(DATA_SET)
    assert(multilayer_graph.get_nodes() == set([1, 2, 3, 4, 5, 6]))

    multilayer_graph.remove_node(1)
    assert(multilayer_graph.get_nodes() == set([2, 3, 4, 5, 6]))

    multilayer_graph.remove_node(2)
    assert(multilayer_graph.get_nodes() == set([3, 4, 5, 6]))

    multilayer_graph.remove_node(3)
    assert(multilayer_graph.get_nodes() == set([4, 5, 6]))

    multilayer_graph.remove_node(4)
    assert(multilayer_graph.get_nodes() == set([5, 6]))

    multilayer_graph.remove_node(5)
    assert(multilayer_graph.get_nodes() == set([]))

    multilayer_graph.remove_node(6)

    assert(multilayer_graph.get_nodes() == set([]))

def test_remove_nodes():
    multilayer_graph = MultilayerGraph(DATA_SET)

    assert(multilayer_graph.get_nodes() == set([1, 2, 3, 4, 5, 6]))

    multilayer_graph.remove_nodes([1, 2])
    assert(multilayer_graph.get_nodes() == set([3, 4, 5, 6]))

    multilayer_graph.remove_nodes([3, 4, 5])
    assert(multilayer_graph.get_nodes() == set([]))

def test_get_number_of_edges():
    multilayer_graph = MultilayerGraph(DATA_SET)

    assert(multilayer_graph.get_number_of_edges() == 17)

    multilayer_graph.remove_node(1)
    assert(multilayer_graph.get_number_of_edges() == 13)

    multilayer_graph.remove_nodes([2, 3, 4, 5, 6])
    assert(multilayer_graph.get_number_of_edges() == 0)


def test_example_pearson_coefficient():

    multilayer_graph = MultilayerGraph("example")

    true_layer_1 = [3, 5, 1, 3, 4, 2]
    true_layer_2 = [1, 5, 3, 1, 3, 3]

    test_layer_1, test_layer_2 = multilayer_graph.get_layer_node_degrees(0, 1)

    assert(true_layer_1 == test_layer_1)
    assert(true_layer_2 == test_layer_2)

    print(multilayer_graph.adjacency_list)


# TODO: need to consider isolated nodes in both layers, and discharge them in calculation process    

def test_pearson_correlation_coefficient():

    data_set = "aps"
    percentage = 0.1

    multilayer_graph = MultilayerGraph(data_set)
    coe_matrix = multilayer_graph.pearson_correlation_coefficient()

    # set number of nodes removed in each iteration
    remove_num_nodes = int(math.ceil(multilayer_graph.number_of_nodes * percentage))

    # Print
    plt.imshow(coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    plt.clim(-1, 1)
    plt.colorbar()
    plt.show() 


    while multilayer_graph.number_of_nodes > remove_num_nodes:

        # Calculate influence
        influence = bfs(multilayer_graph, PrintFile(data_set), False, data_set)

        # Remove nodes
        nodes_to_remove = [pair[0] for pair in influence[:remove_num_nodes]]

        



    # print(influence)
    # print(len(influence) * percentage)
    # print(remove_num_nodes)

    nodes_to_remove = [pair[0] for pair in influence[:remove_num_nodes]]
    print(nodes_to_remove)
    #assert(multilayer_graph.pearson_correlation_coefficient(1, 1) == 1)


def test_playground():

    multilayer_graph = MultilayerGraph("aps")

    # true_layer_1 = [3, 5, 1, 3, 4, 2]
    # true_layer_2 = [1, 5, 3, 1, 3, 3]

    # test_layer_1, test_layer_2 = multilayer_graph.get_layer_node_degrees(0, 1)

    # assert(true_layer_1 == test_layer_1)
    # assert(true_layer_2 == test_layer_2)

    # print(multilayer_graph.adjacency_list)

    # test duplicates 

    print(multilayer_graph.adjacency_list[82])


test_pearson_correlation_coefficient()
#test_example_pearson_coefficient()




'''

[
    [array('i'), array('i')], 
    [array('i', [2, 4, 5]), array('i', [2])], 
    [array('i', [1, 3, 4, 5, 6]), array('i', [1, 3, 4, 5, 6])], 
    [array('i', [2]), array('i', [2, 5, 6])], 
    [array('i', [1, 2, 5]), array('i', [2])], 
    [array('i', [1, 2, 4, 6]), array('i', [2, 3, 6])], 
    [array('i', [2, 5]), array('i', [2, 3, 5])]
]
'''