from multilayer_graph.multilayer_graph import MultilayerGraph
import numpy as np
import pytest

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



def test_pearson_correlation_coefficient():

    multilayer_graph = MultilayerGraph("europe")

    print(multilayer_graph.pearson_correlation_coefficient())

test_pearson_correlation_coefficient()