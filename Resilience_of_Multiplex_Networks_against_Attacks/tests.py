from multilayer_graph.multilayer_graph import MultilayerGraph


def test_remove_node(dataset):
    multilayer_graph = MultilayerGraph(dataset)
    multilayer_graph.remove_node(1)
    multilayer_graph.remove_node(2)
    print(multilayer_graph.adjacency_list)


def test_get_nodes(dataset):
    multilayer_graph = MultilayerGraph(dataset)
    assert(multilayer_graph.get_connected_nodes() == set([1, 2, 3, 4, 5, 6]))
    multilayer_graph.remove_node(1)
    assert(multilayer_graph.get_connected_nodes() == set([2, 3, 4, 5, 6]))




#test_remove_node("example")

test_get_nodes("example")

