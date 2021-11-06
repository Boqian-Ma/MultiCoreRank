from multilayer_graph.multilayer_graph import MultilayerGraph


def test_remove_node(paper, dataset):
    multilayer_graph = MultilayerGraph(paper, dataset)

    multilayer_graph.remove_node(1)


    multilayer_graph.remove_node(2)

    print(multilayer_graph.adjacency_list)




test_remove_node("multilayer_layer_core_decomposition", "example")