from multilayer_graph.multilayer_graph import MultilayerGraph

import sys
import os

graph = MultilayerGraph(sys.argv[1])

print(graph.number_of_nodes)

# find assortavity
print(graph.pearson_correlation_coefficient())


