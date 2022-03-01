import argparse
import math
from resource import error
import time
import os
from scipy.stats import spearmanr
import numpy as np

from multilayer_graph.multilayer_graph import MultilayerGraph
from utilities.print_file import PrintFile 
from core_decomposition.breadth_first_v3 import breadth_first as bfs


def main():
    data_set = "example"
    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)
    
    over_lap_influence_rank = multilayer_graph.overlap_degree_rank()
    
    _, lattice_influence = bfs(multilayer_graph, PrintFile(data_set), False, data_set)
    lattice_influence_rank = [item[1] for item in lattice_influence]

    # try:
    #     lattice_influence_rank = read_influence_nodes_ranking(multilayer_graph, data_set)
    # except ValueError or IOError:
    #     save_influence_ranking(multilayer_graph, data_set)
    #     lattice_influence_rank = read_influence_nodes_ranking(multilayer_graph, data_set)
    # print(over_lap_influence_rank)
    # print(lattice_influence_rank)

    coef, p = spearmanr(over_lap_influence_rank, lattice_influence_rank)

    print("Spearman coef: {}".format(coef))
    print("Time: {}".format(time.time()-start_time))

    with open("ranking/spearman_overlap_{}.txt".format(data_set), "w+") as file:
        file.write("Dataset: {}\nSpearman Coefficient: {}".format(data_set, coef))

if __name__ == "__main__":
    main()