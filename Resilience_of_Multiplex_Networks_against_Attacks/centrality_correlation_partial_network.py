import argparse
from audioop import mul
import math
from resource import error
import time
import os
from scipy.stats import spearmanr

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

def sort_and_get_second_element(dict):
    '''
    Returns sorted key in a list
    '''
    return [i[1] for i in dict.items()]

def main(dataset, layers_to_keep=None):
    # e.g python main.py example i 0.9 5
    data_set = dataset

    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage

    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    if layers_to_keep:
        multilayer_graph.keep_layers(layers_to_keep)

    print_file = PrintFile(data_set)

    influence = bfs(multilayer_graph, print_file, False)
    influence = [(k, v) for k, v in influence.items()]

    ranked_influence = [i[1] for i in ranked_influence]

    # print(ranked_influence)

    #print(multilayer_graph.eigenvector_centrality())

    print("Calculating Overlapping degree: {}".format(data_set))
    overlapping_degree = sort_and_get_second_element(multilayer_graph.overlap_degree_rank())
    print("Calculating Eigenvector centrality")
    eigenvector_centrality = sort_and_get_second_element(multilayer_graph.eigenvector_centrality())

    print("Calculating Betweenness centrality")
    betweenness_centrality = sort_and_get_second_element(multilayer_graph.betweenness_centrality_projection())
    print("Calculating Closeness centrality")
    closeness_centrality = sort_and_get_second_element(multilayer_graph.closeness_centrality_projection())

    overlap_coef, _ = spearmanr(ranked_influence, overlapping_degree)
    eigen_coef, _ = spearmanr(ranked_influence, eigenvector_centrality)
    betweenness_coef, _ = spearmanr(ranked_influence, betweenness_centrality)
    closeness_coef, _ = spearmanr(ranked_influence, closeness_centrality)

    overlap = "Overlaping Centrality Spearman coef: {}\n".format(overlap_coef)
    eigen = "Eigenvector Centrality Spearman coef: {}\n".format(eigen_coef)
    betweenness = "Betweenness Centrality Spearman coef: {}\n".format(betweenness_coef)
    closeness = "Closeness Centrality Spearman coef: {}\n".format(closeness_coef)

    total_time = "Time: {}\n".format(time.time() - start_time)

    coefs = [data_set + "\n", total_time, overlap, eigen, betweenness, closeness]

    print_file.print_correlation(coefs)


if __name__ == "__main__":
    # datasets = ["biogrid", "celegans", "homo", "oceania", "sacchcere", "aps", "northamerica", "higgs","southamerica", "friendfeedtwitter", "friendfeed", "europe", "dblp", "asia", "amazon"]
    datasets = ["southamerica"]

    # south america
    layers_to_keep = [17, 9, 21, 50]

    # north america
    #layers_to_keep = [80,59,26,66,73]

    for dataset in datasets:
        main(dataset, layers_to_keep=layers_to_keep)