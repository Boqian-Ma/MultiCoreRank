import argparse
from cProfile import label

import numpy as np
from scipy.stats import spearmanr
from multilayer_graph.multilayer_graph import MultilayerGraph
from utilities.print_file import PrintFile 
from helpers import get_influence_node_tuples
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sort_and_get_second_element(dict):
    '''
    Returns sorted key in a list
    '''
    return [i[1] for i in dict.items()]

def main():
    print("Calculating centrality heatmap")
    parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks: centrality methods correlation')
    parser.add_argument('d', help='dataset')
    args = parser.parse_args()
    data_set = args.d

    start_time = time.time()
    
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    # load or calculate influence
    print_file = PrintFile(data_set)
    ranked_influence = get_influence_node_tuples(multilayer_graph, print_file)

    # print(ranked_influence)

    ranked_influence = [i[1] for i in ranked_influence]

    #print(multilayer_graph.eigenvector_centrality())
    print("Calculating Overlapping degree: {}".format(data_set))
    overlapping_degree = sort_and_get_second_element(multilayer_graph.overlap_degree_rank())
    print("Calculating Eigenvector centrality")
    eigenvector_centrality = sort_and_get_second_element(multilayer_graph.eigenvector_centrality())
    print("Calculating Betweenness centrality")
    betweenness_centrality = sort_and_get_second_element(multilayer_graph.betweenness_centrality())
    print("Calculating Closeness centrality")
    closeness_centrality = sort_and_get_second_element(multilayer_graph.closeness_centrality())


    methods = ["Overlap", "Eigenvector", "Betweenness", "Closeness", "Ours"] # order of evaluation

    # initialise matrix

    row_len = len(methods)
    corr_matrix = [None for _ in range(row_len)]
    
    rank_list = [overlapping_degree, eigenvector_centrality, betweenness_centrality, closeness_centrality, ranked_influence]

    for i in range(len(methods)):
        corr_matrix[i] = calculate_coefs(rank_list, i, row_len)
        # elif methods[i] == "eigenvector":
        #     corr_matrix[i] = calculate_coefs(eigenvector_centrality, [overlapping_degree, betweenness_centrality, closeness_centrality, ranked_influence], i, row_len)
        
        # elif methods[i] == "betweenness":
        #     corr_matrix[i] = calculate_coefs(betweenness_centrality, [overlapping_degree, eigenvector_centrality, closeness_centrality, ranked_influence], i, row_len)
        # elif methods[i] == "closeness":
        #     coefs = calculate_coefs(closeness_centrality, [overlapping_degree, eigenvector_centrality, betweenness_centrality, ranked_influence], i, row_len)
        #     corr_matrix[i] = coefs
            
        # elif methods[i] == "ours":
        #     coefs = calculate_coefs(ranked_influence, [overlapping_degree, eigenvector_centrality, betweenness_centrality, closeness_centrality], i, row_len)
        #     corr_matrix[i] = coefs

    # overlap = "Overlaping Centrality Spearman coef: {}\n".format(overlap_coef)
    # eigen = "Eigenvector Centrality Spearman coef: {}\n".format(eigen_coef)
    # betweenness = "Betweenness Centrality Spearman coef: {}\n".format(betweenness_coef)
    # closeness = "Closeness Centrality Spearman coef: {}\n".format(closeness_coef)

    # total_time = "Time: {}\n".format(time.time() - start_time)
    print(corr_matrix)

    # coefs = [data_set + "\n", total_time, overlap, eigen, betweenness, closeness]

    # print_file.print_correlation(coefs)

    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix, cmap='Greens', origin='lower', interpolation='none')

    im.set_clim(0, 1)
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_yticklabels(methods)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    ax.set_title("{} Correlation".format(data_set))

    print_file.print_correlation_heatmap(plt)


def calculate_coefs(rank_list, source_index, length):
    '''
    rank_list - list of list of numbers 
    '''

    res = [None for _ in range(length)]
    # print(res)
    res[source_index] = 1

    print(res)

    for i in range(length):
        if i == source_index:
            # skip itself
            continue

        coef, _ = spearmanr(rank_list[source_index], rank_list[i])
        res[i] = coef

        print(res)

    return res

if __name__ == "__main__":
    main()