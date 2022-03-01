from scipy.stats import spearmanr
from multilayer_graph.multilayer_graph import MultilayerGraph
from utilities.print_file import PrintFile 
from helpers import get_influence_node_tuples

import time

def sort_and_get_second_element(dict):
    '''
    Returns sorted key in a list
    '''
    return [int(i[1]) for i in dict.items()]

def main():
    data_set = "aarhus"

    #start_time = time.time()
    start_time = time.time()
    
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    # load or calculate influence
    print_file = PrintFile(data_set)
    ranked_influence = get_influence_node_tuples(multilayer_graph, print_file)

    ranked_influence = [i[1] for i in ranked_influence]

    print(ranked_influence)

    overlapping_degree = sort_and_get_second_element(multilayer_graph.overlap_degree_rank())
    eigenvector_centrality = sort_and_get_second_element(multilayer_graph.eigenvector_centrality())
    betweenness_centrality = sort_and_get_second_element(multilayer_graph.betweenness_centrality())
    closeness_centrality = sort_and_get_second_element(multilayer_graph.closeness_centrality())

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
    main()