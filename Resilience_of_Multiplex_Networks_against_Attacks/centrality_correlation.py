import argparse
from scipy.stats import spearmanr
from multilayer_graph.multilayer_graph import MultilayerGraph
from utilities.print_file import PrintFile 
from helpers import get_influence_node_tuples

import time

def sort_and_get_second_element(dict):
    '''
    Returns sorted key in a list
    '''
    return [i[1] for i in dict.items()]

def main():
    print("Calculating centrality")
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

    # print(ranked_influence)

    #print(multilayer_graph.eigenvector_centrality())
    print("Calculating Overlapping degree: {}".format(data_set))
    overlapping_degree = sort_and_get_second_element(multilayer_graph.overlap_degree_rank())
    print("Calculating Eigenvector centrality")
    eigenvector_centrality = sort_and_get_second_element(multilayer_graph.eigenvector_centrality())
    print("Calculating Betweenness centrality")
    betweenness_centrality = sort_and_get_second_element(multilayer_graph.betweenness_centrality())
    print("Calculating Closeness centrality")
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