'''
Find node influence ranking of our method
'''

import argparse
# from scipy.stats import spearmanr
from multilayer_graph.multilayer_graph import MultilayerGraph
from utilities.print_file import PrintFile 
from helpers import get_influence_node_tuples, get_influence_node_tuples_new

import time

def main():
    print("Calculating centrality")
    parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks: influence ranking')
    parser.add_argument('d', help='dataset')
    args = parser.parse_args()
    data_set = args.d
    multilayer_graph = MultilayerGraph(data_set)
    print_file = PrintFile(data_set)
    _ = get_influence_node_tuples(multilayer_graph, print_file)

if __name__ == "__main__":
    main()