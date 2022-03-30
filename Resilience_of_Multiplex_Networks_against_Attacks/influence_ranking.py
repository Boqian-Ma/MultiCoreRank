import argparse
# from scipy.stats import spearmanr
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

    # start_time = time.time()
    
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    # load or calculate influence
    print_file = PrintFile(data_set)
    _ = get_influence_node_tuples(multilayer_graph, print_file)

if __name__ == "__main__":
    main()