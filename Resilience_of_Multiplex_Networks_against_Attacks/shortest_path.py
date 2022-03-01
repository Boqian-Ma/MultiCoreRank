import argparse
import math
from resource import error
import time
import os

import numpy as np


from multilayer_graph.multilayer_graph import MultilayerGraph
from Resilience_of_Multiplex_Networks_against_Attacks.centrality.betweeness import betweenness # betweeness centrality

def main():

    parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks')
    parser.add_argument('d', help='dataset')
    parser.add_argument('m', help='method: "i"=iterative influence calculation, "o"=once off influence calculation', choices=["i", "o"])
    parser.add_argument('p', help='percentage of node removal', type=float, choices=np.arange(0.0, 1.0, 0.1))
    parser.add_argument('c', help='total columns displayed', type=int, choices=range(1, 6))
    args = parser.parse_args()

    # e.g python main.py example i 0.9 5
    
    data_set = args.d
    type = args.m
    percentage = args.p
    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage
    total_columns = args.c
    

    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    
    print(time.time()-start_time)

if __name__ == "__main__":
    main()