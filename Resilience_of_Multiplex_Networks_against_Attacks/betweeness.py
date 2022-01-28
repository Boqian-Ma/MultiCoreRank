import argparse
from audioop import mul
import math
from resource import error
import time
import os

import numpy as np

from multilayer_graph.multilayer_graph import MultilayerGraph

def update_list_solve_modular(sigma, s, N, L, update_to):
    for w in range(N * L):
        if (s - w) % N == 0:
            sigma[w] = update_to

def betweenness(multilayer_network):
    
    N = multilayer_network.number_of_nodes
    L = multilayer_network.number_of_layers

    c_b = [0 for i in range(N)] #line 2

    big_s = []

    for s in range(N): # Line 2
        S = [] #line 3
        P = [[] for _ in range(N * L)] #line 4, shortest distance from source to all nodes

        sigma = [0] * (N * L) # line 5 pt1
        update_list_solve_modular(sigma, s, N, L, 1) # line 5 pt2

        d = [-1] * (N * L) # line 6 pt1, initialise distance to -1
        update_list_solve_modular(d, s, N, L, 0) # line 6 pt2, distance to source node on all layer are 0

        d_M = [-1] * (N * L)
        update_list_solve_modular(d_M, s, N, L, 0) # line 7 pt2, distance to source node on all layer are 0

        v_order = [[] for _ in range(N)] #line 8, list of lists

        Q = []
        Q.append(s) # enqueue s

        
        print(Q)

        while Q != []: # line 11
            
            v = Q[0]    #line 12, peak first in the queue

            Q.pop(0)    # dequeue

            S.append(v) # line 13, add first in the queue to stack

            # if v != s:  # if first in queue is not the same as the node we are currently exploring
            #     # find the list of v's neighbour
            #     pass
            # else:
            
            # find a list of all neighbour on all level from adjancy list
            W = []
            for layer in range(multilayer_network.number_of_layers):
                for node in multilayer_network.adjacency_list[v + 1][layer]: # +1 because adjancy list is 1 indexed somehow smh...
                    W.append((layer + 1) * node - 1)

            print("Neighbours: {}".format(W))
            for w in W:
                # Visit all neighbours
                print("Visiting neighbour: {}".format(w))

                if d[w] < 0:
                    Q.append(w) # add w to Queue
                    d[w] = d[v] + 1 # increase distance by 1

                    # Check for shortest path?
                    if d_M[w % N] < 0 or d_M[w % N] == d[w]:
                        d_M[w % N] = d[w]
                        v_order[w % N].append(w)

                if d[w] == d[v] + 1: # a shortest path from v to w is found
                    P[w].append(v)

        big_s.append(P)

    return big_s
def main():

    # parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks')
    # parser.add_argument('d', help='dataset')

    # args = parser.parse_args()
    
    # e.g python main.py example i 0.9 5
    
    data_set = "example"

    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage


    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    print(betweenness(multilayer_graph))


if __name__ == "__main__":
    main()