import argparse
from audioop import mul
import math
from resource import error
import time
import os


from multilayer_graph.multilayer_graph import MultilayerGraph

def update_list_solve_modular(sigma, s, N, L, update_to):
    for w in range(N * L):
        if (s - w) % N == 0:
            sigma[w] = update_to

def betweenness(multilayer_network):
    
    N = multilayer_network.number_of_nodes
    L = multilayer_network.number_of_layers

    c_b = [0 for i in range(N)] #line 1

    for s in range(N): # Line 2 for every start node
        S = [] #line 3
        P = [[] for _ in range(N * L)] #line 4, shortest distance from source to all nodes

        sigma = [0] * (N * L) # line 5 pt1
        update_list_solve_modular(sigma, s, N, L, 1) # line 5 pt2

        d = [-1] * (N * L) # line 6 pt1, initialise distance to -1
        update_list_solve_modular(d, s, N, L, 0) # line 6 pt2, distance to source node on all layer are 0

        # print(d)

        d_M = [-1] * (N)
        update_list_solve_modular(d_M, s, N, 1, 0) # line 7 pt2, distance to source node on all layer are 0

        v_order = [[] for _ in range(N)] #line 8, list of lists visiting order v_order[i] is the order
        Q = [] # Queue
        Q.append(s) # enqueue s

        while Q != []: # line 11
            v = Q.pop(0) #line 12, peak first in the queue

            # dequeue
            if v in S:  # prevent a loop back to the visited node and go into circles
                continue

            print("\nv = " + str(v))
            S.append(v) # line 13, add first in the queue to stack

            W = []  # Neighbours

            # if v != s:  # if we are not visiting the first node
            #     # find the list of v's neighbour on 1 layer
            #     layer = v / N # determine which layer v is in

            #     v_temp = v % N
            #     for node in multilayer_network.adjacency_list[v_temp + 1][layer]: # +1 because adjancy list is 1 indexed somehow smh...
            #         W.append((layer + 1) * node - 1) 
            
            # else:
                # print("s = v: {} ".format(s))
            
            # find a list of all neighbour on all level from adjancy list
            for layer in range(multilayer_network.number_of_layers):
                v_temp = v % N
                for node in multilayer_network.adjacency_list[v_temp + 1][layer]: # +1 because adjancy list is 1 indexed somehow smh...
                    W.append((layer + 1) * node - 1)

            


            print("Neighbours: {}".format(W))
            for w in W: # line 18
                # Visit all neighbours
                print("Visiting neighbour: {}".format(w))

                if d[w] < 0: # if w has not been visited
                    print("first time visiting: {}".format(w))

                    if w == 0:
                        print("bruh")
                        print(d)

                    Q.append(w) # add w neighbor to Queue
                    d[w] = d[v] + 1 # increase distance by 1 since w is v's neighbour
                    # Check for shortest path?
                    # mod_d_M = [i for i in range(N) if (w - i) % N == 0]
                    # for node in mod_d_M:
                    if d_M[w % N] < 0 or d_M[w % N] == d[w]:
                        d_M[w % N] = d[w]
                        v_order[w % N].append(w)

                if d[w] == d[v] + 1: # a shortest path from v to w is found
                    sigma[w] = sigma[w] + sigma[v]
                    P[w].append(v)
                    #print(P)
    
        print(Q)
        print(d)
        print(d_M)
        print(v_order)
        print(sigma)
        print(P)
        print(S)
        break
    return P

def main():

    # parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks')
    # parser.add_argument('d', help='dataset')

    # args = parser.parse_args()
    
    # e.g python main.py example i 0.9 5
    
    data_set = "small_sample"
    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage
    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    #print(multilayer_graph.adjacency_list)

    betweenness(multilayer_graph)


if __name__ == "__main__":
    main()