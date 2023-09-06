from __future__ import division
from collections import defaultdict
from array import array
import errno
from subroutines.core import core
from subroutines.commons import *
from utilities.time_measure import ExecutionTime
import os
import time
import numpy as np
from os import getcwd
from os.path import dirname
import math


def breadth_first(multilayer_graph, print_file, distinct_flag):
    # measures
    number_of_cores = 0
    number_of_computed_cores = 0

    # start of the algorithm
    execution_time = ExecutionTime()

    start_time = time.time()

    # create the vector of zeros from which start the computation
    start_vector = tuple([0] * multilayer_graph.number_of_layers)

    # dict of cores
    cores = {}
    cores[start_vector] = array('i', multilayer_graph.get_nodes())

    inner_most_cores = []

    # core [0]
    if print_file is not None and not distinct_flag:
        print_file.print_core(start_vector, array('i', multilayer_graph.get_nodes()))
    elif distinct_flag:
        cores[start_vector] = array('i', multilayer_graph.get_nodes())
    
    number_of_cores += 1

    # initialize the queue of vectors with the descendants of start_vector and the structure that for each vector saves its ancestor vectors
    vectors_queue = deque()
    ancestors = {}
    for index in multilayer_graph.layers_iterator:
        # start from root, find first children
        descendant_vector = build_descendant_vector(start_vector, index)
        # Add to queue
        vectors_queue.append(descendant_vector)
        # map to father set
        ancestors[descendant_vector] = [start_vector]

    # initialize the dictionary that for each vector counts the number of descendants in the queue
    descendants_count = defaultdict(int)

    # Initialise influence dict
    influence = {}
    for node in multilayer_graph.get_nodes():
        influence[node] = 1
        # influence[node] = 1/multilayer_graph.number_of_nodes

    level = 1

    # Influence ranking by level in lattice and by node
    inf_by_core_vector = {}
    inf_by_core_vector[start_vector] = {}
    for i in multilayer_graph.get_nodes():
        inf_by_core_vector[start_vector][i] = 1

    # father level cores to calculate influence
    father_level_cores = cores.copy()
    current_level_ancestors = ancestors.copy()


    while len(vectors_queue) > 0:
        # remove a vector from vectors_queue (FIFO policy BFS)
        vector = vectors_queue.popleft()
        
        # if the number 
        number_of_non_zero_indexes = len([index for index in vector if index > 0])
        number_of_ancestors = len(ancestors[vector])

        if number_of_non_zero_indexes == number_of_ancestors:
            
            # Calculate influence of the level above
            if sum(list(vector)) > level:   # whenever we go down a level in the core lattice
                print("in level {}".format(level))
                get_influence_v3(influence, multilayer_graph, level, cores, father_level_cores, inf_by_core_vector, start_vector, current_level_ancestors)
                inf_by_core_vector = update_influence_by_core_vector(cores, inf_by_core_vector, influence)
                # Cache father level cores
                father_level_cores = cores.copy()
                # Cache current level ancestors
                current_level_ancestors = ancestors.copy()
                level += 1
            
            ancestors_intersection = build_ancestors_intersection(ancestors[vector], cores, descendants_count, distinct_flag, multilayer_graph=multilayer_graph)
            
            # print(ancestors)

            # if the intersection of its ancestor cores is not empty
            if len(ancestors_intersection) > 0:
                # compute the core from it
                k_core = core(multilayer_graph, vector, ancestors_intersection)
                number_of_computed_cores += 1
            else:
                # delete its entry from ancestors and continue
                # print("print ancestors")
                # print(ancestors)
                del ancestors[vector]
                continue

            # if the core is not empty
            if len(k_core) > 0:
                # add the core to the dict of cores and increment the number of cores
                cores[vector] = k_core
                # update influce by level by node

                number_of_cores += 1
                # print("core found...")
                if print_file is not None and not distinct_flag:

                    # print(k_core)

                    print_file.print_core(vector, k_core)

                # compute its descendant vectors by plusing 1 on each element
                for index in multilayer_graph.layers_iterator:
                    descendant_vector = build_descendant_vector(vector, index)

                    try:
                        # update the list of the ancestors of the descendant vector
                        ancestors[descendant_vector].append(vector)
                    # if the descendant vector has not already been found
                    except KeyError:
                        # add the descendant vector to the queue
                        vectors_queue.append(descendant_vector)
                        ancestors[descendant_vector] = [vector]

                    # increment descendants_count
                    descendants_count[vector] += 1
        else:
            # for each ancestor of vector
            for ancestor in ancestors[vector]:
                # decrement its number of descendants
                decrement_descendants_count(ancestor, cores, descendants_count, distinct_flag)

        # delete vector's entry from ancestors after finding the core
        # only keep those directly above
        del ancestors[vector]
    # end of the algorithm

    execution_time.end_algorithm()
    end_time = time.time()

    print("Time taken: " + str(end_time-start_time))
    print_end_algorithm(execution_time.execution_time_seconds, number_of_cores, number_of_computed_cores)
    post_processing(cores, distinct_flag, print_file, multilayer_graph, influence)
    
    return influence, level - 1, number_of_cores

def count_node_apprence_in_father_level(cores, multilayer_graph):
    count = {}
    for v, c in cores.items():  
        for n in multilayer_graph.get_nodes():
            if n in c:
                if n not in count:
                    count[n] = 1
                else:
                    count[n] += 1
    return count

def update_influence_by_core_vector(cores, inf_by_core_vector, influence):
    '''
    Update inf_by_core_vector
    '''
    inf_by_core_vector = {}
    for vector, core in cores.items():
        if vector not in inf_by_core_vector:
            inf_by_core_vector[vector] = {}
        for node in core:
            inf_by_core_vector[vector][node] = influence[node]    
    return inf_by_core_vector

def calculate_average_core_influence(influence_dict):
    '''
    Find average influence of a core
    '''
    cardinality = len(influence_dict)

    total_influence = sum(influence_dict.values())

    return total_influence / cardinality

def find_father_core_vectors(cores, node):
    '''
    Find a list of core vectors with node in the core
    '''
    list = []
    for core_vector, nodes in cores.items():
        if node in nodes:
            list.append(core_vector)
    
    return list

def get_cores_with_node(node, cores, start_vector):
    """
    Find all cores with node in them
    """
    core_vectors = []
    for core_vector, core in cores.items():
        if core_vector == start_vector: continue

        if node in core:
            core_vectors.append(core_vector)
    return core_vectors

def normalize(influence, target=1.0):
    '''
    Normalise a dictionary
    '''
    raw = sum(influence.values())
    factor = target/raw
    for node, inf in influence.items():
        influence[node] = float(inf) * factor

def normalize_new(influence, nodes_to_normalise, target=1.0):
    '''
    Normalise the values of a dictionary
    '''
    # raw = sum(influence.values())

    raw = 0
    for node in nodes_to_normalise:
        raw += influence[node]

    # raw = len(nodes_to_normalise)

    print("\n\n influence = {}\nraw = {} \n".format(influence, raw))

    # find max influence from  lattice level above
    max = 0
    for node, inf in influence.items():
        if node not in nodes_to_normalise and inf > max:
            max = inf

    factor = target/raw

    for node, inf in influence.items():
        if node in nodes_to_normalise:
        # influence[node] = float(inf) * factor
            influence[node] = np.longfloat(inf) * factor + max


def get_influence_v3(influence, multilayer_graph, level, current_level_cores, father_level_cores, inf_by_core_vector, start_vector, current_level_ancestors):
    '''
    Main driver of finding influence in this graph
    '''
    # print("yeet")
    # print(current_level_ancestors)

    current_level_count = {}                  # Count how many times a node appeared in the level
    # Record number of appearence of each node on current lattice level

    for core_vector, core in current_level_cores.items():
        if core_vector == start_vector:
            continue
        for node in core:
            if node not in current_level_count:
                current_level_count[node] = 1
            else:
                current_level_count[node] += 1
    
    # Preserve influence
    temp_influence = influence.copy()

    # reset influence to zero if a node exists in current level
    for node in current_level_count.keys():
        influence[node] = 0

    # for nodes that are on the current level
    for node, _ in current_level_count.items():
        
        # find core vectors of cores that has node in them
        core_vectors_with_node = get_cores_with_node(node, current_level_cores, start_vector)

        # for every core vector with node in it
        for core_vector in core_vectors_with_node:
            # find its ancector core vectors
            ancestor_core_vectors = current_level_ancestors[core_vector]

            # for each of the ancestor core vectors
            for ancestor_core_vector in ancestor_core_vectors:
                # calculate average core influence
                average_father_core_avg_inf = calculate_average_core_influence(inf_by_core_vector[ancestor_core_vector])
                # if node == 1:
                    # print("before: core vector {}, ancestor vector {}, level {}, node {}, inf: {} temp: {} avg: {}".format(core_vector,ancestor_core_vector , level, node, influence[node], temp_influence[node], average_father_core_avg_inf))
                influence[node] += level * temp_influence[node] * average_father_core_avg_inf
                # influence[node] += (1/1+(math.exp(-level))) * temp_influence[node] * average_father_core_avg_inf

                # if node == 1:
                #     print("after: core vector {}, ancestor vector {}, level {}, node {}, inf: {} temp: {} avg: {}".format(core_vector,ancestor_core_vector , level, node, influence[node], temp_influence[node], average_father_core_avg_inf))

    # if level == 5:
    #     quit()

    if level == 1:
        del father_level_cores[start_vector]
        del current_level_cores[start_vector]

    # Normalise influence
    # nodes_to_normalise = current_level_count.keys()

    normalize(influence, target=multilayer_graph.modified_number_of_nodes)





