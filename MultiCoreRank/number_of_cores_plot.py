import argparse
import math
from resource import error
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

from os import getcwd
from os.path import dirname
from helpers import create_plot, create_plots, get_influence_node_ranking, get_influence, get_influence_node_tuples, get_influence_node_tuples_new

import pandas as pd

font = {'size': 25}

matplotlib.rc('font', **font)
   

def main(file_name):
    # retrive data
    # plot
    # set axis /home/z5260890/Resilience_of_Multiplex_Networks_against_Attacks/output/inner_most_cores_map/europe_0_100_1.txt
    # file_name = "europe_0_100_1"

    path = dirname(getcwd()) + "/Resilience_of_Multiplex_Networks_against_Attacks/output/inner_most_cores_map_fast/{}.txt".format(file_name)
    random_path = dirname(getcwd()) + "/Resilience_of_Multiplex_Networks_against_Attacks/output/inner_most_cores_map_random_attack/{}.txt".format(file_name)


    with open(path, "r") as file:
        data = file.read()

    with open(random_path, "r") as file:
        random_attack_data = file.read()

    (x, y) = process_data(data)
    (random_x, random_y) = process_data(random_attack_data)

    return (x, y), (random_x, random_y)

def calculate_slopes(array):
    slopes = []
    for i in range(len(array) - 1):
        slope = array[i + 1] - array[i]
        slopes.append(slope)
    return slopes

def process_data(s):
    l = s.split("\n")
    x = []
    y = []
    for item in l:
        a, _ , b = item.split(" ")
        x.append(int(a))
        y.append(int(b))

    return x, y

def calculate_k_value(a, b):
    if a[0] == b[0]:
        if a[1] < b[1]:
            a, b = b, a
        p = 0
        flag = True
        for i in range(len(a)):
            if a[i] <= b[i] and flag is True:
                flag = False
            else:
                if a[i] <= b[i]:
                    break
            p += 1
    else:
        if a[0] < b[0]:
            a, b = b, a
        p = 0
        for i in range(len(a)):
            if a[i] <= b[i]:
                break
            p += 1

    k1 = a[p] - a[p-1]
    k2 = b[p] - b[p-1]

    c1 = a[p-1] - (a[p] - a[p-1]) * (p-1)
    c2 = b[p-1] - (b[p] - b[p-1]) * (p-1)

    x = (c1 - c2) / (k2 - k1)
    y = (c1 * k2 - c2 * k1) / (k2 - k1)
    
    return x * y


def plot(influence_attack, random_attack, dataset, assortativity,  print_file, id):
    influence_x, influence_y = influence_attack
    random_x, random_y = random_attack

    _, ax = plt.subplots(figsize=(10, 8))

    # ax.step(influence_x , influence_y, '*', where='post', label="Sorted Attack", color="green")
    # ax.step(random_x , random_y, 'x', where='post', label="Random Attack", color="blue")

    ax.step(influence_x , influence_y, where='post', label="Sorted Attack", color="green", linewidth=4)
    ax.step(random_x , random_y, where='post', label="Random Attack", color="blue", linewidth=4)

    # # 1/x 
    # k = calculate_k_value(influence_y, random_y)
    # x = np.linspace(min(influence_x + random_x) + 0.01, max(influence_x + random_x), 400) 
    # y = 400/x
    # ax.plot(x, y, '--', label="y = k/x", color="red")
    # ax.set_ylim([0, max(influence_y)])

    ax.grid(True)
    ax.legend(loc='upper right')

    # ax.set_title('Number of cores vs % of node removal in {}'.format(dataset))
    ax.set_xlabel('% of node removed\n({})'.format(id))
    ax.set_ylabel('Number of cores Remaining')

    ax.set_title("{} - {}".format(dataset, assortativity))

    plt.tight_layout()


    

    print_file.print_number_of_cores_plot(plt, dataset)


if __name__ == "__main__":

    # file_names = [("europe_0_100_1", "Europe"), 
    #                 ("northamerica_0_2_13_14_11_0_100_1", "North America"), 
    #                 ("southamerica_9_17_21_52_0_100_1", "South America"),
    #                 ("celegans_0_100_1", "C.Elegans"),
    #                 ("homo_0_100_1", "Homo"),
    #                 ("biogrid_0_100_1", "Biogrid")
    #             ]

    # file_names = [(sys.argv[1], " ".join(sys.argv[2:]))]

    file_names = [("aarhus_0_100_1", "Aarhus", "Assortative"), ("celegans_0_100_1", "C.Elegans", "Assortative"), ("europe_75_0_100_1", "Airlines-Europe", "Neutral"), ("asia_63_0_100_1", "Airlines-Asia","Neutral"), \
                  ("northamerica_33_0_100_1", "Airlines-NorthAmerica", "Disassortative"), \
                  ("southamerica_13_0_100_1", "Airlines-SouthAmerica", "Disassortative"), ]

    # file_names = [("celegans_0_100_1", "C.Elegans", "Assortative")]

    # file_names = [("fao_trade_multiplex_3_0_100_1", "FAO Trade Network", "Assortative")]

    count = 1
    for pair in file_names:
        file_name = pair[0]
        dataset = pair[1]
        assortativity = pair[2]
        print_file = PrintFile(dataset)
        print(file_name)
        influence_attack_data, random_attack_data = main(file_name)
        plot(influence_attack_data, random_attack_data, dataset, assortativity, print_file, count)
        count+=1
        
