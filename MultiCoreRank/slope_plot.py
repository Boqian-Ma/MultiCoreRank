import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

from os import getcwd
from os.path import dirname

import pandas as pd
from scipy.optimize import curve_fit


font = {'size': 18}

matplotlib.rc('font', **font)

def main(file_name):

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

    size = array[0]
    for i in range(len(array) - 1):
        slope = array[i + 1] - array[i]
        slopes.append(abs(slope)/float(size))
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


def plot(influence_attack, random_attack, dataset, assortativity,  print_file):

    influence_x, influence_y = influence_attack
    random_x, random_y = random_attack

    _, ax = plt.subplots(figsize=(8, 8))

    # ax.step(influence_x , influence_y, '*', where='post', label="Sorted Attack", color="green")
    # ax.step(random_x , random_y, 'x', where='post', label="Random Attack", color="blue")

    ax.step(influence_x , influence_y, where='post', label="Sorted Attack", color="green")
    ax.step(random_x , random_y, where='post', label="Random Attack", color="blue")

    # # 1/x 
    k = calculate_k_value(influence_y, random_y)
    x = np.linspace(min(influence_x + random_x) + 0.01, max(influence_x + random_x), 400) 
    y = 400/x
    ax.plot(x, y, '--', label="y = k/x", color="red")
    ax.set_ylim([0, max(influence_y)])

    ax.grid(True)
    ax.legend(loc='upper right')

    # ax.set_title('Number of cores vs % of node removal in {}'.format(dataset))
    ax.set_xlabel('% of node removed')
    ax.set_ylabel('Number of cores')

    ax.set_title("{} - {}".format(dataset, assortativity))

    print_file.print_number_of_cores_plot(plt, dataset)

def take_average(arr1, arr2):
    res = []
    for i in range(len(arr1)):
        try:
            res.append((arr1[i]+arr2[i])/2)
        except:
            break
    return res

def max_from_indices(arr1, arr2):
    # Determine the length of the longer array
    max_len = max(len(arr1), len(arr2))
    
    # Create a new array to store the max values
    result = []
    
    # Iterate over the range of the longer array's length
    for i in range(max_len):
        # Check if an index exists in both arrays
        if i < len(arr1) and i < len(arr2):
            result.append(max(arr1[i], arr2[i]))
        # If the index doesn't exist in the first array, append from the second array
        elif i >= len(arr1):
            result.append(arr2[i])
        # If the index doesn't exist in the second array, append from the first array
        else:
            result.append(arr1[i])
    
    return result


# def fit_line():
    
def exponential_func(x, a, b):
    return a * np.exp(b * -x)

if __name__ == "__main__":

    # file_names = [("europe_0_100_1", "Europe"), 
    #                 ("northamerica_0_2_13_14_11_0_100_1", "North America"), 
    #                 ("southamerica_9_17_21_52_0_100_1", "South America"),
    #                 ("celegans_0_100_1", "C.Elegans"),
    #                 ("homo_0_100_1", "Homo"),
    #                 ("biogrid_0_100_1", "Biogrid")
    #             ]

    # file_names = [(sys.argv[1], " ".join(sys.argv[2:]))]

    # file_names = [("aarhus_0_100_1", "Aarhus", "Assortative"), ("celegans_0_100_1", "C.Elegans", "Assortative"), ("northamerica_33_0_100_1", "Airlines-NorthAmerica", "Disassortative"), \
    #               ("southamerica_13_0_100_1", "Airlines-SouthAmerica", "Disassortative"), ("europe_75_0_100_1", "Airlines-Europe", "Neutral"), ("asia_63_0_100_1", "Airlines-Asia","Neutral")]

    file_names = [("aarhus_0_100_1", "Aarhus", "Assortative"), ("celegans_0_100_1", "C.Elegans", "Assortative"), ("northamerica_33_0_100_1", "Airlines-NorthAmerica", "Disassortative"), \
                  ("southamerica_13_0_100_1", "Airlines-SouthAmerica", "Disassortative"), ("europe_75_0_100_1", "Airlines-Europe", "Neutral"), ("asia_63_0_100_1", "Airlines-Asia","Neutral")]


    assortative_data = [("aarhus_0_100_1", "Aarhus", "Assortative"), ("celegans_0_100_1", "C.Elegans", "Assortative")]

    disassortative_data = [("northamerica_33_0_100_1", "Airlines-NorthAmerica", "Disassortative"), ("southamerica_13_0_100_1", "Airlines-SouthAmerica", "Disassortative")]

    neutral_data = [("europe_75_0_100_1", "Airlines-Europe", "Neutral"), ("asia_63_0_100_1", "Airlines-Asia","Neutral")]
                  

    files = [("Assortative", assortative_data), ("Neutral", neutral_data), ("Disassortative", disassortative_data)]


    plt.rcParams['font.size'] = 60

    fig, axs = plt.subplots(2, 3, figsize=(60, 30))

    row = 0
    fig_count = 1

    col = 0
    for file in files:
        type = file[0]
        data = file[1]

        ranked_attack_data = []
    
        for pair in data:
            file_name = pair[0]
            dataset = pair[1]
            assortativity = pair[2]
            print_file = PrintFile(dataset)
            print(file_name)

            influence_attack_data, random_attack_data = main(file_name)

            influence_x, influence_y = influence_attack_data
            random_x, random_y = random_attack_data


            # calculate slope
            random = calculate_slopes(random_y)
            ranked = calculate_slopes(influence_y)

            ranked_attack_data.append(ranked)


            # spline = UnivariateSpline(x_smooth, avg_y)
            # y_smooth = spline(x_smooth)
            # axs[row][0].plot(x_smooth, y_smooth, color='blue', label='Smooth Line')

            axs[0][col].plot(influence_x[1:], ranked, label=dataset, linewidth=4)
            axs[0][col].set_title('Sorted Attack - {}'.format(type))
            id = col * 2 + 1
            axs[0][col].set_xlabel('% of node removed\n({})'.format(id), fontsize=70)
            axs[0][col].set_ylabel('Change in % of cores', fontsize=70)

            axs[1][col].plot(random_x[1:], random, label=dataset, linewidth=4)
            axs[1][col].set_title('Random Attack - {}'.format(type))     
            axs[1][col].set_xlabel('% of node removed\n({})'.format(id+1), fontsize=70)
            axs[1][col].set_ylabel('Change in % of cores', fontsize=70)

            axs[0][col].tick_params(axis='both', which='major', labelsize=70)
            axs[1][col].tick_params(axis='both', which='major', labelsize=70)

            fig_count = fig_count + 1

            # plt.title("Neutral")
            # fig.suptitle(type, fontsize=30)

        avg_y = max_from_indices(ranked_attack_data[0], ranked_attack_data[1])
        x_avg = [i for i in range(len(avg_y))]

        params, covariance = curve_fit(exponential_func, x_avg, avg_y)
        x_fit = np.linspace(min(x_avg), max(x_avg), 100)
        y_fit = exponential_func(x_fit, *params)


        # y_fit = [i+0.02 for i in y_fit]
        
        axs[0][col].plot(x_fit, y_fit, color='blue', marker="x", label='Exponential Fit', linewidth=6, markersize=25)
        axs[0][col].legend(loc='upper right', fontsize=60)
        axs[1][col].legend(loc='upper right', fontsize=60)
        # row += 1
        col += 1
    
    # plt.tick_params(axis='both', which='major', labelsize=70)
    plt.tight_layout()
    print_file.print_number_of_cores_plot(plt, "slopes")