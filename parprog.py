from data_handler  import data_load, GroupLst, writeResults
import math
import random
import time
import os
import matplotlib.pyplot as plt
import argparse
from pympler import classtracker

def acceptMove(delCost, T, num_cells):
    #Logic to accept a move
    if delCost < 0:
        #If cost improves, accept
        return 0, True
    else:
        #Compute the boltz constant, scaled by the number of cells.
        boltz = math.exp((-(delCost*num_cells))/(T))
        r = random.uniform(0, 1)  # Pick a random number between 0 and 1.
        #Probabilstically determine if the move should be accepted.
        if r < boltz:
            return boltz, True
        else:
            return boltz, False

def coolDown(T, params):
    # Reduce T by a factor of Trate
    return T * params["Trate"]


def simulatedAnnealing(graph, num_cells, params, fn):
    cltr = classtracker.ClassTracker()
    cltr.track_class(GroupLst)  # Follow the memory size of the graph data struct.

    # Instantiate GroupLst objects
    curSol, nextSol = GroupLst(graph, num_cells), GroupLst(graph, num_cells)
    bestSol, bestCost = list(curSol.V), curSol.cost

    T = params["T0"]  # Initial Temp

    # Initialize graphing variables
    last_time, total_time = time.time(), time.time()
    iter_cntr,tmp_cntr = 0, 0
    iter_arr, temp_arr = [], []
    best_cutset_arr, cutset_arr, t2_arr, boltz_arr, nam_arr = [], [], [], [], []
    print_skip = 10

    while T > params["Tfreeze"]:
        #While the temperature is above Tfreeze,

        num_accepted_moves = 0  # reset accepted moves counter

        for i in range(params["num_moves_per_T"]):
            # For the number of moves per T,
            # copy over the curSol into the nextSol.
            nextSol.V, nextSol.cost = list(curSol.V), curSol.cost
            nextSol.perturb()  # Swap two random cells and update the cost.
            delCost = nextSol.cost - curSol.cost
            boltz, accept = acceptMove(delCost, T, num_cells)
            if accept:
                #If the move is accepted, copy nextSol back into curSol
                curSol.V, curSol.cost = list(nextSol.V), nextSol.cost
                num_accepted_moves += 1

            if curSol.cost < bestCost:
                # If curSol is the best solution seen so far, upate bestSol.
                bestSol, bestCost = list(curSol.V), curSol.cost


            iter_cntr += 1

        # Update graph data. Only once per temp to save space
        iter_arr.append(iter_cntr)
        t2_arr.append(T)
        temp_arr.append(T)
        cutset_arr.append(curSol.cost)
        best_cutset_arr.append(bestCost)
        nam_arr.append(num_accepted_moves)
        boltz_arr.append(boltz)
        tmp_cntr += 1

        if tmp_cntr % print_skip == 0:
            # Give output every print_skip temp updates
            print("{} T: {} curCost: {}, num_accepted_moves: {}".format(tmp_cntr, T, curSol.cost, num_accepted_moves))
            print('T step took {:0.3f} seconds'.format((time.time()-last_time)/print_skip))
            last_time = time.time()

        T = coolDown(T, params)

    # Plot the requisite graphs
    fig, ax = plt.subplots()
    ax.plot(temp_arr, cutset_arr, label="Cutset Size")
    ax.plot(temp_arr, best_cutset_arr, label="Best Cutset Size")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Cutset Size")
    ax.set_title("Cutset Size vs. Temperature")
    ax.legend()
    fig.set_size_inches(6, 4)
    fig.savefig("Images/{}_CS_v_T.png".format(fn), dpi=200)

    fig, ax = plt.subplots()
    ax.plot(iter_arr, t2_arr)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Temperature")
    ax.set_title("Temperature vs. Iteration")
    fig.set_size_inches(6, 4)
    fig.savefig("Images/{}_T_v_I.png".format(fn), dpi=200)

    fig, ax = plt.subplots()
    ax.plot(iter_arr, boltz_arr)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Boltz")
    ax.set_title("Boltz vs. Iteration")
    fig.set_size_inches(6, 4)
    fig.savefig("Images/{}_B_v_I.png".format(fn), dpi=200)

    fig, ax = plt.subplots()
    ax.plot(temp_arr, nam_arr)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Number of Accepted Moves")
    ax.set_title("Number of Accepted Moves vs. Temperature")
    fig.set_size_inches(6, 4)
    fig.savefig("Images/{}_NAM_v_T.png".format(fn), dpi=200)

    print("\n\nBest Cost: {}, temp iteration {}".format(bestCost, tmp_cntr))
    print("Total Time {:0.3f} seconds".format(time.time()-total_time))

    cltr.create_snapshot()
    cltr.stats.print_summary()  #display the size of the instatiated GroupLst data structs

    return bestSol, bestCost


def main():
    # Needed for the specified command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default="1",
                        help='What is the input filename?')
    parser.add_argument('-o', type=str, default="R_1",
                        help='What is the output filename?')
    args = parser.parse_args()

    # generate the output file directories if the don't exist
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists("Images"):
        os.makedirs("Images")

    # Load the input data and generate the graph
    connect_lst = data_load(args.i)
    print("Graph Created")

    #nmpt = int(0.0542*(num_cells*num_nets)**0.4921+10)  # Heuristic to scale number of iterations

    #params = {"T0": 10000, "Tfreeze": 0.01, "num_moves_per_T": nmpt, "Trate":0.99}  # k=1, so it won't be considered

    # Execute placement engine.
    solution, cost = placement(connect_lst)

    print("num moves per T {}".format(nmpt))
    # Write the solution to output file
    writeResults(solution, cost, args.o)


if __name__ == "__main__":
    main()
