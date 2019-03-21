from import_data  import data_load, GroupLst, writeResults
import math
import random
import time
import os
import matplotlib.pyplot as plt
import argparse
from pympler import classtracker
from operator import itemgetter

def con_lin_plmt(n, C, P):
	print("hi")

def placement(connect_lst, P, place_params):
	print("nbrs")
	print(connect_lst.cells.values())
	tups = [(x.num, len(x.nbrs)) for x in connect_lst.cells.values()]
	print([(x.num, len(x.nbrs)) for x in connect_lst.cells.values()])
	print("sorted")
	print(sorted(tups, key=itemgetter(1), reverse=True))



def main():
	# Needed for the specified command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', type=str, default="Example-Netlists/1",
	                  help='What is the input filename?')
	parser.add_argument('-o', type=str, default="Results/R_1",
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

	# Figure out the min grid size
	x = 1
	while x**2 < connect_lst.num_cells:
		x += 1
	print("x = {}".format(x))

	# cell num, vacant,
	P = [[(0, True, False ) for i in range(x)] for j in range(x)]  # Instantiate a 2-D list matrix

	#nmpt = int(0.0542*(num_cells*num_nets)**0.4921+10)  # Heuristic to scale number of iterations

	place_params = {}

	# Execute placement engine.
	solution, cost = placement(connect_lst, P, place_params)

	print("num moves per T {}".format(nmpt))
	# Write the solution to output file
	writeResults(solution, cost, args.o)


if __name__ == "__main__":
	main()
