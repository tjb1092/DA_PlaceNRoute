from import_data  import data_load
from magic import magic
from placement import placement
from routing import routing
import time
import os
import matplotlib.pyplot as plt
import argparse


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

	# Perform Placement
	place_params = {"is2D":True, "iteration_count": 100, "abort_limit": round(0.3 * connect_lst.num_cells) }
	cost, routing_lst, channel_lst, place_matrix = placement(connect_lst, place_params)

	# Perform Routing
	print("Begin Routing")
	routing(routing_lst)

	# Write the solution to output file
	magic()


if __name__ == "__main__":
	main()
