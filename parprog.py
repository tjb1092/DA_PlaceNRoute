#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main
~~~~~~~~~~~~~~~~~~~
Physical Synthesis performing Place and Route
~~~~~~~~~~~~~~~~~~~

Authors:
Tony Bailey
Zach Collins

Usage:
Requires Python3.5 +
Tested and benchmarked on VLSI lab computers 

General CLI Interface:
python3 parprog.py -i Benchmarks/InputBenchmark -o Results/OutputMagic -padding 3 -iter_num 150

padding and iter_num are optional arguments that default to 3 and 150 respectively
"""

from import_data  import data_load
from magic import magic
from placement import placement
from routing import routing
import time
import os
import argparse

def main():
  total_time = time.time()

  # Needed for the specified command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', type=str, default="Example-Netlists/1",
                    help='What is the input filename?')
  parser.add_argument('-o', type=str, default="Results/R_1",
                    help='What is the output filename?')
  parser.add_argument('-iter_num', type=int, default=150,
                    help='Number of iterations for 1st placement')
  parser.add_argument('-place_padding', type=int, default=3,
                    help='(rows + padding) x (col+padding) dimensions in placement')
  args = parser.parse_args()

  # generate the output file directories if the don't exist
  if not os.path.exists("Results"):
    os.makedirs("Results")

  # Load the input data and generate the graph
  connect_lst = data_load(args.i)
  print("Graph Created")

  # Perform Placement

  place_params = {"is2D":True, "iteration_count": args.iter_num, "abort_limit": round(0.3 * connect_lst.num_cells), "padding":args.place_padding}
  cost, feedthrough_count, routing_lst, channel_lst, place_matrix = placement(connect_lst, place_params)
  print(".\n.\n.")

  # Perform Routing
  print("Begin Routing")
  all_channels, doglegs, routing_list, net_to_leftedge, net_to_rightedge = routing(routing_lst)
  print("Routing Finished!")
  print(".\n.\n.")

  # Write the solution to output file
  print("Begin File Generation")
  magic(all_channels, doglegs, routing_list, net_to_leftedge, net_to_rightedge, args.o, connect_lst, place_matrix)
  print("File Generation Finished!")

  print("Total Execution time: {:0.3f} seconds".format(time.time()-total_time))


if __name__ == "__main__":
  main()
