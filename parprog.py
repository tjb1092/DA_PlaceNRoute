from import_data  import data_load, GroupLst, writeResults
import math
import random
import time
import os
import matplotlib.pyplot as plt
import argparse
#from pympler import classtracker
from operator import itemgetter
from collections import deque

def con_lin_plmt(n, C, P):
	print("hi")

def init_placement(connect_lst, place_matrix):
	l = len(place_matrix)
	print("L: {}".format(l))
	x, y, val = 0, 0, 1
	while val <= connect_lst.num_cells:
		place_matrix[x][y] = [val, False]
		connect_lst.cells[val].place_loc = (x, y)  # Add info to cell object
		val += 1
		y += 1
		if y >= l:
			y = 0
			x += 1


def unlock_positions(place_matrix):
	# reset all locked positions to unlocked.
	for i in range(len(place_matrix)):
		for j in range(len(place_matrix)):
			place_matrix[i][j][1] = False



def find_vacant_loc(place_matrix, start):
	# breadth first search the grid to find the nearest vacant spot
	l = len(place_matrix)
	queue = deque([start])
	seen = set([start])

	while queue:
		coord = queue.popleft()
		x, y = coord[0], coord[1]
		if place_matrix[x][y][0] == 0:
			return coord
		for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
			if 0 <= x2 < l and 0 <= y2 < l and (x2, y2) not in seen:
				queue.append((x2, y2))
				seen.add((x2, y2))

	return "ERROR"

def placement(connect_lst, place_matrix, place_params):
	debug = False
	init_placement(connect_lst, place_matrix)

	cost = connect_lst.compute_place_cost()
	print("Initial Cost: {}".format(cost))
	input("pause")

	tups = [(x.num, sum(x.nbrs.values())) for x in connect_lst.cells.values()]
	iter_num = 0
	last_time, total_time = time.time(), time.time()

	while iter_num < place_params["iteration_count"]:
		cost = connect_lst.compute_place_cost()
		print("Cost: {}".format(cost))
		print('Iteration took {:0.3f} seconds'.format((time.time()-last_time)))
		last_time = time.time()
		abort_count = 0
		sorted_cells = deque([x[0] for x in sorted(tups, key=itemgetter(1), reverse=True)]) # re-sort list based on connectivity

		skip_pop = False
		while len(sorted_cells) > 0:
			if not skip_pop:
				# skip if in a ripple
				cell = sorted_cells.popleft()
				cur_pos = connect_lst.cells[cell].place_loc
				place_matrix[cur_pos[0]][cur_pos[1]][0] = 0  # set current cell loc to vacant
			else:
				skip_pop = False

			x0, y0 = connect_lst.cells[cell].compute_place_loc()

			if x0 == cur_pos[0] and y0 == cur_pos[1]:
				# already in correct position? lock location
				place_matrix[cur_pos[0]][cur_pos[1]][0] = cell
				place_matrix[cur_pos[0]][cur_pos[1]][1] = True
				abort_count = 0
				if debug:
					print("cell {} moved from ({},{}) to ({},{}) using case {}".format(cell,cur_pos[0], cur_pos[1], cur_pos[0], cur_pos[1], 0))

			elif place_matrix[x0][y0][0] == 0:
				# ideal spot is vacant. move and lock
				place_matrix[x0][y0][0] = cell
				place_matrix[x0][y0][1] = True
				connect_lst.cells[cell].place_loc = (x0, y0)  # update cell pos
				abort_count = 0
				if debug:
					print("cell {} moved from ({},{}) to ({},{}) using case {}".format(cell,cur_pos[0], cur_pos[1], x0, y0, 1))

			elif place_matrix[x0][y0][0] != 0 and place_matrix[x0][y0][1] == False:
				# spot occupied, but not locked.
				# pop cell from that location. Then, put current cell in that pos and lock.
				# override queue with this new cell
				tmp_cell = place_matrix[x0][y0][0]
				sorted_cells.remove(tmp_cell)

				place_matrix[x0][y0][0] = cell
				place_matrix[x0][y0][1] = True
				connect_lst.cells[cell].place_loc = (x0, y0)  # update cell pos
				if debug:
					print("cell {} displaced cell {} by moving from ({},{}) to ({},{}) using case {}".format(cell, tmp_cell, cur_pos[0], cur_pos[1], x0, y0, 2))
				cell = tmp_cell
				cur_pos = connect_lst.cells[cell].place_loc
				skip_pop = True
				abort_count = 0

			else:
				# spot occupied and locked.
				# Find nearest vacant spot.
				# consider ripple aborted here.
				x0, y0 = find_vacant_loc(place_matrix, (x0, y0))
				place_matrix[x0][y0][0] = cell
				place_matrix[x0][y0][1] = True
				connect_lst.cells[cell].place_loc = (x0, y0)  # update cell pos
				abort_count += 1
				#print("abort_count: {}".format(abort_count))
				if debug:
					print("cell {} moved from ({},{}) to ({},{}) using case {}".format(cell,cur_pos[0], cur_pos[1], x0, y0, 3))

				if abort_count > place_params["abort_limit"]:
					unlock_positions(place_matrix)
					iter_num += 1

		unlock_positions(place_matrix)
		iter_num += 1  # Completed full list w/o hitting abort limit


	cost = connect_lst.compute_place_cost()
	print("Final cost: {}".format(cost))
	return cost

def construct_channel_lst(p_m_len):
	# Construct  channel list based on number of rows in grid
	channel_lst = [[0]]
	for i in range(1, p_m_len):
		channel_lst.append([2*i-1, 2*i])
	channel_lst.append([2*p_m_len - 1])
	return channel_lst


def add_feedthrough(connect_lst, place_matrix, channel_lst):

	# For each net, determine if a feedthrough cell needs to be place.

	for net in connect_lst.nets:
		cell1, term1 =  net.terminals[0]
		cell2, term2 = net.terminals[1]
		tx1, ty1 = connect_lst.cells[cell1].get_term_location(term1)
		tx2, ty2 = connect_lst.cells[cell2].get_term_location(term2)
		ch1 = [tx1 in x for x in channel_lst].index(True)
		ch2 = [tx2 in x for x in channel_lst].index(True)

		if abs(ch1-ch2) > 0:
			# if in different channels, need to add a feedthrough.
			for row in range(min(ch1), max(ch2)):
				# add feedthrough cell to these rows.
				add_feedthrough_cell(row, cell1, cell2, place_matrix, channel_lst)
				




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

	# cell num,locked. num = 0 = vacant
	place_matrix = [[[0, False] for i in range(x)] for j in range(x)]  # Instantiate a 2-D list matrix

	place_params = {"iteration_count": 100, "abort_limit": round(0.3 * connect_lst.num_cells) }

	# Execute placement engine.
	cost = placement(connect_lst, place_matrix, place_params)

	channel_lst = construct_channel_lst(len(place_matrix))

	# Based on row-placement, add feedthrough cells to allow for proper channel routing.
	add_feedthrough(connect_lst, place_matrix, channel_lst)




	input("Finished Placement")
	# Write the solution to output file
	writeResults(solution, cost, args.o)


if __name__ == "__main__":
	main()
