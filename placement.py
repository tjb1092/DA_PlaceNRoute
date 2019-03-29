import time
import matplotlib.pyplot as plt
import copy
#from pympler import classtracker
from operator import itemgetter
from collections import deque


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

def find_vacant_loc(place_matrix, start, is2D):
	# breadth first search the grid to find the nearest vacant spot
	l = len(place_matrix[start[0]])
	queue = deque([start])
	seen = set([start])

	while queue:
		coord = queue.popleft()
		x, y = coord[0], coord[1]

		if place_matrix[x][y][0] == 0:
			return coord, True

		if is2D:
			search_ops = ((x+1,y), (x-1,y), (x,y+1), (x,y-1))
		else:
			# Restrict to row search
			search_ops = ((x,y+1), (x,y-1))

		for x2, y2 in search_ops:
			if 0 <= x2 < l and 0 <= y2 < l and (x2, y2) not in seen:
				queue.append((x2, y2))
				seen.add((x2, y2))

	return "ERROR"

def force_directed_placement(connect_lst, place_matrix, place_params):
	if place_params["is2D"]:
		debug = False
	else:
		debug = False
	cost = connect_lst.compute_place_cost()
	best_place, best_cost = copy.deepcopy(place_matrix), cost

	print("Initial Cost: {}".format(cost))
	input("pause")

	tups = [(x.num, sum(x.nbrs.values())) for x in connect_lst.cells.values()]

	iter_num = 0
	last_time, total_time = time.time(), time.time()

	while iter_num < place_params["iteration_count"]:
		"""
		if not place_params["is2D"]:
			for row in place_matrix:
				print(row)
			#input("pause")
		"""
		abort_count = 0
		sorted_cells = deque([x[0] for x in sorted(tups, key=itemgetter(1), reverse=True)]) # re-sort list based on connectivity
		#print(sorted_cells)

		skip_pop = False
		while len(sorted_cells) > 0:

			if not skip_pop:
				# skip if in a ripple
				cell = sorted_cells.popleft()
				cur_pos = connect_lst.cells[cell].place_loc
				place_matrix[cur_pos[0]][cur_pos[1]][0] = 0  # set current cell loc to vacant
			else:
				skip_pop = False


			x0, y0 = connect_lst.cells[cell].compute_place_loc(place_params["is2D"])

			"""
			if not place_params["is2D"]:
				for row in place_matrix:
					print(row)
				print("len cells: {}".format(len(sorted_cells)))
				print("cell: {}, current: {}, desired ({}, {})".format(cell, cur_pos, x0, y0))
				input("pause")
			"""
			if x0 == cur_pos[0] and y0 == cur_pos[1] and not place_matrix[x0][y0][1]:
				# already in correct position? lock location.
				# The 3rd condition makes sure that a cell can't be placed in it's current spot
				# if it was displaced in a ripple.
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

			elif place_matrix[x0][y0][0] != 0 and not place_matrix[x0][y0][1]:
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
				coord, isVacant = find_vacant_loc(place_matrix, (x0, y0), place_params["is2D"])
				x0, y0 = coord[0], coord[1]
				"""
				if not isVacant:
					tmp_cell = place_matrix[x0][y0][0]
					print("cell: {}, Tmp cell: {}".format(cell, tmp_cell))
					print(sorted_cells)
					sorted_cells.remove(tmp_cell)
				"""
				place_matrix[x0][y0][0] = cell
				place_matrix[x0][y0][1] = True
				connect_lst.cells[cell].place_loc = (x0, y0)  # update cell pos
				abort_count += 1
				#print("abort_count: {}".format(abort_count))
				"""
				if not isVacant:
					cell = tmp_cell
					cur_pos = connect_lst.cells[cell].place_loc
					skip_pop = True
				"""
				if debug:
					print("cell {} moved from ({},{}) to ({},{}) using case {}".format(cell,cur_pos[0], cur_pos[1], x0, y0, 3))

				if abort_count > place_params["abort_limit"]:
					break

		unlock_positions(place_matrix)
		iter_num += 1  # Completed full list w/o hitting abort limit
		cost = connect_lst.compute_place_cost()
		print("Cost: {}".format(cost))
		print('Iteration took {:0.3f} seconds'.format((time.time()-last_time)))
		if cost < best_cost:
			best_place, best_cost = copy.deepcopy(place_matrix), cost

		last_time = time.time()

	#place_matrix = copy.deepcopy(best_place)
	#connect_lst.update_locations(place_matrix)
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
	del_net_lst = []  # Will delete all of the needed nets after the loop that is dependent on them
	current_nets = dict(connect_lst.nets)
	for net in current_nets.values():
		cell1, term1 =  net.terminals[0]
		cell2, term2 = net.terminals[1]
		tx1, ty1 = connect_lst.cells[cell1].get_term_location(term1)
		tx2, ty2 = connect_lst.cells[cell2].get_term_location(term2)
		ch1 = [tx1 in x for x in channel_lst].index(True)
		ch2 = [tx2 in x for x in channel_lst].index(True)
		#print("c1: {}, ch1: {}, c2: {}, ch2: {}".format(cell1, ch1, cell2, ch2))
		#input("pause")
		if abs(ch1-ch2) > 0:
			net_num = net.num
			if ch1 < ch2:
				splice_net = [net.terminals[0], net.terminals[1]]
			else:
				splice_net = [net.terminals[1], net.terminals[0]]
			#print("Adding feedthrough")
			#print("net num: {}, net: {}".format(net_num, splice_net))
			# if in different channels, need to add a feedthrough.
			for row in range(min(ch1, ch2), max(ch1, ch2)):
				# add feedthrough cell to these rows.
				#print("row: {}".format(row))
				ft_cell  = connect_lst.add_feedthrough_cell()
				place_matrix[row].append([ft_cell, False])  # Append to appropriate row
				connect_lst.cells[ft_cell].place_loc = (row, len(place_matrix[row])-1) # Update FT cell with coordinates

				#print("FT_cell loc: {}".format(connect_lst.cells[ft_cell].place_loc))
				#input("pause")
				del connect_lst.nets[net_num] # delete net to be spliced

				# splice the net
				spliced_net2 = connect_lst.splice_net(splice_net, ft_cell)
				splice_net = spliced_net2.terminals  # already ordered from top->bot
				net_num = spliced_net2.num

				#print([x for x in connect_lst.nets])
				#input("pause")

	# add dummy cells to make place_matrix square
	# determine the longest row
	l = 0
	for row in place_matrix:
		if len(row) > l:
			l = len(row)
	old_len = len(place_matrix) * 6
	print("old len {}".format(old_len))
	new_len = old_len + (l-len(place_matrix)) * 3
	print("new len {}".format(new_len))
	for row in range(len(place_matrix)):
		# for each row, append dummy cells till it's square
		while len(place_matrix[row]) < l:
			ft_cell  = connect_lst.add_feedthrough_cell()
			place_matrix[row].append([ft_cell, False])  # Append to appropriate row
			connect_lst.cells[ft_cell].place_loc = (row, len(place_matrix[row])-1) # Update FT cell with coordinates
		place_matrix[row].append([0, False])


def construct_routing_lst(connect_lst, place_matrix, channel_lst):
	# each row should have the net #
	print("hi")


def placement(connect_lst, place_params):

	# Figure out the min grid size
	x = 1
	while x**2 < connect_lst.num_cells:
		x += 1
	print("x = {}".format(x))

	# cell num,locked. num = 0 = vacant
	place_matrix = [[[0, False] for i in range(x)] for j in range(x)]  # Instantiate a 2-D list matrix

	# initalize placement
	init_placement(connect_lst, place_matrix)

	# Execute force-directed placement engine.
	print("Starting 1st Placement")
	cost = force_directed_placement(connect_lst, place_matrix, place_params)

	channel_lst = construct_channel_lst(len(place_matrix))

	# Based on row-placement, add feedthrough cells to allow for proper channel routing.
	add_feedthrough(connect_lst, place_matrix, channel_lst)

	print("Starting 2nd Placement")
	place_params["is2D"] = False

	cost = force_directed_placement(connect_lst, place_matrix, place_params)

	routing_lst = construct_routing_lst(connect_lst, place_matrix, channel_lst)
	return cost, routing_lst, place_matrix
