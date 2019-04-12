import time
import copy
#from pympler import classtracker
from operator import itemgetter
from collections import deque


def init_placement(connect_lst, place_matrix):
	l = len(place_matrix)
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
	#input("pause")

	tups = [(x.num, sum(x.nbrs.values())) for x in connect_lst.cells.values()]

	iter_num = 0
	last_time, total_time = time.time(), time.time()

	while iter_num < place_params["iteration_count"]:

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

			x0, y0 = connect_lst.cells[cell].compute_place_loc(place_params["is2D"])

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
				# spot occupied and locked. Find nearest vacant spot.
				# consider ripple aborted here.
				coord, isVacant = find_vacant_loc(place_matrix, (x0, y0), place_params["is2D"])
				x0, y0 = coord[0], coord[1]
				place_matrix[x0][y0][0] = cell
				place_matrix[x0][y0][1] = True
				connect_lst.cells[cell].place_loc = (x0, y0)  # update cell pos
				abort_count += 1

				if debug:
					print("cell {} moved from ({},{}) to ({},{}) using case {}".format(cell,cur_pos[0], cur_pos[1], x0, y0, 3))
					print("abort_count: {}".format(abort_count))
				if abort_count > place_params["abort_limit"]:
					break

		unlock_positions(place_matrix)
		iter_num += 1  # Completed full list w/o hitting abort limit
		cost = connect_lst.compute_place_cost()
		print("Cost: {}, Iteration {}/{} took {:0.3f} seconds".format(cost, iter_num, place_params["iteration_count"], time.time()-last_time), end="\r")
		if cost < best_cost:
			best_place, best_cost = copy.deepcopy(place_matrix), cost

		last_time = time.time()

	#place_matrix = copy.deepcopy(best_place)
	#connect_lst.update_locations(place_matrix)
	cost = connect_lst.compute_place_cost()
	print("\nFinal cost: {}".format(cost))

	print("Placement took {:0.3f} seconds".format(time.time()-total_time))
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
	feedthrough_count = 0  # Keep track of number of added feedthrough cells.
	for net in current_nets.values():
		cell1, term1 =  net.terminals[0]
		cell2, term2 = net.terminals[1]
		tx1, ty1 = connect_lst.cells[cell1].get_term_location(term1)
		tx2, ty2 = connect_lst.cells[cell2].get_term_location(term2)
		ch1 = [tx1 in x for x in channel_lst].index(True)
		ch2 = [tx2 in x for x in channel_lst].index(True)

		if abs(ch1-ch2) > 0:
			net_num = net.num
			if ch1 < ch2:
				splice_net = [net.terminals[0], net.terminals[1]]
			else:
				splice_net = [net.terminals[1], net.terminals[0]]

			# if in different channels, need to add a feedthrough.
			for row in range(min(ch1, ch2), max(ch1, ch2)):
				# add feedthrough cell to these rows.
				ft_cell  = connect_lst.add_feedthrough_cell()
				place_matrix[row].append([ft_cell, False])  # Append to appropriate row
				connect_lst.cells[ft_cell].place_loc = (row, len(place_matrix[row])-1) # Update FT cell with coordinates

				del connect_lst.nets[net_num] # delete net to be spliced

				# splice the net
				spliced_net2 = connect_lst.splice_net(splice_net, ft_cell)
				splice_net = spliced_net2.terminals  # already ordered from top->bot
				net_num = spliced_net2.num

				feedthrough_count += 1

	# determine the longest row
	l = 0
	for row in place_matrix:
		if len(row) > l:
			l = len(row)
	cell_num = len(place_matrix)
	print("Width after adding feedthroughs: {} lambda".format(l*6))

	for row in range(len(place_matrix)):
		# for each row, append vacant spots till it's rectangular
		while len(place_matrix[row]) < l:
			#ft_cell  = connect_lst.add_feedthrough_cell()
			place_matrix[row].append([0, False])  # Append to appropriate row
			#connect_lst.cells[ft_cell].place_loc = (row, len(place_matrix[row])-1) # Update FT cell with coordinates
		place_matrix[row].append([0, False])
	return feedthrough_count

def construct_routing_lst(connect_lst, place_matrix, channel_lst):
	# each row should have the net #
	channels_num = 2* len(place_matrix)
	term_num = 0
	for i in range(len(place_matrix[0])):
		tmp_cell = place_matrix[0][i][0]
		term_num += 2

	routing_lst = [[0 for i in range(term_num)] for j in range(channels_num)]  # Instantiate a 2-D list matrix

	for net in connect_lst.nets.values():
		# for each net, assign the proper terminals to routing_lst

		for [cell, term] in net.terminals:
			# for the two cells, find row and col.
			# const height, so row is straight forward from X coord
			c = connect_lst.cells[cell]
			if (c.isCkt and (term == 1 or term == 2)) or (not c.isCkt and term == 1):
				row = c.place_loc[0] * 2
			else:
				row = c.place_loc[0] * 2 + 1

			# for col, the cell widths are not equal, so a traversal of place_matrix is needed
			# note: I'll make this the same as the row col later. It works now and it's pretty fast
			col = 0
			for i in range(c.place_loc[1]):
				tmp_cell = place_matrix[c.place_loc[0]][i][0]
				col += 2

			if c.isCkt and (term == 2 or term == 4):
				col += 1

			routing_lst[row][col] = net.num
	return routing_lst

def placement(connect_lst, place_params):

	# Figure out the min grid size
	x = 1
	while x**2 < connect_lst.num_cells:
		x += 1

	x += place_params["padding"]  # Add extra vacant spots for cells to move to. proportional to the sqrt(number of cells)

	# [cell num, locked]. num = 0 = vacant
	place_matrix = [[[0, False] for i in range(x)] for j in range(x)]  # Instantiate a 2-D list matrix

	# inital, sequential placement
	init_placement(connect_lst, place_matrix)

	# Execute force-directed placement engine.
	print("Starting 1st Placement")
	cost = force_directed_placement(connect_lst, place_matrix, place_params)
	print(".\n.\n.")

	channel_lst = construct_channel_lst(len(place_matrix))

	# Based on row-placement, add feedthrough cells to allow for proper channel routing.
	feedthrough_count = add_feedthrough(connect_lst, place_matrix, channel_lst)
	print("Number of feedthrough cells added: {}".format(feedthrough_count))

	print("Starting 2nd Placement")
	place_params["is2D"] = False
	place_params["iteration_count"] = 0.2 * place_params["iteration_count"]
	cost = force_directed_placement(connect_lst, place_matrix, place_params)

	routing_lst = construct_routing_lst(connect_lst, place_matrix, channel_lst)
	print("routing dim: ({}, {})".format(len(routing_lst), len(routing_lst[0])))

	print("Placement Finished!")
	return cost, feedthrough_count, routing_lst, channel_lst, place_matrix
