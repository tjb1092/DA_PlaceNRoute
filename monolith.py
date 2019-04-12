#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monolithic Python File
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

# Data structures used for placement and routing.

class Net:
  def __init__(self, num):
    self.num = num
    self.terminals = []


class Cell:
  def __init__(self, num, isCkt):
    self.num = num
    self.place_loc = (0, 0) # Define the relative placement to other cells.
    self.nbrs = {}  # constant time lookup for membership/weight
    self.isCkt = isCkt  # True = ckt block, False = feedthrough
    self.nets = {}

  def compute_place_loc(self, is2D):
    # Calculate ideal placement location
    num_x, num_y, denom = 0, 0, 0
    for nbr in self.nbrs:

      w = self.nbrs[nbr]
      num_x += w * nbr.place_loc[0]
      num_y += w * nbr.place_loc[1]
      denom += w
    if denom == 0:
      denom += 1  # prevent /0 issue for cells w/ no neighbors. This will try to stick it in the 0,0 corner.

    if is2D:
      out_x = round(num_x / denom)
    else:
      out_x = self.place_loc[0]  # restrict to only present row
    return out_x, round(num_y / denom)

  def get_term_location(self, term):

    # compute tx:
    if term == 1 or term == 2:
      tx = 2 * self.place_loc[0]
    else:
      tx = 2 * self.place_loc[0] + 1

    # compute ty
    if term == 1 or term == 3:
      ty = 2 * self.place_loc[1]
    else:
      ty = 2 * self.place_loc[1] + 1
    return tx, ty


class ConnectivityList:
  def __init__(self, num_cells, num_nets):
    self.num_cells = num_cells
    self.num_nets = num_nets

    self.cells = {}  # constant time lookup for membership/weight
    for i in range(1, self.num_cells+1):
      # Instantiate all cells
      self.cells[i] = Cell(i, True)

    self.nets = {}
    for i in range(1, self.num_nets+1):
      # Instantiate all nets
      self.nets[i] = Net(i)

  def add_net(self, net):
    # net is a list of length 5
    # [0-net#, 1-cell1#, 2-term1#, 3-cell2#, 2-term2#]

    # add 0->1 net
    if self.cells[net[3]] in self.cells[net[1]].nbrs:
      # if it's a repeat, add 1 to "weight"
      self.cells[net[1]].nbrs[self.cells[net[3]]] += 1
    else:
      #else, create connection
      self.cells[net[1]].nbrs[self.cells[net[3]]] = 1

    # add 1->0 net
    if self.cells[net[1]] in self.cells[net[3]].nbrs:
      # if it's a repeat, add 1 to "weight"
      self.cells[net[3]].nbrs[self.cells[net[1]]] += 1
    else:
      #else, create connection
      self.cells[net[3]].nbrs[self.cells[net[1]]] = 1
    self.nets[net[0]].terminals = [(net[1], net[2]), (net[3], net[4])]

  def add_feedthrough_cell(self):
    # Add new feedthrough cell to the dict of total cells.
    self.num_cells += 1

    self.cells[self.num_cells] = Cell(self.num_cells, False)
    return self.num_cells  # Return the ft cell number

  def splice_net(self, net, ft_cell):
    # Create top ft net
    self.num_nets += 1
    self.nets[self.num_nets] = Net(self.num_nets)
    self.add_net([self.num_nets, net[0][0], net[0][1], ft_cell, 1])

    # Create bottom ft net
    self.num_nets += 1
    self.nets[self.num_nets] = Net(self.num_nets)
    self.add_net([self.num_nets, ft_cell, 2, net[1][0], net[1][1]])

    return self.nets[self.num_nets]

  def compute_place_cost(self):
    # For each net, compute the rectilinear distance
    # between the two connected cells.
    cost = 0
    for net in self.nets.keys():
      cell1 = self.nets[net].terminals[0][0]
      cell2 = self.nets[net].terminals[1][0]

      x1, y1 = self.cells[cell1].place_loc
      x2, y2 = self.cells[cell2].place_loc

      cost += abs(x1 - x2) + abs(y1 - y2)
    return cost




def data_load(fn):
  # Open file and read its contents
  with open(fn) as f:
    content = f.readlines()
    f.close()
  # Extract the number of cells and nets
  num_cells, num_nets = int(content[0][:-1]), int(content[1][:-1])
  connect_lst = ConnectivityList(num_cells, num_nets)

  print("Cell Count: {}, Net Count: {}".format(num_cells, num_nets))

  for row in content[2:]:
    # For each row, add the net to the graph
    cells = row[:-1].split(" ")
    connect_lst.add_net([int(x) for x in cells])
  return connect_lst


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
  print("Initial Cost: {}".format(cost))
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

    last_time = time.time()

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






from collections import defaultdict

# Function to build a vcg from channel lists plus all nets
def create_vcg(top, bottom, unassigned):
  vcg = {}
  for item in top:
    if item != 0:
      vcg[item] = set()
  for item in bottom:
    if item != 0:
      vcg[item] = set()
  for column in zip(top, bottom):
    if column[0] != 0 and column[1] != 0 and column[0] in unassigned and column[1] in unassigned and column[0] != column[1]:
      vcg[column[0]].add(column[1])
  return vcg


# Function to see if node in vcg has parents
# VCGs are so small there is nearly no performance difference
def has_parents(net, vcg):
  for par, children in vcg.items():
    if net in children:
      return True
  return False


# Function to see if node in vcg has children
def has_children(net, vcg):
  return bool(vcg[net])


# Depth first search of a graph represented as a matrix
def dfs(graph, start, end):
  fringe = [(start, [])]
  while fringe:
    state, path = fringe.pop()
    if path and state == end:
      yield path
      continue
    for next_state in graph[state]:
      if next_state in path:
        continue
      fringe.append((next_state, path+[next_state]))


def routing(routing_list):
  # Initialize counter
  start = time.time()
  # Append rows of all 0s so we can make tracks channel by channel
  zero_list = [0 for x in routing_list[0]]
  routing_list = [zero_list] + routing_list + [zero_list]
  # Initialize variables
  all_channels = []
  net_to_leftedge_by_channel = []
  net_to_rightedge_by_channel = []
  doglegs_by_channel = []
  # Calculate width of each channel
  channel_width = len(zero_list)
  # Iterate through all channels, top + bottom of channel at a time
  it = iter(routing_list)
  for top in it:
    # Grab next row too
    bottom = next(it)

    # Generate list of all nets, and dicts of left and right edges
    all_nets = set()
    net_to_leftedge = {}
    net_to_rightedge = {}
    for idx, item in enumerate(top):
      if item != 0:
        net_to_leftedge[item] = min(net_to_leftedge.get(item, channel_width), idx)
        net_to_rightedge[item] = max(net_to_rightedge.get(item, 0), idx)
        all_nets.add(item)
    for idx, item in enumerate(bottom):
      if item != 0:
        net_to_leftedge[item] = min(net_to_leftedge.get(item, channel_width), idx)
        net_to_rightedge[item] = max(net_to_rightedge.get(item, 0), idx)
        all_nets.add(item)
    nets_unrouted = set()
    nets_unassigned = all_nets.copy()
    tracks = []

    # This section checks for and removes cycles from the VCG
    # TODO: exit this section and redo placement if we have an unrouted net
    doglegs = []
    vcg = create_vcg(top, bottom, nets_unassigned)
    # check for cycles in VCG
    cycles = [[node]+path for node in vcg for path in dfs(vcg, node, node)]
    while cycles:
      # Grab the outermost net in the cycle. The first net must be at the left and right edges of the cycle.
      net = cycles[0][0]
      # Generate a number for our new doglegged net
      new_net_num = 1
      while new_net_num in all_nets:
        new_net_num += 100
      # Grab left and right edge of the net we're splitting
      leftedge = net_to_leftedge[net]
      rightedge = net_to_rightedge[net]
      # Mark cycle as unsolved for now
      solved = False

      # Look through the columns between the left and right edge of the outermost net
      for idx, col in enumerate(list(zip(top, bottom))[leftedge + 1:rightedge]):
        # All nets are two-terminal in this assignment, so the
        # only place we can dogleg is an empty column.
        if col == (0, 0):
          # Calculate the actual column number.
          col_num = idx + leftedge + 1
          # Split the net
          top[col_num] = net
          bottom[col_num] = new_net_num
          bottom[rightedge] = new_net_num
          net_to_rightedge[net] = col_num
          net_to_leftedge[new_net_num] = col_num
          net_to_rightedge[new_net_num] = rightedge
          # Add our new net to unassigned
          nets_unassigned.add(new_net_num)
          # Keep track of this dogleg for file generation
          doglegs += [(net, new_net_num)]
          solved = True
          break
      # If we didn't solve the dogleg, just mark this net as not routed
      if not solved:
        nets_unassigned.remove(net)
        nets_unrouted.add(net)
      # Recreate VCG and cycles for the next cycle removal
      vcg = create_vcg(top, bottom, nets_unassigned)
      cycles = [[node]+path for node in vcg for path in dfs(vcg, node, node)]
    doglegs_by_channel += [doglegs]
    # Perform net assignment!
    # Net assignment is top to bottom in each track
    while nets_unassigned:
      # Calculate VCG
      vcg = create_vcg(top, bottom, nets_unassigned)
      # This routing uses leftedge algo, so we sort by leftedge
      nets_by_leftedge = sorted(list(nets_unassigned), key=lambda x: net_to_leftedge[x])
      # Initialize current track
      track = []
      track_right_edge = -1
      iter_nets_assigned = False
      # Try assigning all the nets to this track
      for net in nets_by_leftedge:
        # If the net has no parent in the VCG and it fits in this track, place it
        if not has_parents(net, vcg) and net_to_leftedge[net] > track_right_edge:
          nets_unassigned.remove(net)
          iter_nets_assigned = True
          track += [net]
          track_right_edge = net_to_rightedge[net]
      # If we didn't place any nets this iteration, we still have a cycle... Uh oh.
      if not iter_nets_assigned:
        print("!!!Uncaught Cycle!!!")
        print("This should never occur.")
        print("Moving on to next channel.")
        break
      # Add track to channel
      tracks += [track]
    # Add channel to list of all channels
    net_to_leftedge_by_channel += [net_to_leftedge]
    net_to_rightedge_by_channel += [net_to_rightedge]
    all_channels += [tracks]

  # Repair dogleg columns so we don't draw the wrong M2
  for idx, doglegs in enumerate(doglegs_by_channel):
    for dogleg in doglegs:
      top_row = routing_list[idx*2]
      bottom_row = routing_list[idx*2 + 1]
      top_row[bottom_row.index(dogleg[1])] = 0
      bottom_row[bottom_row.index(dogleg[1])] = 0

  # Time it!
  end = time.time()
  print("Time Elapsed: " + str(end - start)[:4] + " seconds.")

  return all_channels, doglegs_by_channel, routing_list, net_to_leftedge_by_channel, net_to_rightedge_by_channel


def length_of_wire(x_min, y_min, x_max, y_max):
  return (x_max - x_min) * (y_max - y_min)

def magic(all_channels, doglegs, routing_list, net_to_leftedge, net_to_rightedge, outfile, connect_list, place_matrix):
  # cell_num = place_matrix[row][col][0]
  # isCircuit = connect_list[cell_num].isCkt
  # Initialize timer
  start = time.time()
  num_vias = 0
  wire_length = 0
  # Append .mag to the output file if we don't get one
  if not outfile.endswith(".mag"):
    outfile += ".mag"
  # Open file for writing
  f = open(outfile, "w")
  # Create file header, tech, and timestamp
  f.write("magic\n")
  f.write("tech scmos\n")
  f.write("timestamp " + str(int(time.time())) + " \n")

  # Produce p-diffusion region (standard cells)
  f.write("<< pdiffusion >>\n")
  standard_cells = []
  all_labels = []
  # Group routing list by what cells the terminals belong to
  cell_row = 0
  it = iter(routing_list[1:-1])
  for top in it:
    cells_here = []
    bottom = next(it)
    # Group columns by the cell they belong to
    cell_col = -1
    for i in range(0, len(top), 2):
      cell_col += 1
      # If the cell exists, we'll need to print it
      # Even if it doesn't have any terminals
      cell_num = place_matrix[cell_row][cell_col][0]
      if cell_num != 0:
        top_left = False
        top_right = False
        bottom_left = False
        bottom_right = False
        # Figure out what pins the cell has
        if top[i] != 0:
          top_left = True
        if top[i + 1] != 0:
          top_right = True
        if bottom[i] != 0:
          bottom_left = True
        if bottom[i + 1] != 0:
          bottom_right = True
        # Collect all the information about the cell
        if not connect_list.cells[cell_num].isCkt and not (top_left or top_right or bottom_left or bottom_right):
          continue
        cells_here += [(i // 2, top_left, top_right, bottom_left, bottom_right, cell_num)]
    standard_cells += [cells_here]
    cell_row += 1
  # Print all non-empty cells
  y_coord = 0
  for idx, row in enumerate(standard_cells):
    y_coord -= 2 * len(all_channels[idx]) + 1
    for cell in row:
      x_coord = cell[0] * 7
      if connect_list.cells[cell[5]].isCkt:
        f.write("rect " + str(x_coord + 1) + " " + str(y_coord - 5) + " " + str(x_coord + 7) + " " + str(y_coord + 1) + "\n")
      else:
        f.write("rect " + str(x_coord + 1) + " " + str(y_coord - 5) + " " + str(x_coord + 4) + " " + str(y_coord + 1) + "\n")
      labelText = "feedthrough"
      if connect_list.cells[cell[5]].isCkt:
        labelText = "cellNo=" + str(cell[5])
      all_labels += [("pdiffusion", x_coord + 3, y_coord - 3, x_coord + 3, y_coord - 3, 0, labelText)]
    y_coord -= 8

  # Print polysilicon pads for standard cells
  f.write("<< polysilicon >>\n")
  y_coord = 0
  for idx, row in enumerate(standard_cells):
    y_coord -= 2 * len(all_channels[idx]) + 1
    for cell in row:
      x_coord = cell[0] * 7
      if cell[1]:
        f.write("rect " + str(x_coord + 2) + " " + str(y_coord) + " " + str(x_coord + 3) + " " + str(y_coord + 2) + "\n")
        all_labels += [("polysilicon", x_coord + 2, y_coord + 1, x_coord + 2, y_coord + 1, 0, "1")]
      if cell[2]:
        f.write("rect " + str(x_coord + 5) + " " + str(y_coord) + " " + str(x_coord + 6) + " " + str(y_coord + 2) + "\n")
        all_labels += [("polysilicon", x_coord + 5, y_coord + 1, x_coord + 5, y_coord + 1, 0, "2")]
      if cell[3]:
        f.write("rect " + str(x_coord + 2) + " " + str(y_coord - 6) + " " + str(x_coord + 3) + " " + str(y_coord - 4) + "\n")
        all_labels += [("polysilicon", x_coord + 2, y_coord - 5, x_coord + 2, y_coord - 5, 0, "3")]
      if cell[4]:
        f.write("rect " + str(x_coord + 5) + " " + str(y_coord - 6) + " " + str(x_coord + 6) + " " + str(y_coord - 4) + "\n")
        all_labels += [("polysilicon", x_coord + 5, y_coord - 5, x_coord + 5, y_coord - 5, 0, "4")]
    y_coord -= 8

  # Produce metal1
  vias = []
  f.write("<< metal1 >>\n")
  y_coord = 0
  for idx, channel in enumerate(all_channels):
    for track in channel:
      for wire in track:
        if net_to_leftedge[idx][wire] == net_to_rightedge[idx][wire]:
          continue
        num_cells_left = net_to_leftedge[idx][wire] // 2
        x_coord_left = num_cells_left * 7 + 2
        if net_to_leftedge[idx][wire] % 2 == 1:
          x_coord_left += 3
        num_cells_right = net_to_rightedge[idx][wire] // 2
        x_coord_right = num_cells_right * 7 + 2
        if net_to_rightedge[idx][wire] % 2 == 1:
          x_coord_right += 3
        f.write("rect " + str(x_coord_left) + " " + str(y_coord) + " " + str(x_coord_right + 1) + " " + str(y_coord + 1) + "\n")
        num_vias += 2
        wire_length += length_of_wire(x_coord_left, y_coord, x_coord_right, y_coord + 1)
        vias += [(x_coord_left, y_coord, x_coord_left + 1, y_coord + 1)]
        vias += [(x_coord_right, y_coord, x_coord_right + 1, y_coord + 1)]
      y_coord -= 2
    y_coord -= 9

  # Produce vias
  f.write("<< m2contact >>\n")
  for via in vias:
    f.write("rect " + str(via[0]) + " " + str(via[1]) + " " + str(via[2]) + " " + str(via[3]) + "\n")

  # Produce metal2
  # This is by far the messiest function
  f.write("<< metal2 >>\n")
  # Track the y coordinate of each track, and the coordinates for the top and
  # bottom of the current channel
  y_coord = 0
  y_coord_top = 0
  y_coord_bottom = 0
  for idx, channel in enumerate(all_channels):
    y_coord_bottom -= 2 * len(channel)
    for track in channel:
      for wire in track:
        # Get the leftedge coordinate of the wire
        num_cells_left = net_to_leftedge[idx][wire] // 2
        x_coord_left = num_cells_left * 7 + 2
        if net_to_leftedge[idx][wire] % 2 == 1:
          x_coord_left += 3
        # Get the rightedge coordinate of the wire
        num_cells_right = net_to_rightedge[idx][wire] // 2
        x_coord_right = num_cells_right * 7 + 2
        if net_to_rightedge[idx][wire] % 2 == 1:
          x_coord_right += 3
        # Grab the top and bottom pinouts
        top_row = routing_list[idx*2]
        bottom_row = routing_list[idx*2 + 1]
        # Get the leftedge and rightedge pins
        leftedge = net_to_leftedge[idx][wire]
        rightedge = net_to_rightedge[idx][wire]
        # Print leftedge
        # This section won't be activated on the bottom-right of the dogleg
        if top_row[leftedge] == wire:
          f.write("rect " + str(x_coord_left) + " " + str(y_coord) + " " + str(x_coord_left + 1) + " " + str(y_coord_top) + "\n")
          wire_length += length_of_wire(x_coord_left, y_coord, x_coord_left + 1, y_coord_top + 1)
          all_labels += [("metal2", x_coord_left, y_coord + 1, x_coord_left, y_coord + 1, 0, "net=" + str(wire))]
        if bottom_row[leftedge] == wire:
          f.write("rect " + str(x_coord_left) + " " + str(y_coord_bottom + 1) + " " + str(x_coord_left + 1) + " " + str(y_coord + 1) + "\n")
          wire_length += length_of_wire(x_coord_left, y_coord_bottom + 1, x_coord_left + 1, y_coord + 1)
          all_labels += [("metal2", x_coord_left, y_coord + 1, x_coord_left, y_coord + 1, 0, "net=" + str(wire))]
        # Rightedge is where we do dogleg, so we need to keep track of whether
        # the right edge went all the way down to the bottom
        did_rightedge = False
        if top_row[rightedge] == wire:
          f.write("rect " + str(x_coord_right) + " " + str(y_coord) + " " + str(x_coord_right + 1) + " " + str(y_coord_top) + "\n")
          wire_length += length_of_wire(x_coord_right, y_coord, x_coord_right + 1, y_coord_top)
          did_rightedge = True
        if bottom_row[rightedge] == wire:
          f.write("rect " + str(x_coord_right) + " " + str(y_coord_bottom + 1) + " " + str(x_coord_right + 1) + " " + str(y_coord + 1) + "\n")
          wire_length += length_of_wire(x_coord_right, y_coord_bottom + 1, x_coord_right + 1, y_coord + 1)
          did_rightedge = True
        # If we didn't print a rightedge, it's a dogleg
        # All doglegs have a top-left and a bottom-right trunk
        if not did_rightedge:
          dogleg = None
          # Find the correct wire pair
          for d in doglegs[idx]:
            if d[0] == wire:
              dogleg = d
              break
          if dogleg != None:
            dogleg_y_bottom = y_coord_top - 2
            for d_track in channel:
              if dogleg[1] in d_track:
                break
              dogleg_y_bottom -= 2
            f.write("rect " + str(x_coord_right) + " " + str(dogleg_y_bottom) + " " + str(x_coord_right + 1) + " " + str(y_coord + 1) + "\n")
            wire_length += length_of_wire(x_coord_right, dogleg_y_bottom, x_coord_right + 1, y_coord + 1)
      y_coord -= 2
    y_coord -= 9
    y_coord_top = y_coord + 2
    y_coord_bottom = y_coord

  f.write("<< labels >>\n")
  for label in all_labels:
    f.write("rlabel " + label[0] + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + " " + str(label[4]) + " " + str(label[5]) + " " + str(label[6]) + "\n")

  f.write("<< end >>\n")
  f.close()
  # Time it!
  end = time.time()
  print("Number of vias: " + str(num_vias))
  print("Total wire length: " + str(wire_length))
  print("Time Elapsed: " + str(end - start)[:4] + " seconds.")





if __name__ == "__main__":
  main()
