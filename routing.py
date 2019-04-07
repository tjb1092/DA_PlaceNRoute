from collections import defaultdict
import time


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
