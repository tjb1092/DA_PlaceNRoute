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
