import random

class Cell:
	def __init__(self, num):
		self.num = num
		self.place_loc = (0, 0) # Define the relative placement to other cells.
		self.nbrs = {}  # constant time lookup for membership/weight

	def compute_place_loc(self):
		num_x, num_y, denom = 0, 0, 0
		for nbr in self.nbrs:

			w = self.nbrs[nbr]
			num_x += w * nbr.place_loc[0]
			num_y += w * nbr.place_loc[1]
			denom += w
		if denom == 0:
			denom += 1  # prevent /0 issue for cells w/ no neighbors. This will try to stick it in the 0,0 corner.
		return round(num_x / denom), round(num_y / denom)


class Net:
	def __init__(self, num):
		self.num = num
		self.terminals = {}


class ConnectivityList:
	def __init__(self, num_cells, num_nets):
		self.num_cells = num_cells
		self.num_nets = num_nets

		self.cells = {}  # constant time lookup for membership/weight
		for i in range(1, self.num_cells+1):
			# Instantiate all cells
			self.cells[i] = Cell(i)

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
		self.nets[net[0]].terminals = {net[1]:net[2], net[3]: net[4]}

	def compute_place_cost(self):
		cost = 0
		for cell in self.cells.values():
			x, y = cell.place_loc[0], cell.place_loc[1]
			for nbr in cell.nbrs:
				w = cell.nbrs[nbr]
				cost += (w * (abs(x - nbr.place_loc[0]) + abs(y - nbr.place_loc[1])))
		return cost


def data_load(fn):
	# Open file and read its contents
	with open(fn) as f:
		content = f.readlines()
		f.close()
	# Extract the number of cells and nets
	num_cells, num_nets = int(content[0][:-1]), int(content[1][:-1])
	connect_lst = ConnectivityList(num_cells, num_nets)

	print(num_cells, num_nets)

	for row in content[2:]:
		# For each row, add the net to the graph
		cells = row[:-1].split(" ")
		connect_lst.add_net([int(x) for x in cells])

	#print([(x.num,x.terminals) for x in connect_lst.nets.values()])
	#print([(x.num,[y.num for y in x.nbrs]) for x in connect_lst.cells.values()])
	return connect_lst



if __name__ == "__main__":
	data_load("Example-Netlists/1")
