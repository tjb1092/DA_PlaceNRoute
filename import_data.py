import random

class Cell:
	def __init__(self, num):
		self.num = num
		self.nbrs = {}  # constant time lookup for membership/weight

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
		#print(net)
		#print(self.cells[net[1]].nbrs)
		if net[3] in self.cells[net[1]].nbrs:
			# if it's a repeat, add 1 to "weight"
			self.cells[net[1]].nbrs[net[3]] += 1
		else:
			#else, create connection
			self.cells[net[1]].nbrs[net[3]] = 1

		#print(self.cells[net[1]].nbrs)
		# add 1->0 net
		if net[1] in self.cells[net[3]].nbrs:
			# if it's a repeat, add 1 to "weight"
			self.cells[net[3]].nbrs[net[1]] += 1
		else:
			#else, create connection
			self.cells[net[3]].nbrs[net[1]] = 1
		self.nets[net[0]].terminals = {net[1]:net[2], net[3]: net[4]}


class GroupLst:
	def __init__(self, graph, num_cells):
		self.graph = graph  # pointer to adjacency list
		self.num_cells = num_cells
		# initialize group array to have 50% group A and 50% group B.
		self.V = [0]*(self.num_cells//2) + [1]*(self.num_cells//2)
		random.shuffle(self.V)  # initialize random partition
		self.init_cost()  # Initialize the cost of the random partition

	def init_cost(self):
		cost = 0
		for i, n in enumerate(self.V):
			# Find 0 cells w/ connections in 1
			if n == 0:
				if i+1 in self.graph.cells:
					for nbr in self.graph.cells[i+1].nbrs:
						if self.V[nbr-1] == 1:
							# Add their weights
							cost += self.graph.cells[i+1].nbrs[nbr]
		self.cost = cost

	def perturb(self):
		# randomly switch two cells
		A = random.randint(0,self.num_cells-1)
		chosen = False
		while not chosen:
			# Ensures that the B cell is not in A
			B = random.randint(0,self.num_cells-1)
			if B == A or self.V[B] == self.V[A]:
				continue
			else:
				chosen = True
		# Switch the groups
		self.V[A], self.V[B] = int(not(self.V[A])), int(not(self.V[B]))

		# Update the costs based on that move
		self.update_cost(A+1, B+1)  # The +1 offsets the 0-index of the list
		self.update_cost(B+1, A+1)

	def update_cost(self, A, B):
		# get A's group
		gA = self.V[A-1]
		if A in self.graph.cells:
			# Traverse through all of A's nbrs.
			for nbr in self.graph.cells[A].nbrs:
				if self.V[nbr-1] == gA and nbr != (B):
					# Reduce cost if nbr is now in the same group as A
					self.cost = self.cost - self.graph.cells[A].nbrs[nbr]
				elif self.V[nbr-1] != gA and nbr != (B):
					# Increase "    "
					self.cost = self.cost + self.graph.cells[A].nbrs[nbr]

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
		print(cells)
		connect_lst.add_net([int(x) for x in cells])

	#print([(x.num,x.terminals) for x in connect_lst.nets.values()])
	#print([(x.num,x.nbrs) for x in connect_lst.cells.values()])
	return connect_lst


def writeResults(solution, cost, fn):
	# Write the cost and cutsets to an output file.
	f = open("Results/{}".format(fn), "w")
	newline = str(cost)+ "\n"  # Generate cost string
	f.write(newline)
	A, B = "", ""
	# Generate strings for the A group and the B group
	for i, n in enumerate(solution):
		if n == 0:
			A += (str(i+1) + " ")
		else:
			B += (str(i+1) + " ")

	f.write(A+"\n")
	f.write(B+"\n")
	f.close()

if __name__ == "__main__":
	data_load("Example-Netlists/1")
