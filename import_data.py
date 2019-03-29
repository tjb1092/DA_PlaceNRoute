from data_structs import ConnectivityList

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
