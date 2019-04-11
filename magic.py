import time

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
