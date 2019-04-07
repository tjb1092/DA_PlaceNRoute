import time

def magic(all_channels, doglegs, routing_list, net_to_leftedge, net_to_rightedge, outfile):
  if not outfile.endswith(".mag"):
    outfile += ".mag"
  f = open(outfile, "w")
  f.write("magic\n")
  f.write("tech scmos\n")
  f.write("timestamp " + str(int(time.time())) + " \n")

  # Produce p-diffusion region (standard cells)
  f.write("<< pdiffusion >>\n")
  standard_cells = []
  it = iter(routing_list[1:-1])
  for top in it:
    cells_here = []
    bottom = next(it)
    for i in range(0, len(top), 2):
      if (top[i] != 0 or top[i+1] != 0 or bottom[i] != 0 or bottom[i+1] != 0):
        cells_here += [i // 2]
    standard_cells += [cells_here]
  y_coord = 0
  for idx, row in enumerate(standard_cells):
    y_coord -= 2 * len(all_channels[idx])
    for cell in row:
      x_coord = cell * 7
      f.write("rect " + str(x_coord + 1) + " " + str(y_coord - 5) + " " + str(x_coord + 7) + " " + str(y_coord + 1) + "\n")
    y_coord -= 7

  # Produce metal1
  f.write("<< metal1 >>\n")
  y_coord = 0
  for idx, channel in enumerate(all_channels):
    for track in channel:
      for wire in track:
        num_cells_left = net_to_leftedge[idx][wire] // 2
        x_coord_left = num_cells_left * 7 + 2
        if net_to_leftedge[idx][wire] % 2 == 1:
          x_coord_left += 3
        num_cells_right = net_to_rightedge[idx][wire] // 2
        x_coord_right = num_cells_right * 7 + 2
        if net_to_rightedge[idx][wire] % 2 == 1:
          x_coord_right += 3
        f.write("rect " + str(x_coord_left) + " " + str(y_coord) + " " + str(x_coord_right + 1) + " " + str(y_coord + 1) + "\n")
      y_coord -= 2
    y_coord -= 7
  f.write("<< metal2 >>")

  f.write("<< end >>")
  f.close()