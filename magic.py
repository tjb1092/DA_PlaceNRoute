def magic(all_channels, doglegs, routing_list, net_to_leftedge, net_to_rightedge, outfile):
  if not outfile.endswith(".mag"):
    outfile += ".mag"
  f = open(outfile, "w")
  f.write("magic\n")
  # f.write("tech scmos\n")
  f.write("<< metal1 >>\n")
  # Produce metal1
  y_coord = 0
  for idx, channel in enumerate(all_channels):
    y_coord -= 8
    for track in channel:
      y_coord -= 2
      for wire in track:
        num_cells_left = net_to_leftedge[idx][wire] // 2
        x_coord_left = num_cells_left * 7 + 2
        if net_to_leftedge[idx][wire] % 2 == 1:
          x_coord_left += 3
        num_cells_right = net_to_rightedge[idx][wire] // 2
        x_coord_right = num_cells_right * 7 + 2
        if net_to_rightedge[idx][wire] % 2 == 1:
          x_coord_right += 3
        f.write("rect " + str(x_coord_left) + " " + str(y_coord) + " " + str(x_coord_right) + " " + str(y_coord + 1) + "\n")
    return
  f.write("<< metal2 >>")

  f.close()