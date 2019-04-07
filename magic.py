import time

def magic(all_channels, doglegs, routing_list, net_to_leftedge, net_to_rightedge, outfile):
  start = time.time()
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
      if top[i] != 0 or top[i+1] != 0 or bottom[i] != 0 or bottom[i+1] != 0:
        top_left = False
        top_right = False
        bottom_left = False
        bottom_right = False
        if top[i] != 0:
          top_left = True
        if top[i + 1] != 0:
          top_right = True
        if bottom[i] != 0:
          bottom_left = True
        if bottom[i + 1] != 0:
          bottom_right = True
        cells_here += [(i // 2, top_left, top_right, bottom_left, bottom_right)]
    standard_cells += [cells_here]
  y_coord = 0
  for idx, row in enumerate(standard_cells):
    y_coord -= 2 * len(all_channels[idx]) + 1
    for cell in row:
      x_coord = cell[0] * 7
      f.write("rect " + str(x_coord + 1) + " " + str(y_coord - 5) + " " + str(x_coord + 7) + " " + str(y_coord + 1) + "\n")
    y_coord -= 8

  f.write("<< polysilicon >>\n")
  y_coord = 0
  for idx, row in enumerate(standard_cells):
    y_coord -= 2 * len(all_channels[idx]) + 1
    for cell in row:
      x_coord = cell[0] * 7
      if cell[1]:
        f.write("rect " + str(x_coord + 2) + " " + str(y_coord) + " " + str(x_coord + 3) + " " + str(y_coord + 2) + "\n")
      if cell[2]:
        f.write("rect " + str(x_coord + 5) + " " + str(y_coord) + " " + str(x_coord + 6) + " " + str(y_coord + 2) + "\n")
      if cell[3]:
        f.write("rect " + str(x_coord + 2) + " " + str(y_coord - 6) + " " + str(x_coord + 3) + " " + str(y_coord - 4) + "\n")
      if cell[4]:
        f.write("rect " + str(x_coord + 5) + " " + str(y_coord - 6) + " " + str(x_coord + 6) + " " + str(y_coord - 4) + "\n")
    y_coord -= 8

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
    y_coord -= 9

  # Produce metal2
  f.write("<< metal2 >>\n")
  y_coord = 0
  y_coord_top = 0
  y_coord_bottom = 0
  for idx, channel in enumerate(all_channels):
    y_coord_bottom -= 2 * len(channel)
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
        top_row = routing_list[idx*2]
        bottom_row = routing_list[idx*2 + 1]
        leftedge = net_to_leftedge[idx][wire]
        rightedge = net_to_rightedge[idx][wire]
        if top_row[leftedge] == wire:
          f.write("rect " + str(x_coord_left) + " " + str(y_coord) + " " + str(x_coord_left + 1) + " " + str(y_coord_top + 1) + "\n")
        if bottom_row[leftedge] == wire:
          f.write("rect " + str(x_coord_left) + " " + str(y_coord_bottom) + " " + str(x_coord_left + 1) + " " + str(y_coord + 1) + "\n")
        did_rightedge = False
        if top_row[rightedge] == wire:
          f.write("rect " + str(x_coord_right) + " " + str(y_coord) + " " + str(x_coord_right + 1) + " " + str(y_coord_top + 1) + "\n")
          did_rightedge = True
        if bottom_row[rightedge] == wire:
          f.write("rect " + str(x_coord_right) + " " + str(y_coord_bottom) + " " + str(x_coord_right + 1) + " " + str(y_coord + 1) + "\n")
          did_rightedge = True
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
      y_coord -= 2
    y_coord -= 9
    y_coord_top = y_coord + 2
    y_coord_bottom = y_coord

  f.write("<< end >>\n")
  f.close()

  end = time.time()
  print("Time Elapsed: " + str(end - start)[:4] + " seconds.")