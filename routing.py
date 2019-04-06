from collections import defaultdict

def create_vcg(top, bottom, unassigned):
  vcg = defaultdict(set)
  for column in zip(top, bottom):
    if column[0] != 0 and column[1] != 0 and column[0] in unassigned and column[1] in unassigned:
      vcg[column[0]].add(column[1])
  return vcg

def has_parents(net, vcg):
  for par, children in vcg.items():
    if net in children:
      return True
  return False


def routing(routing_list):
  zero_list = [0 for x in routing_list[0]]
  routing_list = [zero_list] + routing_list + [zero_list]
  all_tracks = []
  channel_width = len(zero_list)
  it = iter(routing_list)
  for top in it:
    bottom = next(it)
    # print(*zip(top, bottom))
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
    nets_unassigned = all_nets.copy()
    tracks = []
    while nets_unassigned:
      vcg = create_vcg(top, bottom, nets_unassigned)
      # print(vcg)
      nets_by_leftedge = sorted(list(nets_unassigned), key=lambda x: net_to_leftedge[x])
      # print(nets_by_leftedge)
      track = []
      track_right_edge = -1
      for net in nets_by_leftedge:
        if not has_parents(net, vcg) and net_to_leftedge[net] > track_right_edge:
          # print("net " + str(net) + " had no parents")
          # print(nets_unassigned)
          nets_unassigned.remove(net)
          track += [net]
          track_right_edge = net_to_rightedge[net]
      tracks += [track]
    print(tracks)
    all_tracks += [tracks]
  print("done")
  print(all_tracks)
