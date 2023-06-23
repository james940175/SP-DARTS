import math, random, torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import sys
sys.path.insert(0, '../')
from cell_operations import OPS


class NAS201SearchCell(nn.Module):

  def __init__(self, C_in, C_out, stride, max_nodes, op_names, dropout, affine=False, track_running_stats=True):
    super(NAS201SearchCell, self).__init__()

    self.op_names  = deepcopy(op_names)
    self.edges     = nn.ModuleDict()
    self.max_nodes = max_nodes
    self.in_dim    = C_in
    self.out_dim   = C_out
    
    self.dropout_rate = dropout
    
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        xlists = []
        if j == 0:
          for op_name in op_names:
            op = OPS[op_name](C_in , C_out, stride, affine, track_running_stats)
            if op_name == 'skip_connect' and dropout > 0:
              op = nn.Sequential(op, nn.Dropout(self.dropout_rate))
            xlists.append(op)
        else:
          for op_name in op_names:
            op = OPS[op_name](C_in , C_out,      1, affine, track_running_stats)
            if op_name == 'skip_connect' and dropout > 0:
              op = nn.Sequential(op, nn.Dropout(self.dropout_rate))
            xlists.append(op)
          
        self.edges[ node_str ] = nn.ModuleList( xlists )
    self.edge_keys  = sorted(list(self.edges.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edges)

  def extra_repr(self):
    string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    return string

  def forward(self, inputs, weightss):
    return self._forward(inputs, weightss)

  def _forward(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]
