import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import dgl

class LogNetConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, bias=True, num_channels=8, aggr_mode="sum"):
        super(LogNetConvLayer, self).__init__()
        self.num_channels = num_channels
        self._in_feats = in_feats
        self._out_feats = out_feats

        #Module Parameters (|IN_FEAT|x|OUT_FEAT|x|NUM_FEATURES|
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats, num_channels))
        if bias:
            # Module Parameters (|OUT_FEAT|x|NUM_FEATURES|
            self.bias = nn.Parameter(torch.Tensor(out_feats, num_channels))
        else:
            self.bias = None
        self.reset_parameters()
        self.activation = activation

        ##Aggregation mechanism for the different edge feature channels
        if aggr_mode == "concat":
            self.aggr_mode = "concat"
            self.final = nn.Linear(out_feats * self.num_channels, out_feats)
        elif aggr_mode == "sum":
            self.aggr_mode = "sum"
            self.final = nn.Linear(out_feats, out_feats)

    """Reinitialize model parameters."""
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            stdv = 1.0 / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    """Forward pass"""
    #Update node states via weighte convolution on edge features
    #Aggregates the different channels (sum, concat)
    #Update the node states
    #@input: a block from the sampler
    def forward(self, mfg, node_state_prev):
        #g is a subgraph from the Sampler
        #g=g[0]
        #g=dgl.block_to_graph(g)
        #@TODOmode fe

        node_state = node_state_prev

        #The returned graph object shares the feature data and graph structure of this graph
        mfg = mfg.local_var()

        #updated node state that take into account edge features
        new_node_states_per_channel = []
        #mfg=mfg['_N']
        #GlobalAttentionPooling(dgl.graph(([], []), num_nodes=g.num_nodes(ntype='a')), g.nodes['a'].data['x'])

        ## Weighted convolution for every channel of edge feature
        for channel_number in range(self.num_channels):
            node_state_c = node_state
            if self._out_feats < self._in_feats:
                mfg.ndata["feat_" + str(channel_number)] = torch.mm(node_state_c, self.weight[:, :, channel_number])
            else:
                mfg.ndata["feat_" + str(channel_number)] = node_state_c

            ## MESSAGE PASSING PHASE for each channel
            mfg.update_all(
                #MAP FUNCTION
                # binary operation mul between src feature and edge feature
                fn.src_mul_edge("feat_" + str(channel_number), "feat_" + str(channel_number), "m"),
                #REDUCE FUNCTION
                fn.sum("m", "feat_" + str(channel_number) + "_new")
            )

            node_state_c = mfg.ndata.pop("feat_" + str(channel_number) + "_new")
            if self._out_feats >= self._in_feats:
                node_state_c = torch.mm(node_state_c, self.weight[:, :, channel_number])
            if self.bias is not None:
                node_state_c = node_state_c + self.bias[:, channel_number]

            node_state_c = self.activation(node_state_c)
            new_node_states_per_channel.append(node_state_c)
        #aggregate the node states from the different channels
        if self.aggr_mode == "sum":
            node_states = torch.stack(new_node_states_per_channel, dim=1).sum(1)
        elif self.aggr_mode == "concat":
            node_states = torch.cat(new_node_states_per_channel, dim=1)
        #Final layer
        node_states = self.final(node_states)
        return node_states

class LogNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers, activation, dropout, aggr_mode="sum", device="cpu"):
        super(LogNet, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.layers = nn.ModuleList()

        #Use n_layers of LogNetConvLayers
        self.layers.append(LogNetConvLayer(n_input, n_hidden, activation=activation, aggr_mode=aggr_mode))
        for i in range(n_layers - 1):
            self.layers.append(LogNetConvLayer(n_hidden, n_hidden, activation=activation, aggr_mode=aggr_mode))
        self.pred_out = nn.Linear(n_hidden, n_output)
        self.device = device

    """Forward pass"""
    ##must be defined on the basis of the depth of the sampling (#layers)
   # def forward(self, mfgs, x):
    #    h_dst = x[:mfgs[0].num_dst_nodes()]
    #    h = self.conv1(mfgs[0], (x, h_dst))
    #    h = F.relu(h)
    #    h_dst = h[:mfgs[1].num_dst_nodes()]
    #    h = self.conv2(mfgs[1], (h, h_dst))
    #    return h

    #### mfgs will be  the set of graphs sampled for each layer
    def forward(self, mfgs, node_state=None):

        node_state = torch.ones(mfgs[0].number_of_nodes(), 1).float().to(self.device)
        #@TODO:  GIUS pass the i-th mfgs
        i=0
        print(self.layers)
        for layer in self.layers:
            node_state = F.dropout(node_state, p=self.dropout, training=self.training)
            node_state = layer(mfgs[i], node_state)
            node_state = self.activation(node_state)
            i=i+1
        out = self.pred_out(node_state)
        return out
