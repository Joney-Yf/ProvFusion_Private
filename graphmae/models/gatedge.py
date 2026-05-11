import torch
import torch.nn as nn
import torch.nn.functional as F # For F.elu if needed
import dgl
import dgl.function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

class GATEdge(nn.Module):
    def __init__(self,
                 in_dim: int,
                 edge_in_dim: int,
                 num_hidden: int, # Per-head hidden dimension
                 out_dim: int,    # Per-head output dimension for the last layer
                 num_layers: int,
                 nhead: int,
                 nhead_out: int,
                 activation, # nn.Module instance
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm, # create_norm(norm) instance or None
                 concat_out=False,
                 encoding=False):
        super(GATEdge, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation # Usually applied in GATEdgev2Conv
        self.concat_out = concat_out
        self.norm = norm # Norm applied after each layer's output (if not None)
        self.norm = norm(nhead * num_hidden)
        self.encoding = encoding # For PreModel specific logic

        current_in_dim = in_dim
        current_nhead = nhead

        for l in range(num_layers):
            is_last_layer = (l == num_layers - 1)
            
            # Determine per-head output dimension and number of heads for this layer
            layer_nhead = nhead_out if is_last_layer else nhead
            
            if is_last_layer:
                # If encoding, out_dim is num_hidden (per head)
                # Else, it's the provided out_dim (per head)
                layer_out_feats_per_head = num_hidden if encoding else out_dim
            else:
                layer_out_feats_per_head = num_hidden

            self.gat_layers.append(GATEdgeConv(
                current_in_dim,
                edge_in_dim,
                layer_out_feats_per_head,
                layer_nhead,
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                self.activation if not is_last_layer or self.concat_out or self.encoding else None, # Activation for intermediate or if specified for output
                allow_zero_in_degree=True # Or make this a parameter
            ))
            
            # Update input dimension for the next layer
            if self.concat_out or (is_last_layer and self.encoding) or not is_last_layer : # Always concat for intermediate
                current_in_dim = layer_out_feats_per_head * layer_nhead
            else: # Averaging for last layer if not concat_out and not encoding
                current_in_dim = layer_out_feats_per_head


    def forward(self, g, x, edge_x, return_hidden=False):
        h = x
        hidden_list = []
        
        for l in range(self.num_layers):
            is_last_layer = (l == self.num_layers - 1)
            
            # GATEdgev2Conv returns (N, H, D_out_per_head)
            h = self.gat_layers[l](g, h, edge_x) # Pass edge_x
            
            # Concatenate or average heads
            if not is_last_layer: # Intermediate layers always concatenate
                h = h.flatten(start_dim=1) # (N, H * D_out_per_head)
            else: # Last layer
                if self.concat_out or self.encoding:
                    h = h.flatten(start_dim=1)
                else: # Average
                    h = h.mean(dim=1)
            
            if self.norm is not None and not is_last_layer: # Apply norm to intermediate layers after head aggregation
                 h = self.norm(h)
            
            if return_hidden:
                hidden_list.append(h)

        # Apply norm to the final output if concat_out is false and norm is specified
        # (because norm is usually applied *before* final activation or output)
        # This logic might need adjustment based on desired norm placement for final layer
        if self.norm is not None and is_last_layer and not (self.concat_out or self.encoding) :
            # if not self.concat_out and not self.encoding, h is (N, D_out_per_head)
            # norm usually expects (N, TotalFeatures)
            # If norm is LayerNorm, it will work.
            # Let's assume norm is applied after head aggregation for all layers except the very last one if not concatenating.
            # The current loop applies norm to intermediate layers. If you need it on the final averaged output, add it here.
            pass


        if return_hidden:
            return h, hidden_list
        return h

# --- GATEdgev2Conv Layer (GATv2 style attention with edge features) ---
class GATEdgeConv(nn.Module):
    def __init__(self,
                 node_feats_dim,
                 edge_feats_dim, # Dimension of edge features
                 out_feats,      # Output feature dimension per head
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True): # Bias for linear layers
        super(GATEdgeConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats_dim)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._edge_feats_dim = edge_feats_dim

        # Linear transformation for node features (W in GATv2)
        # This W is applied to the concatenation [h_i || h_j] OR separately to h_i and h_j then summed/concatenated
        # DGL's GATv2Conv applies W_l to h_i and W_r to h_j, then sums them before LeakyReLU.
        # W_l projects src_feats to num_heads * out_feats
        # W_r projects dst_feats to num_heads * out_feats
        self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)

        # Linear transformation for edge features, if used in attention
        if self._edge_feats_dim > 0:
            self.fc_edge = nn.Linear(self._edge_feats_dim, out_feats * num_heads, bias=False)
        else:
            self.register_buffer('fc_edge', None) # Or self.fc_edge = None

        # Attention mechanism's learnable vector 'a' (applied after LeakyReLU)
        # It takes input of size out_feats and outputs 1 (per head)
        self.attn_weights = nn.Parameter(torch.Tensor(1, num_heads, out_feats)) # a^T

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_feats))
        else:
            self.register_buffer('bias', None)

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu') # GAT paper uses gain for LeakyReLU with negative_slope
        if hasattr(self, 'fc_src'):
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        if hasattr(self, 'fc_dst'):
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if hasattr(self, 'fc_edge') and self.fc_edge is not None:
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_weights, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, edge_feat=None, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise dgl.DGLError('There are 0-in-degree nodes in the graph.')

            # Feature dropout
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block: # For bipartite graphs (sampling)
                feat_dst = feat_dst[:graph.number_of_dst_nodes()]
            
            # Apply W_l and W_r (fc_src, fc_dst)
            # h_src shape: (N_src, H, D_out), h_dst shape: (N_dst, H, D_out)
            h_src = self.fc_src(feat_src).view(-1, self._num_heads, self._out_feats)
            h_dst = self.fc_dst(feat_dst).view(-1, self._num_heads, self._out_feats)

            graph.srcdata.update({'h_src': h_src}) # Store transformed source features
            graph.dstdata.update({'h_dst': h_dst}) # Store transformed dest features (for residual)

            # Apply W_edge to edge features if they exist
            if self.fc_edge is not None and edge_feat is not None:
                h_edge = self.fc_edge(edge_feat).view(-1, self._num_heads, self._out_feats)
                graph.edata.update({'h_edge': h_edge})


            # --- GATv2 Attention Calculation with Edge Features ---
            # e_ij = a^T LeakyReLU( W_l h_i + W_r h_j + W_e e_ij )  (if edge features are added this way)
            # Or, e_ij = a^T LeakyReLU( W_l h_i + W_r h_j ) and edge features are used differently.
            # Let's stick to adding transformed edge features into the sum before LeakyReLU.
            
            # Step 1: Compute W_l h_i + W_r h_j (or + W_e e_ij) on edges
            # graph.apply_edges(fn.u_add_v('h_src', 'h_dst', 'e_sum_nodes')) # e_sum_nodes = Wh_i + Wh_j
            def edge_attn_fn(edges):
                # Wh_i from source, Wh_j from destination
                el = edges.src['h_src']  # (num_edges, num_heads, out_feats)
                er = edges.dst['h_dst']  # (num_edges, num_heads, out_feats)
                
                # Summing transformed node features (GATv2 style)
                node_attn_input = el + er 

                # Add transformed edge features if available
                if 'h_edge' in edges.data:
                    edge_attn_comp = edges.data['h_edge'] # (num_edges, num_heads, out_feats)
                    attn_input = node_attn_input + edge_attn_comp
                else:
                    attn_input = node_attn_input
                
                # Apply LeakyReLU
                activated_attn_input = self.leaky_relu(attn_input) # (num_edges, num_heads, out_feats)
                
                # Multiply by attention vector 'a' (self.attn_weights)
                # (1, num_heads, out_feats) * (num_edges, num_heads, out_feats) -> sum over last dim
                attn_scores = (activated_attn_input * self.attn_weights).sum(dim=-1, keepdim=True) # (num_edges, num_heads, 1)
                return {'e': attn_scores}

            graph.apply_edges(edge_attn_fn)
            
            # Softmax attention scores (alpha_ij)
            alpha = self.attn_drop(dgl.ops.edge_softmax(graph, graph.edata.pop('e'))) # (num_edges, num_heads, 1)

            # Message passing: \sum_j alpha_ij * Wh_j (or more generally, alpha_ij * transformed_src_feat)
            # GAT usually uses Wh_i (from source node) as the message after attention.
            # Here, h_src is already W_l h_i.
            graph.edata['alpha'] = alpha
            graph.update_all(fn.u_mul_e('h_src', 'alpha', 'm'), # message = h_src * alpha
                             fn.sum('m', 'ft'))               # aggregate messages
            
            rst = graph.dstdata['ft'] # (N_dst, H, D_out)

            # Residual connection
            if self.res_fc is not None:
                res = self.res_fc(feat_dst).view(-1, self._num_heads, self._out_feats)
                rst = rst + res
            
            # Bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats) # Or reshape bias earlier

            # Activation
            if self.activation:
                rst = self.activation(rst)
            
            if get_attention:
                return rst, alpha
            else:
                return rst