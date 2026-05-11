from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .gatedge import GATEdge
from .loss_func import sce_loss
from graphmae.utils import create_norm, drop_edge


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True, edge_in_dim: Optional[int] = 0) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gatedge": # Our new GAT with edge features
        # For GATEdge, num_hidden is per head, out_dim for GATEdge is also per head for intermediate
        # The GATEdge class handles the nhead and nhead_out internally for layer construction.
        mod = GATEdge(
            in_dim=in_dim,
            edge_in_dim=edge_in_dim, # Pass edge_in_dim
            num_hidden=num_hidden, # This is per head for GATEdgeConv layers
            out_dim=out_dim, # This is per head for the output GATEdgeConv layer
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            edge_in_dim: Optional[int] = 0, # Added edge_in_dim
            edge_feat_name: Optional[str] = 'label' # Name of edge features in g.edata
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.edge_feat_name = edge_feat_name
        self.edge_in_dim = edge_in_dim # Store edge_in_dim

        if isinstance(activation, str):
            if activation == "relu": act_fn = nn.ReLU()
            elif activation == "elu": act_fn = nn.ELU()
            elif activation == "leaky_relu": act_fn = nn.LeakyReLU(negative_slope)
            elif activation == "prelu": act_fn = nn.PReLU()
            else: act_fn = Identity() # Or raise error
        else: # Assume it's already an nn.Module
            act_fn = activation

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0

        concat_out = True
        if encoder_type in ("gat", "dotgat", "gatedge"):
            enc_num_hidden_per_head = num_hidden // nhead 
            enc_nhead = nhead

            
            encoder_hidden_per_head = num_hidden // enc_nhead
            encoder_out_dim_for_setup = num_hidden // enc_nhead # This is the target per-head dim for GATEdge's last layer.
                                                               # Final output will be this * nhead_out if concat.
            dec_in_dim = num_hidden 

        else: # GIN, GCN
            encoder_hidden_per_head = num_hidden # Not per-head for these models
            encoder_out_dim_for_setup = num_hidden
            enc_nhead = 1 # Not applicable
            dec_in_dim = num_hidden


        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            edge_in_dim=self.edge_in_dim, # Pass edge_in_dim
            num_hidden=encoder_hidden_per_head, # Per-head hidden dim
            out_dim=encoder_out_dim_for_setup,  # Per-head output dim for last layer
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead, # Num heads for output layer
            concat_out=concat_out, # Always concat encoder output heads to get full dec_in_dim
            activation=act_fn,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        if decoder_type in ("gat", "dotgat", "gatedge"):
            decoder_hidden_per_head = num_hidden // nhead # Assuming decoder also uses heads
            decoder_out_dim_for_setup = in_dim # Target original feature dim (potentially per head if not last step)
            # If decoder has 1 layer and outputs in_dim, and uses nhead_out,
            # then out_dim per head should be in_dim // nhead_out if concat_out=True.
            # Or in_dim if concat_out=False (average).
            # For simplicity, let decoder output `in_dim` and handle heads internally.
            # If decoder is GATEdge, it needs edge_in_dim too.
            if decoder_type == "gatedge":
                 dec_edge_in_dim = self.edge_in_dim
                 dec_out_per_head = in_dim // nhead_out if concat_out else in_dim
            else: # GAT, DotGAT
                 dec_edge_in_dim = 0 # These don't take edge features
                 dec_out_per_head = in_dim // nhead_out if concat_out else in_dim

        else: # MLP, Linear, GCN, GIN for decoder
            decoder_hidden_per_head = num_hidden # Or some other suitable hidden dim for MLP/GCN
            decoder_out_dim_for_setup = in_dim
            dec_edge_in_dim = 0

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim, # Input to decoder is encoder's output
            edge_in_dim=dec_edge_in_dim, # Pass edge_in_dim for gatedge decoder
            num_hidden=decoder_hidden_per_head, # Hidden dim (per head if applicable)
            out_dim=dec_out_per_head,  # Output dim (per head if applicable, aiming for `in_dim` total)
            num_layers=1, # Decoder is 1 layer
            nhead=nhead, # Use nhead_out for decoder's attention heads
            nhead_out=nhead_out, # Final layer heads
            activation=act_fn,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=concat_out, # Decoder output heads are concatenated to achieve `in_dim`
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    # def forward(self, g, x):
    #     # ---- attribute reconstruction ----
    #     loss = self.mask_attr_prediction(g, x)
    #     loss_item = {"loss": loss.item()}
    #     return loss, loss_item
    def forward(self, g, x, edge_x: Optional[torch.Tensor] = None): # Added edge_x
        # ---- attribute reconstruction ----
        # If edge_x is not provided but needed, try to get from graph or raise error
        if self._encoder_type == "gatedge" or self._decoder_type == "gatedge":
            if edge_x is None:
                if self.edge_feat_name and self.edge_feat_name in g.edata:
                    edge_x = g.edata[self.edge_feat_name]
                elif self.edge_in_dim > 0: # If edge_in_dim is configured, features are expected
                    raise ValueError("GATEdge encoder/decoder requires edge features (edge_x or g.edata[edge_feat_name]), but none were provided.")
                # If self.edge_in_dim is 0, GATEdgeConv handles it (fc_edge_attn is None)
            
            # Ensure edge_x has features if edge_in_dim > 0
            if self.edge_in_dim > 0 and (edge_x is None or edge_x.shape[1] == 0):
                 raise ValueError(f"GATEdge expects edge features with dimension {self.edge_in_dim}, but got {edge_x.shape if edge_x is not None else 'None'}")


        loss = self.mask_attr_prediction(g, x, edge_x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, g, x, edge_x: Optional[torch.Tensor] = None): # Added edge_x
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        # Pass edge features to encoder if it's GATEdge
        encoder_args = [use_g, use_x]
        if self._encoder_type == "gatedge":
            encoder_args.append(edge_x if self.edge_in_dim > 0 else None)
        
        enc_rep, all_hidden = self.encoder(*encoder_args, return_hidden=True)
        
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            rep[mask_nodes] = 0 # Re-mask for GNN decoders

        # Pass edge features to decoder if it's GATEdge
        decoder_args = []
        if self._decoder_type in ("mlp", "linear") : # Corrected "liear" to "linear"
            decoder_args.append(rep)
        else: # GNN decoders
            decoder_args.extend([pre_use_g, rep])
            if self._decoder_type == "gatedge":
                decoder_args.append(edge_x if self.edge_in_dim > 0 else None)
        
        recon = self.decoder(*decoder_args)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        # print(x_init)
        # print(x_rec)
        loss = self.criterion(x_rec, x_init)
        return loss



    def embed(self, g, x, edge_x: Optional[torch.Tensor] = None): # Added edge_x
        encoder_args = [g, x]
        if self._encoder_type == "gatedge":
            if edge_x is None and self.edge_feat_name and self.edge_feat_name in g.edata:
                edge_x = g.edata[self.edge_feat_name]
            # Add edge_x if GATEdge and edge_in_dim > 0
            if self.edge_in_dim > 0:
                 if edge_x is None:
                     raise ValueError("GATEdge encoder requires edge_x for embedding when edge_in_dim > 0.")
                 encoder_args.append(edge_x)
            # If edge_in_dim is 0, GATEdgeConv handles it, no need to pass None explicitly unless API demands it
            # The GATEdge model's forward takes edge_x, which can be None if GATEdgeConv handles it.
            # Our GATEdge forward takes edge_x, and GATEdgeConv handles if edge_feat is None or edge_feat_dim is 0.
            elif self.edge_in_dim == 0: # GATEdgeConv has fc_edge_attn=None
                 encoder_args.append(None)


        rep = self.encoder(*encoder_args)
        return rep
        
    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])