import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.inverse_folding.features import GVPInputFeaturizer
from esm.inverse_folding.gvp_modules import GVP, LayerNorm, GVPConvLayer
from esm.inverse_folding.util import rbf, normalize, nan_to_num, get_rotation_frames, rotate
from esm.inverse_folding.gvp_utils import flatten_graph, unflatten_graph


# Referenced from the https://github.com/facebookresearch/esm implementation
class GVPGraphEmbedding(GVPInputFeaturizer):
    def __init__(self):
        super().__init__()
        self.top_k_neighbors = 30
        self.num_positional_embeddings = 16
        self.remove_edges_without_coords = True
        # self.node_hidden_dim_scalar = 1024
        self.node_hidden_dim_scalar = 512
        self.node_hidden_dim_vector = 256
        self.edge_hidden_dim_scalar = 32
        self.edge_hidden_dim_vector = 1
        node_input_dim = (7, 3)
        edge_input_dim = (34, 1)
        node_hidden_dim = (self.node_hidden_dim_scalar, self.node_hidden_dim_vector)
        edge_hidden_dim = (self.edge_hidden_dim_scalar, self.edge_hidden_dim_vector)
        self.embed_node = nn.Sequential(
            GVP(node_input_dim, node_hidden_dim, activations=(None, None)),
            LayerNorm(node_hidden_dim, eps=1e-4)
        )
        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-4)
        )
        self.embed_confidence = nn.Linear(16, self.node_hidden_dim_scalar)

    def forward(self, coords, coord_mask, padding_mask, confidence):
        with torch.no_grad():
            node_features = self.get_node_features(coords, coord_mask)
            edge_features, edge_index = self.get_edge_features(
                coords, coord_mask, padding_mask)
        node_embeddings_scalar, node_embeddings_vector = self.embed_node(node_features)
        edge_embeddings = self.embed_edge(edge_features)

        rbf_rep = rbf(confidence, 0., 1.)
        node_embeddings = (
            node_embeddings_scalar + self.embed_confidence(rbf_rep),
            node_embeddings_vector
        )

        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)
        return node_embeddings, edge_embeddings, edge_index

    def get_edge_features(self, coords, coord_mask, padding_mask):
        X_ca = coords[:, :, 1]
        # Get distances to the top k neighbors
        E_dist, E_idx, E_coord_mask, E_residue_mask = GVPInputFeaturizer._dist(
            X_ca, coord_mask, padding_mask, self.top_k_neighbors)
        # Flatten the graph to be batch size 1 for torch_geometric package
        dest = E_idx
        B, L, k = E_idx.shape[:3]
        src = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        # After flattening, [B, E]
        E_dist = E_dist.flatten(1, 2)
        E_coord_mask = E_coord_mask.flatten(1, 2).unsqueeze(-1)
        E_residue_mask = E_residue_mask.flatten(1, 2)
        # Calculate relative positional embeddings and distance RBF
        pos_embeddings = GVPInputFeaturizer._positional_embeddings(
            edge_index,
            num_positional_embeddings=self.num_positional_embeddings,
        )
        D_rbf = rbf(E_dist, 0., 20.)
        # Calculate relative orientation
        X_src = X_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        X_dest = torch.gather(
            X_ca,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L * k, 3])
        )
        coord_mask_src = coord_mask.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)
        coord_mask_dest = torch.gather(
            coord_mask,
            1,
            edge_index[1, :, :].expand([B, L * k])
        )
        E_vectors = X_src - X_dest
        # For the ones without coordinates, substitute in the average vector
        E_vector_mean = torch.sum(E_vectors * E_coord_mask, dim=1,
                                keepdims=True) / torch.sum(E_coord_mask, dim=1, keepdims=True)
        E_vectors = E_vectors * E_coord_mask + E_vector_mean * ~E_coord_mask
        # Normalize and remove nans
        edge_s = torch.cat([D_rbf, pos_embeddings], dim=-1)
        edge_v = normalize(E_vectors).unsqueeze(-2)
        edge_s, edge_v = map(nan_to_num, (edge_s, edge_v))
        # Also add indications of whether the coordinates are present
        edge_s = torch.cat([
            edge_s,
            (~coord_mask_src).float().unsqueeze(-1),
            (~coord_mask_dest).float().unsqueeze(-1),
        ], dim=-1)
        edge_index[:, ~E_residue_mask] = -1
        if self.remove_edges_without_coords:
            edge_index[:, ~E_coord_mask.squeeze(-1)] = -1
        return (edge_s, edge_v), edge_index.transpose(0, 1)


class MySimpleRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_graph = GVPGraphEmbedding()
        # self.node_hidden_dim_scalar = 1024
        self.node_hidden_dim_scalar = 512
        self.node_hidden_dim_vector = 256
        self.edge_hidden_dim_scalar = 32
        self.edge_hidden_dim_vector = 1
        self.dropout = 0.1
        self.num_encoder_layers = 1
        # self.embed_dim = 512
        self.embed_dim = 256

        node_hidden_dim = (self.node_hidden_dim_scalar, self.node_hidden_dim_vector)
        edge_hidden_dim = (self.edge_hidden_dim_scalar, self.edge_hidden_dim_vector)

        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(
                node_hidden_dim,
                edge_hidden_dim,
                drop_rate=self.dropout,
                vector_gate=True,
                attention_heads=0,
                n_message=3,
                conv_activations=conv_activations,
                n_edge_gvps=0,
                eps=1e-4,
                layernorm=True,
            )
            for _ in range(self.num_encoder_layers)
        )
        gvp_out_dim = self.node_hidden_dim_scalar + (3 * self.node_hidden_dim_vector)
        self.embed_gvp_output = nn.Linear(gvp_out_dim, self.embed_dim)

    def forward_once(self, coords, coord_mask, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
            coords, coord_mask, padding_mask, confidence)

        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings, edge_index, edge_embeddings)

        gvp_out_scalars, gvp_out_vectors = unflatten_graph(node_embeddings, coords.shape[0])
        R = get_rotation_frames(coords)
        gvp_out_features = torch.cat([
            gvp_out_scalars,
            rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        
        return self.embed_gvp_output(gvp_out_features)

    def forward(self, batch):
        structure1, structure2 = batch
        coord_mask_q = torch.all(torch.all(torch.isfinite(structure1[0]), dim=-1), dim=-1).to(torch.device('cuda'))
        coord_mask_c = torch.all(torch.all(torch.isfinite(structure2[0]), dim=-1), dim=-1).to(torch.device('cuda'))
        q_embedding = self.forward_once(coords=structure1[0].to(torch.device('cuda')), coord_mask=coord_mask_q,
                                        padding_mask=structure1[4].to(torch.device('cuda')), 
                                        confidence=structure1[1].to(torch.device('cuda')))
        c_embedding = self.forward_once(coords=structure2[0].to(torch.device('cuda')), coord_mask=coord_mask_c,
                                        padding_mask=structure2[4].to(torch.device('cuda')), 
                                        confidence=structure2[1].to(torch.device('cuda')))
        q_embedding = torch.mean(q_embedding, dim=1)
        c_embedding = torch.mean(c_embedding, dim=1)
        return q_embedding, c_embedding
    
    def get_loss(self, embedding):
        q_embedding, c_embedding = embedding
        # print('q_embedding:', q_embedding.shape)
        # print('c_embedding:', c_embedding.shape)
        if q_embedding.shape[0] <= c_embedding.shape[0]:
            sim_mx = dot_product_scores(q_embedding, c_embedding)
        else:
            sim_mx = dot_product_scores(c_embedding, q_embedding)
        # print('sim_mx:', sim_mx.shape)
        label = torch.arange(sim_mx.shape[0], dtype=torch.long, device='cuda')
        # print('label:', label.shape)
        sm_score = F.log_softmax(sim_mx, dim=1).requires_grad_(True)
        # print('sm_score:', sm_score.shape)
        sm_score.to('cuda')
        loss = F.nll_loss(
            sm_score,
            label.to(sm_score.device),
            reduction="mean"
        )
        return loss

    def get_accuracy(self, embedding):
        q_embedding, c_embedding = embedding
        sim_mx = dot_product_scores(q_embedding, c_embedding)
        label = torch.arange(sim_mx.shape[0], dtype=torch.long)
        sm_score = F.log_softmax(sim_mx, dim=1)
        _, max_idxs = torch.max(sm_score, 1)
        correct_predictions_count = (
                max_idxs == label.to(sm_score.device)
        ).sum()
        return correct_predictions_count, sim_mx.shape[0]


def dot_product_scores(q_vectors, ctx_vectors):
    """
    calculates q->ctx scores for every row in ctx_vector
    """
    return torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
