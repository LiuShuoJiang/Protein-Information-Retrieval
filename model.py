import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.inverse_folding.util import get_encoder_output


class MyRepresentation(nn.Module):
    def __init__(self, huge_encoder):
        super(MyRepresentation, self).__init__()
        self.gvp_huge_model = huge_encoder.encoder

    def forward_once(self, coords, padding_mask, confidence):
        encoder_out = self.gvp_huge_model.forward(coords, padding_mask, confidence, return_all_hiddens=False)
        # remove beginning and end (bos and eos tokens)
        return encoder_out['encoder_out'][0][1:-1, 0]

    def forward(self, batch):
        structure1, structure2 = batch
        q_embedding = self.forward_once(structure1[0].to(torch.device('cuda')), structure1[4].to(torch.device('cuda')), structure1[1].to(torch.device('cuda')))
        c_embedding = self.forward_once(structure2[0].to(torch.device('cuda')), structure2[4].to(torch.device('cuda')), structure2[1].to(torch.device('cuda')))
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
        max_score, max_idxs = torch.max(sm_score, 1)
        correct_predictions_count = (
                max_idxs == label.to(sm_score.device)
        ).sum()
        return correct_predictions_count, sim_mx.shape[0]


def dot_product_scores(q_vectors, ctx_vectors):
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vectors: n1 x D
    :param ctx_vectors: n2 x D
    :return: n1 x n2
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
