import torch
import torch.nn.functional as F


def compute_sim_matrix(feats):
    """
    Takes in a batch of features of size (bs, feat_len).
    """
    sim_matrix = F.cosine_similarity(feats.unsqueeze(2).expand(-1, -1, feats.size(0)),
                                     feats.unsqueeze(2).expand(-1, -1, feats.size(0)).transpose(0, 2),
                                     dim=1)

    return sim_matrix


def compute_target_matrix(labels):
    """
    Takes in a label vector of size (bs)
    """
    label_matrix = labels.unsqueeze(-1).expand((labels.shape[0], labels.shape[0]))
    trans_label_matrix = torch.transpose(label_matrix, 0, 1)
    target_matrix = (label_matrix == trans_label_matrix).type(torch.float)

    return target_matrix


def contrastive_loss(pred_sim_matrix, target_matrix, temperature, labels):
    return F.kl_div(F.softmax(pred_sim_matrix / temperature).log(), F.softmax(target_matrix / temperature),
                    reduction="batchmean", log_target=False)
