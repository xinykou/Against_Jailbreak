import torch

def get_safety_feature(safety_vectors, top_k=30,
                       return_mean=False, return_variance=True):

    n_samples, n_layers, d_model = safety_vectors.size()
    top_k_nums = round(top_k / 100 * d_model)
    masks_indexes = torch.zeros((n_layers, d_model), dtype=torch.float32, device=safety_vectors.device)
    # n_layers * d_model * n_samples
    safety_vectors_permuted = safety_vectors.permute(1, 2, 0)

    mean_last_dim = safety_vectors_permuted.mean(dim=-1)   # n_layers * d_model
    variance_last_dim = safety_vectors_permuted.var(dim=-1)

    # get minimum value of top_k_nums
    top_k_values, top_k_indexes = torch.topk(variance_last_dim, top_k_nums, dim=-1, largest=False, sorted=True)

    for i in range(n_layers):
        masks_indexes[i, top_k_indexes[i]] = 1

    if return_mean and return_variance:
        return masks_indexes, mean_last_dim, variance_last_dim
    elif return_mean and not return_variance:
        return masks_indexes, mean_last_dim
    else:
        return masks_indexes, None


