import torch
import os
import gc
from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from utilss.hook_utils import add_hooks
from .feature_index_localization import get_safety_feature


def get_activations_pre_hook(layer, cache: Float[Tensor, "batch_size layer d_model"], positions: int):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] = activation[:, positions, :]

    return hook_fn

def get_activations_output_hook(layer, cache: Float[Tensor, "batch_size layer d_model"], positions: int):
    def hook_fn(module, input, output):
        activation: Float[Tensor, "batch_size seq_len d_model"] = output[0].clone().to(cache)
        cache[:, layer] = activation[:, positions, :]

    return hook_fn

def get_activations(model,
                    instructions,
                    block_modules: List[torch.nn.Module],
                    batch_size=8,
                    layer_ids=None):
    torch.cuda.empty_cache()

    n_layers = len(layer_ids)
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    bt_last_token_activations = torch.zeros((batch_size, n_layers, d_model), dtype=torch.float32, device=model.device)
    last_token_activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float32, device=model.device)

    cache_layer_ids = [i for i in range(len(layer_ids))]
    fwd_hooks = [(block_modules[layer],
                  get_activations_output_hook(layer=cache_layer_id, cache=bt_last_token_activations,
                                              positions=-1)) for cache_layer_id, layer in zip(cache_layer_ids, layer_ids)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = instructions[i:i + batch_size]
        input_ids = torch.cat([inp["input_ids"].to(model.device) for inp in inputs], dim=0)
        attention_mask = torch.cat([inp["attention_mask"].to(model.device) for inp in inputs], dim=0)
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks), torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_token_activations[i:i + batch_size] = bt_last_token_activations
            gc.collect()

    return last_token_activations


def get_diff(model,
             harmful_instructions_tokenized,
             harmless_instructions_tokenized,
             block_modules: List[torch.nn.Module],
             batch_size=2,
             layer_ids=None):
    activations_harmful = get_activations(model,
                                          harmful_instructions_tokenized,
                                          block_modules,
                                          batch_size=batch_size,
                                          layer_ids=layer_ids)
    activations_harmless = get_activations(model,
                                           harmless_instructions_tokenized,
                                           block_modules,
                                           batch_size=batch_size,
                                           layer_ids=layer_ids)

    diff: Float[Tensor, "n_samples n_layers d_model"] = activations_harmful - activations_harmless

    return diff


def generate_masks(model_base,
                   harmful_instructions_tokenized,
                   harmless_instructions_tokenized,
                   selected_layer_ids,
                   top_k=30,
                   feature_type=None
                   ):

    batch_size = 4
    layer_ids = selected_layer_ids
    activations_harmful = get_activations(model_base,
                                          harmful_instructions_tokenized,
                                          model_base.model.layers,
                                          batch_size=batch_size,
                                          layer_ids=layer_ids)
    activations_harmless = get_activations(model_base,
                                           harmless_instructions_tokenized,
                                           model_base.model.layers,
                                           batch_size=batch_size,
                                           layer_ids=layer_ids)

    _diffs = activations_harmful - activations_harmless
    # n_layers * d_model
    masks_indexes, mean_last_dim, variance_last_dim = get_safety_feature(_diffs, top_k=top_k, return_mean=True, return_variance=True)

    mean_harmless_last_dim = None
    variance_harmless_last_dim = None
    if feature_type == 'constant' or feature_type == 'mask_and_map':
        pass
    elif feature_type == 'variable':
        # n_layers * d_model * n_samples
        safety_vectors_permuted = activations_harmless.permute(1, 2, 0)
        mean_harmless_last_dim = safety_vectors_permuted.mean(dim=-1)  # n_layers * d_model
        variance_harmless_last_dim = safety_vectors_permuted.var(dim=-1)  # n_layers * d_model

    else:
        raise ValueError(f"feature_type must be either 'constant' or 'variable' but got {feature_type}")

    return masks_indexes, mean_last_dim, variance_last_dim, mean_harmless_last_dim
