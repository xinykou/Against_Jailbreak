
import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def add_single_hooks(module_forward_pre_hooks):
    handles = []
    for module, hook in module_forward_pre_hooks:
        handles.append(module.register_forward_pre_hook(hook))
    return handles

def remove_single_hooks(handles):
    for h in handles:
        h.remove()

global constant_sample_id
constant_sample_id = 0
global layer_list
layer_list = []
def get_direction_ablation_output_hook(layer_id: int,
                                       mask_index=None,
                                       mask_direction=None,
                                       coefficient=None,
                                       pert_coef=None,
                                       mask_variance_device=None,
                                       mask_harmless_direction=None,
                                       refusal_feature_type=None
                                       ):
    def hook_fn(module, input, output, sample_id=None):
        global constant_sample_id
        global layer_list
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        current_direction = mask_direction[layer_id, :].to(activation) if mask_direction.device != activation.device else mask_direction[layer_id, :]
        current_direction = current_direction.to(dtype=activation.dtype)

        if refusal_feature_type == 'constant':
            activation = activation - coefficient * current_direction
        elif refusal_feature_type == 'variable':
            current_variance_last_dim = mask_variance_device[layer_id, :].to(activation) if mask_variance_device.device != activation.device else mask_variance_device[layer_id, :]
            current_variance_last_dim = current_variance_last_dim.to(dtype=activation.dtype)
            # current_harmless_direction = mask_harmless_direction[layer_id, :].to(activation) if mask_harmless_direction.device != activation.device else mask_harmless_direction[layer_id, :]
            # current_harmless_direction = current_harmless_direction.to(dtype=activation.dtype)
            # mask_activation = activation * mask_index[layer_id, :]
            # distance_ratio = torch.norm(mask_activation - current_harmless_direction, dim=-1, keepdim=True) / torch.norm(current_direction, dim=-1, keepdim=True)
            # if (constant_sample_id != sample_id):
            #     constant_sample_id = sample_id
            #     layer_list.clear()
            #     layer_list.append(layer_id)
            #     count = torch.sum((distance_ratio > 0) & (distance_ratio < 1))
            #     # print("数量：", count.item())
            # elif not (layer_id in layer_list):
            #     count = torch.sum((distance_ratio > 0) & (distance_ratio < 1))
            #     # print("数量：", count.item())
            #     layer_list.append(layer_id)
            #
            # distance_ratio = torch.clamp(distance_ratio, 0, 1).to(dtype=activation.dtype)
            # activation = activation - coefficient * current_direction * distance_ratio
            activation = activation - coefficient * current_direction
            mean_device = torch.zeros_like(current_variance_last_dim, device=activation.device)
            epsilon = torch.normal(mean=mean_device,
                                   std=current_variance_last_dim,
                                   )
            epsilon = epsilon.to(dtype=activation.dtype)

            activation = activation - pert_coef * epsilon
        elif refusal_feature_type == 'mask_and_map':
            direction = mask_direction[layer_id, :]/(mask_direction[layer_id, :].norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to(activation)
            activation -= (activation @ direction).unsqueeze(-1) * direction

        else:
            raise ValueError("refusal_feature_type must be either 'constant' or 'variable'")

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn


def get_all_direction_ablation_hooks(
    model_base,
    mask_index_device: Float[Tensor, 'd_model'],
    mask_direction: Float[Tensor, 'd_model'],
    coefficient: Float[Tensor, ''],
    mask_variance_device: Float[Tensor, ''],
    mask_harmless_direction: Float[Tensor, 'd_model'],
    pert_coef: Float[Tensor, ''],
    selected_layer_ids=None,
    refusal_feature_type=None
):
    order_layer_ids = [i for i in range(len(selected_layer_ids))]
    fwd_hooks = [(model_base.model.layers[layer], get_direction_ablation_output_hook(mask_index=mask_index_device,
                                                                                     mask_direction=mask_direction,
                                                                                     layer_id=order_id,
                                                                                     coefficient=coefficient,
                                                                                     pert_coef=pert_coef,
                                                                                     mask_variance_device=mask_variance_device,
                                                                                     mask_harmless_direction=mask_harmless_direction,
                                                                                     refusal_feature_type=refusal_feature_type))
                                                                                     for order_id, layer in zip(order_layer_ids, selected_layer_ids)]
    # fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    # fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_pre_hooks = []
    return fwd_pre_hooks, fwd_hooks

def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn