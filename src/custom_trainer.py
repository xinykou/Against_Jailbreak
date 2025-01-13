from trainer import Trainer
import torch
import gc
from torch.nn.functional import cosine_similarity
from refusal_feature.utilss.hook_utils import remove_single_hooks, add_single_hooks, get_all_direction_ablation_hooks, \
    add_hooks
import sys
import os
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(current_directory+"/"+"scav")
from scav.perturbation import Perturbation
from functools import partial

class CustomTrainer(Trainer):

    def __init__(self, lorra_args=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation_fwd_pre_hooks = None
        self.num_training_steps = self.args.max_steps
        self.lorra_args = lorra_args
        self.tokenizer = kwargs.get("tokenizer")
        self.current_training_step = 0
        self.coefficient_step = self.args.gradient_accumulation_steps
        mask_layer = None
        # if "llama3" in self.args.classifier_path.lower() or "mistral" in self.args.classifier_path.lower():
        #     layers_ids = [id for id in range(32)]
        #     mask_layer = [i for i in layers_ids if i not in self.selected_layer_ids]
        #
        # self.classifier = torch.load(self.args.classifier_path)
        # self.perturbation = Perturbation(self.classifier,
        #                          target_probability=0.05,
        #                          accuracy_threshold=0.9,
        #                          layer_mask=mask_layer
        #                          )
        self.clfr_hooks = []
        self.n_layer = self.model.base_model.config.num_hidden_layers

    # def clfr_register_hooks(self, model):
    #     def _hook_fn(module, input, output, layer_idx):
    #         if self.perturbation is not None:
    #             output = self.perturbation.get_perturbation(output, layer_idx)
    #         return output
    #
    #     for i in range(self.n_layer):
    #         layer = model.model.layers[i]
    #         hook = layer.register_forward_hook(partial(_hook_fn, layer_idx=i))
    #         self.clfr_hooks.append(hook)
    #
    # def remove_clfr_hooks(self):
    #     for hook in self.clfr_hooks:
    #         hook.remove()

    def get_training_progress(self):
        if self.args.gradient_accumulation_steps != 1:
            return self.args.total_steps // self.args.per_device_train_batch_size * 2
        else:
            return self.args.total_steps * 2

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs_help = {
            "input_ids": inputs['input_ids'],
            "labels": inputs['labels'],
            "attention_mask": inputs['attention_mask']
        }

        inputs_algn = {
            "input_ids": inputs['refusal_input_ids'],
            "labels": inputs['refusal_labels'],
            "attention_mask": inputs['refusal_attention_mask']
        }

        input_over_refusal = {
            "input_ids": inputs['over_refusal_input_ids'],
            "labels": inputs['over_refusal_labels'],
            "attention_mask": inputs['over_refusal_attention_mask']
        }

        output_algn = 0.0
        if self.args.coeff_harmful > 0.0:
            if self.current_training_step == 0 or (self.args.update_refusal_features and self.current_training_step % (
                    self.coefficient_step * self.args.update_refusal_features_step) == 0):
                mask_index_device, directions_device, variance_last_dim, harmless_directions_device \
                    = self.train_dataset.get_refusal_feature(step=self.current_training_step)

                mask_direction_device = mask_index_device * directions_device
                if self.args.refusal_feature_type == "variable":
                    mask_harmless_direction_device = mask_index_device * harmless_directions_device
                    mask_variance_device = mask_index_device * variance_last_dim
                else:
                    mask_harmless_direction_device = None
                    mask_variance_device = None
                delta_device = torch.tensor(self.args.deltas, device=model.base_model.model.device)
                belta_device = torch.tensor(self.args.beta, device=model.base_model.model.device)
                ablation_fwd_pre_hooks, self.ablation_fwd_hooks = get_all_direction_ablation_hooks(model.base_model.model,
                                                                                              mask_index_device,
                                                                                              mask_direction_device,
                                                                                              delta_device,
                                                                                              mask_variance_device,
                                                                                              mask_harmless_direction_device,
                                                                                              pert_coef=belta_device,
                                                                                              selected_layer_ids=self.selected_layer_ids,
                                                                                              refusal_feature_type=self.args.refusal_feature_type
                                                                                              )

                with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=self.ablation_fwd_hooks):
                    output_algn = model(**inputs_algn)

            else:
                with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=self.ablation_fwd_hooks):
                    output_algn = model(**inputs_algn)

            output_help = model(**inputs_help)

            self.current_training_step += 1

            total_loss = self.args.coeff_harmful * output_algn.loss + \
                         self.args.coeff_help * output_help.loss

            if self.args.using_adaptive_loss and self.current_training_step % (self.args.gradient_accumulation_steps * 10) == 0:
                self.args.coeff_harmful = self.args.coeff_harmful * 0.5

            self.log(
                {
                    "help_loss": output_help.loss.item(),
                    "algn_loss": output_algn.loss.item(),
                }
            )


        else:
            raise ValueError("coeff_harmful must be greater than 0.0")

        return (total_loss,) if return_outputs else total_loss
