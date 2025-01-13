from .classifier_manager import ClassifierManager
import torch

class Perturbation:
    def __init__(self, classifier_manager: ClassifierManager, target_probability: float=0.001, accuracy_threshold: float=0.9,
                 layer_mask: list[int]=None,
                 use_masked_scav=False,
                 map_current_direction_mask_dict={},
                 use_ablation_coefficient=False,
                 ablation_coefficient=1.0
                 ):
        self.classifier_manager = classifier_manager
        self.target_probability = target_probability
        self.accuracy_threshold = accuracy_threshold
        self.layer_mask = layer_mask

        # scav settings
        self.use_masked_scav = use_masked_scav
        self.map_current_direction_mask_dict = map_current_direction_mask_dict
        self.use_ablation_coefficient = use_ablation_coefficient
        self.ablation_coefficient = ablation_coefficient

    def get_perturbation(self, output_hook: torch.Tensor, layer: int) -> torch.Tensor:
        if self.layer_mask is None or layer not in self.layer_mask:
            if self.classifier_manager.testacc[layer] > self.accuracy_threshold:
                perturbed_embds = self.classifier_manager.cal_perturbation(
                    embds_tensor=output_hook[0][:, -1, :],
                    layer=layer,
                    target_prob=self.target_probability,
                    use_masked_scav=self.use_masked_scav,
                    map_current_direction_mask_dict=self.map_current_direction_mask_dict,
                    use_ablation_coefficient=self.use_ablation_coefficient,
                    ablation_coefficient=self.ablation_coefficient
                )
                output_hook[0][:, -1, :] = perturbed_embds
        return output_hook