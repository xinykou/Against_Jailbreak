from .classifier_manager import ClassifierManager
import torch

class Perturbation:
    def __init__(self, classifier_manager: ClassifierManager, target_probability: float=0.001, accuracy_threshold: float=0.9, layer_mask: list[int]=None):
        self.classifier_manager = classifier_manager
        self.target_probability = target_probability
        self.accuracy_threshold = accuracy_threshold
        self.layer_mask = layer_mask

    def get_perturbation(self, output_hook: torch.Tensor, layer: int) -> torch.Tensor:
        if self.layer_mask is None or layer not in self.layer_mask:
            if self.classifier_manager.testacc[layer] > self.accuracy_threshold:
                perturbed_embds = self.classifier_manager.cal_perturbation(
                    embds_tensor=output_hook[0][:, -1, :],
                    layer=layer,
                    target_prob=self.target_probability,
                )
                output_hook[0][:, -1, :] = perturbed_embds
        return output_hook