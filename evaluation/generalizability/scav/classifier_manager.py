from .embedding_manager import EmbeddingManager
from .layer_classifier import LayerClassifier
from tqdm import tqdm
import torch
import os

class ClassifierManager:
    def __init__(self, classifier_type: str):
        self.type = classifier_type
        self.classifiers = []
        self.testacc = []

    def _train_classifiers(
        self, 
        pos_embds: EmbeddingManager,
        neg_embds: EmbeddingManager,
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        print("Training classifiers...")

        self.llm_cfg = pos_embds.llm_cfg

        for i in tqdm(range(self.llm_cfg.n_layer)):
            layer_classifier = LayerClassifier(pos_embds.llm_cfg, lr)
            layer_classifier.train(
                pos_tensor=pos_embds.layers[i],
                neg_tensor=neg_embds.layers[i],
                n_epoch=n_epochs,
                batch_size=batch_size,
            )

            self.classifiers.append(layer_classifier)

    def _evaluate_testacc(self, pos_embds: EmbeddingManager, neg_embds: EmbeddingManager):
        for i in tqdm(range(len(self.classifiers))):
            self.testacc.append(
                self.classifiers[i].evaluate_testacc(
                    pos_tensor=pos_embds.layers[i],
                    neg_tensor=neg_embds.layers[i],
                )
            )
    
    def fit(
        self, 
        pos_embds_train: EmbeddingManager,
        neg_embds_train: EmbeddingManager,
        pos_embds_test: EmbeddingManager,
        neg_embds_test: EmbeddingManager,
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        self._train_classifiers(
            pos_embds_train,
            neg_embds_train,
            lr,
            n_epochs,
            batch_size,
        )

        self._evaluate_testacc(
            pos_embds_test,
            neg_embds_test,
        )

        return self
    
    def save(self, relative_path: str):
        file_name = f"{self.type}_{self.llm_cfg.model_nickname}.pth"
        path = os.path.join(relative_path, file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self, path)
    
    def cal_perturbation(
        self,
        embds_tensor: torch.tensor,
        layer: int,
        target_prob: float=0.001,
        use_masked_scav=False,
        map_current_direction_mask_dict={},
        use_ablation_coefficient=False,
        ablation_coefficient=1.0
    ):
        if use_masked_scav:
            w, b = self.classifiers[layer].get_weights_bias()
            layer_mask = map_current_direction_mask_dict[layer]
            embds_tensor = embds_tensor * layer_mask
            w = w * layer_mask

            logit_target = torch.log(torch.tensor(target_prob / (1 - target_prob)))
            w_norm = torch.norm(w)

            epsilon = (logit_target - b - torch.sum(embds_tensor * w, dim=1)) / w_norm
            expanded_epsilon = epsilon.unsqueeze(1).unsqueeze(2)  # 变为 batch_size*1*1

            perturbation = expanded_epsilon * w / w_norm
            if use_ablation_coefficient:
                perturbation = perturbation * ablation_coefficient

            return embds_tensor + perturbation.squeeze(1)

        else:
            w, b = self.classifiers[layer].get_weights_bias()
            logit_target = torch.log(torch.tensor(target_prob / (1 - target_prob)))
            w_norm = torch.norm(w)

            epsilon = (logit_target - b - torch.sum(embds_tensor * w, dim=1)) / w_norm
            expanded_epsilon = epsilon.unsqueeze(1).unsqueeze(2)  # 变为 batch_size*1*1

            perturbation = expanded_epsilon * w / w_norm
            if use_ablation_coefficient:
                perturbation = perturbation * ablation_coefficient
            return embds_tensor + perturbation.squeeze(1)
    
def load_classifier_manager(file_path: str):
    return torch.load(file_path, weights_only=False)