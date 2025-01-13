from .llm_config import cfg
from sklearn.linear_model import LogisticRegression

import torch.nn as nn
import torch

class LayerClassifier:
    def __init__(self, llm_cfg: cfg, lr: float=0.01, max_iter: int=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = LogisticRegression(solver="saga", max_iter=max_iter)
        
        self.data = {
            "train": {
                "pos": None,
                "neg": None,
            },
            "test": {
                "pos": None,
                "neg": None,
            }
        }

    def train(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, n_epoch: int=100, batch_size: int=64) -> list[float]:
        X = torch.vstack([pos_tensor, neg_tensor]).to(self.device)
        y = torch.cat((torch.zeros(pos_tensor.size(0)), torch.ones(neg_tensor.size(0)))).to(self.device)
        
        self.data["train"]["pos"] = pos_tensor.cpu()
        self.data["train"]["neg"] = neg_tensor.cpu()

        self.linear.fit(X.cpu().numpy(), y.cpu().numpy())

        return []
    
    def predict(self, tensor: torch.tensor) -> torch.tensor:
        return torch.tensor(self.linear.predict(tensor.cpu().numpy()))
        
    def evaluate_testacc(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor) -> float:
        test_data = torch.vstack([pos_tensor, neg_tensor]).to(self.device)
        predictions = self.predict(test_data)
        true_labels = torch.cat((torch.zeros(pos_tensor.size(0)), torch.ones(neg_tensor.size(0))))

        correct_count = torch.sum((predictions > 0.5) == true_labels).item()

        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()

        return correct_count / len(true_labels)
    
    def get_weights_bias(self) -> tuple[torch.tensor, torch.tensor]:
        return torch.tensor(self.linear.coef_).to(self.device), torch.tensor(self.linear.intercept_).to(self.device)

