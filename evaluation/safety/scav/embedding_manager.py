import torch
import os
from .llm_config import *

class EmbeddingManager:
    def __init__(self, llm_cfg: cfg=None, message: str=None):
        self.llm_cfg = llm_cfg
        self.message = message
        self.layers = []

    def save(self, relative_path: str):
        file_name = f"{self.llm_cfg.model_nickname}_{self.message}.pth"
        torch.save(self, os.path.join(relative_path, file_name))

def load_embds_manager(file_path: str) -> EmbeddingManager:
    return torch.load(file_path, weights_only=False)

def merge_tensor(tensors: list) -> torch.tensor:
    return torch.cat(tensors, dim=0)