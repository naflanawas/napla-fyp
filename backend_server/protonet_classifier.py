import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from model import MSTCN
from config import EMBEDDING_DIM, MODEL_PATH

class ProtoNetClassifier:
    """
    Prototypical Network Classifier for Personalized Breath Recognition.
    Wraps MSTCN for embedding extraction and performs prototype-based inference.
    """
    
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = MSTCN(in_channels=3, base_channels=64, embedding_dim=EMBEDDING_DIM, num_classes=2).to(self.device)
        
        # Load the personalized weights
        if Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            # Handle potential wrapping and different formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded ProtoNet personalized weights from {weights_path}")
        else:
            print(f"Warning: Weights not found at {weights_path}, using default weights.")
            
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed_sample(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for a batch of windows.
        Args:
            windows: Tensor of shape (batch, 3, 64, 1024)
        Returns:
            Embeddings: (batch, embedding_dim)
        """
        windows = windows.to(self.device)
        emb = self.model(windows, return_embedding=True)
        return emb.cpu()

    def compute_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
        """
        Compute class prototypes as the mean of support embeddings.
        Args:
            embeddings: (n_support, embedding_dim)
            labels: (n_support,) with integer class IDs [0, n_classes-1]
        Returns:
            Prototypes: (n_classes, embedding_dim)
        """
        prototypes = []
        for i in range(n_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                p = embeddings[mask].mean(dim=0)
            else:
                # Fallback if no samples for a class - should be avoided by calibration
                p = torch.zeros(embeddings.shape[1])
            prototypes.append(p)
        return torch.stack(prototypes)

    def predict(self, query_emb: torch.Tensor, prototypes: torch.Tensor) -> Tuple[int, float]:
        """
        Predict class for a single query embedding using nearest prototype.
        Args:
            query_emb: (embedding_dim,)
            prototypes: (n_classes, embedding_dim)
        Returns:
            predicted_class: int
            confidence_margin: float (d2 - d1)
        """
        # Distance to all prototypes
        dists = torch.cdist(query_emb.unsqueeze(0), prototypes).squeeze(0)
        
        # Sort distances
        sorted_dists, sorted_idx = torch.sort(dists)
        
        d1 = sorted_dists[0].item()
        
        if len(sorted_dists) > 1:
            d2 = sorted_dists[1].item()
            confidence_margin = d2 - d1
        else:
            # Only one class defined
            confidence_margin = 1.0 / (1.0 + d1)
            
        pred = sorted_idx[0].item()
        return int(pred), float(confidence_margin)
