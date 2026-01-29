"""
MSTCN Model Architecture for MURMUR

Multi-Scale Temporal Convolutional Network for breath pattern embedding extraction.
Architecture based on provided specification with 4 branches of dilated convolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from pathlib import Path

from config import MODEL_PATH, EMBEDDING_DIM


class TCNBlock(nn.Module):
    """Temporal Convolutional Block with residual connection"""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation  # Same padding for kernel_size=3
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), 
                      padding=(0, padding), dilation=(1, dilation)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), 
                      padding=(0, padding), dilation=(1, dilation)),
            nn.ReLU(inplace=True),
        )
        
        # Residual connection (identity if channels match)
        self.residual = nn.Identity() if in_channels == out_channels else \
                        nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class MSTCN(nn.Module):
    """
    Multi-Scale Temporal Convolutional Network
    
    Architecture:
    - Stem: 2-layer conv for initial feature extraction
    - Branches: 4 parallel TCN branches with different dilations (1, 2, 4, 8)
    - Fuse: 1x1 conv to merge branch outputs
    - Pool: Global average pooling
    - Embed: Linear layer for embedding
    - Classifier: Linear layer for classification (2 classes)
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_branches: int = 4,
                 blocks_per_branch: int = 2,
                 embedding_dim: int = 64,
                 num_classes: int = 2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        
        # Branches: Multi-scale temporal processing with different dilations
        self.branches = nn.ModuleList()
        dilations = [1, 2, 4, 8]  # Multi-scale dilations
        
        for dilation in dilations[:num_branches]:
            branch = nn.Sequential(*[
                TCNBlock(base_channels, base_channels, dilation=dilation)
                for _ in range(blocks_per_branch)
            ])
            self.branches.append(branch)
        
        # Fuse: Combine all branches
        fuse_channels = base_channels * num_branches
        self.fuse = nn.Conv2d(fuse_channels, base_channels, kernel_size=(1, 1))
        
        # Pool: Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embed: Embedding layer
        self.embed = nn.Linear(base_channels, embedding_dim)
        
        # Classifier: Final classification
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            return_embedding: If True, return embedding instead of classification
        
        Returns:
            Embedding (batch, embedding_dim) or logits (batch, num_classes)
        """
        # Stem
        x = self.stem(x)
        
        # Parallel branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate and fuse
        x = torch.cat(branch_outputs, dim=1)
        x = self.fuse(x)
        
        # Global pooling
        x = self.pool(x)
        x = x.flatten(1)
        
        # Embedding
        embedding = self.embed(x)
        
        if return_embedding:
            return embedding
        
        # Classification
        logits = self.classifier(embedding)
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from input"""
        return self.forward(x, return_embedding=True)


class MSTCNEmbedder:
    """Wrapper for MSTCN model inference"""
    
    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = MSTCN(
            in_channels=3,
            base_channels=64,
            num_branches=4,
            blocks_per_branch=2,
            embedding_dim=EMBEDDING_DIM,
            num_classes=2
        ).to(self.device)
        
        # Load weights if available
        model_path = model_path or MODEL_PATH
        if model_path.exists():
            self._load_weights(model_path)
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Model weights not found at {model_path}")
        
        self.model.eval()
    
    def _load_weights(self, path: Path):
        """Load model weights with error handling"""
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    
    @torch.no_grad()
    def extract_embedding(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from spectrogram
        
        Args:
            spectrogram: Tensor of shape (batch, channels, freq, time) or (channels, freq, time)
        
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)
        
        spectrogram = spectrogram.to(self.device)
        embedding = self.model.get_embedding(spectrogram)
        return embedding
    
    @torch.no_grad()
    def extract_embeddings_batch(self, spectrograms: List[torch.Tensor]) -> torch.Tensor:
        """Extract embeddings for a batch of spectrograms"""
        batch = torch.stack(spectrograms).to(self.device)
        return self.model.get_embedding(batch)
