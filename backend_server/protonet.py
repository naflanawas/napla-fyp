"""
Prototypical Network for Few-Shot Learning

Implements prototype-based classification for personalized breath pattern recognition.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import DISTANCE_METRIC, CONFIDENCE_THRESHOLD, EMBEDDING_DIM


@dataclass
class PredictionResult:
    """Result of a prediction"""
    intent: str
    confidence: float
    distance: float
    all_distances: Dict[str, float]
    is_confident: bool
    
    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "distance": self.distance,
            "all_distances": self.all_distances,
            "is_confident": self.is_confident
        }


class ProtoNet:
    """
    Prototypical Network for few-shot breath classification
    
    Each user has a set of prototypes (mean embeddings) for their intents.
    Classification is done by finding the nearest prototype.
    """
    
    def __init__(self, 
                 distance_metric: str = DISTANCE_METRIC,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 embedding_dim: int = EMBEDDING_DIM):
        self.distance_metric = distance_metric
        self.confidence_threshold = confidence_threshold
        self.embedding_dim = embedding_dim
        
        # User prototypes: {user_id: {intent: prototype_tensor}}
        self.prototypes: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Sample storage for incremental updates: {user_id: {intent: [embeddings]}}
        self.samples: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    
    def compute_distance(self, 
                        embedding: torch.Tensor, 
                        prototype: torch.Tensor) -> float:
        """Compute distance between embedding and prototype"""
        if self.distance_metric == "euclidean":
            return float(torch.norm(embedding - prototype, p=2))
        elif self.distance_metric == "cosine":
            similarity = F.cosine_similarity(
                embedding.unsqueeze(0), 
                prototype.unsqueeze(0)
            )
            return float(1 - similarity)  # Convert to distance
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def create_prototype(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Create prototype as mean of embeddings
        
        Uses Dynamic-K: adapts to available samples
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot create prototype from empty embeddings")
        
        stacked = torch.stack(embeddings)
        prototype = stacked.mean(dim=0)
        
        # Normalize for cosine distance
        if self.distance_metric == "cosine":
            prototype = F.normalize(prototype, p=2, dim=0)
        
        return prototype
    
    def add_sample(self, 
                   user_id: str, 
                   intent: str, 
                   embedding: torch.Tensor) -> Tuple[bool, float]:
        """
        Add a calibration sample for a user's intent with outlier rejection
        
        Returns:
            (accepted, distance) - True if added, False if rejected (too far from prototype)
        """
        MAX_DISTANCE = 25.0 # Validation threshold

        # Initialize user storage if needed
        if user_id not in self.samples:
            self.samples[user_id] = {}
            self.prototypes[user_id] = {}
        
        if intent not in self.samples[user_id]:
            self.samples[user_id][intent] = []
            
        # Check against existing prototype if it exists
        distance = 0.0
        embedding_cpu = embedding.detach().cpu()
        
        if intent in self.prototypes[user_id] and len(self.samples[user_id][intent]) > 0:
            existing_proto = self.prototypes[user_id][intent]
            distance = self.compute_distance(embedding_cpu, existing_proto)
            
            if distance > MAX_DISTANCE:
                return False, distance
        
        # Add sample
        self.samples[user_id][intent].append(embedding_cpu)
        
        # Update prototype
        self.prototypes[user_id][intent] = self.create_prototype(
            self.samples[user_id][intent]
        )
        
        return True, distance
    
    def add_samples_batch(self,
                          user_id: str,
                          intent: str,
                          embeddings: torch.Tensor) -> int:
        """
        Add multiple calibration samples at once
        
        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)
        
        Returns:
            Total number of samples for this intent
        """
        for i in range(embeddings.shape[0]):
            self.add_sample(user_id, intent, embeddings[i])
        
        return len(self.samples[user_id][intent])
    
    def predict(self, 
                user_id: str, 
                embedding: torch.Tensor) -> PredictionResult:
        """
        Predict intent for a given embedding
        
        Uses nearest prototype classification with confidence margin.
        """
        if user_id not in self.prototypes or len(self.prototypes[user_id]) == 0:
            return PredictionResult(
                intent="unknown",
                confidence=0.0,
                distance=float('inf'),
                all_distances={},
                is_confident=False
            )
        
        user_prototypes = self.prototypes[user_id]
        
        # Compute distances to all prototypes
        distances = {
            intent: self.compute_distance(embedding, proto)
            for intent, proto in user_prototypes.items()
        }
        
        # Sort by distance
        sorted_intents = sorted(distances.items(), key=lambda x: x[1])
        
        best_intent, best_distance = sorted_intents[0]
        
        # Compute confidence using margin
        if len(sorted_intents) > 1:
            second_distance = sorted_intents[1][1]
            margin = second_distance - best_distance
            # Confidence: how much closer is the best match compared to second best
            # Normalized to (0, 1) range using sigmoid-like function
            confidence = 1 / (1 + np.exp(-margin * 2))
        else:
            # Only one intent - use distance-based confidence
            confidence = 1 / (1 + best_distance)
        
        is_confident = confidence >= self.confidence_threshold
        
        return PredictionResult(
            intent=best_intent,
            confidence=float(confidence),
            distance=float(best_distance),
            all_distances=distances,
            is_confident=is_confident
        )
    
    def predict_with_aggregation(self,
                                  user_id: str,
                                  embeddings: torch.Tensor,
                                  method: str = "voting") -> PredictionResult:
        """
        Predict intent from multiple window embeddings
        
        Args:
            embeddings: Tensor of shape (num_windows, embedding_dim)
            method: "voting" (majority vote) or "mean" (average embedding)
        
        Returns:
            Aggregated prediction result
        """
        if method == "mean":
            # Average all embeddings and predict once
            mean_embedding = embeddings.mean(dim=0)
            return self.predict(user_id, mean_embedding)
        
        elif method == "voting":
            # Predict for each window and take majority vote
            predictions = []
            for i in range(embeddings.shape[0]):
                result = self.predict(user_id, embeddings[i])
                predictions.append(result)
            
            # Count votes for each intent
            vote_counts: Dict[str, List[PredictionResult]] = {}
            for pred in predictions:
                if pred.intent not in vote_counts:
                    vote_counts[pred.intent] = []
                vote_counts[pred.intent].append(pred)
            
            # Find winner
            winner_intent = max(vote_counts.keys(), key=lambda x: len(vote_counts[x]))
            winner_predictions = vote_counts[winner_intent]
            
            # Average confidence of winning predictions
            avg_confidence = np.mean([p.confidence for p in winner_predictions])
            avg_distance = np.mean([p.distance for p in winner_predictions])
            
            # Aggregate all_distances
            all_intents = set()
            for pred in predictions:
                all_intents.update(pred.all_distances.keys())
            
            avg_distances = {}
            for intent in all_intents:
                intent_dists = [p.all_distances.get(intent, float('inf')) 
                               for p in predictions]
                avg_distances[intent] = np.mean(intent_dists)
            
            return PredictionResult(
                intent=winner_intent,
                confidence=float(avg_confidence),
                distance=float(avg_distance),
                all_distances=avg_distances,
                is_confident=avg_confidence >= self.confidence_threshold
            )
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_user_intents(self, user_id: str) -> List[str]:
        """Get list of intents for a user"""
        if user_id not in self.prototypes:
            return []
        return list(self.prototypes[user_id].keys())
    
    def get_sample_count(self, user_id: str, intent: str) -> int:
        """Get number of samples for a user's intent"""
        if user_id not in self.samples:
            return 0
        if intent not in self.samples[user_id]:
            return 0
        return len(self.samples[user_id][intent])
    
    def delete_intent(self, user_id: str, intent: str) -> bool:
        """Delete an intent for a user"""
        if user_id not in self.prototypes:
            return False
        
        if intent in self.prototypes[user_id]:
            del self.prototypes[user_id][intent]
        if user_id in self.samples and intent in self.samples[user_id]:
            del self.samples[user_id][intent]
        
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete all data for a user"""
        if user_id in self.prototypes:
            del self.prototypes[user_id]
        if user_id in self.samples:
            del self.samples[user_id]
        return True
    
    def get_state(self) -> dict:
        """Get serializable state"""
        return {
            "prototypes": {
                user_id: {
                    intent: proto.tolist()
                    for intent, proto in intents.items()
                }
                for user_id, intents in self.prototypes.items()
            },
            "samples": {
                user_id: {
                    intent: [emb.tolist() for emb in embs]
                    for intent, embs in intents.items()
                }
                for user_id, intents in self.samples.items()
            }
        }
    
    def load_state(self, state: dict) -> None:
        """Load state from dict"""
        self.prototypes = {
            user_id: {
                intent: torch.tensor(proto)
                for intent, proto in intents.items()
            }
            for user_id, intents in state.get("prototypes", {}).items()
        }
        self.samples = {
            user_id: {
                intent: [torch.tensor(emb) for emb in embs]
                for intent, embs in intents.items()
            }
            for user_id, intents in state.get("samples", {}).items()
        }
