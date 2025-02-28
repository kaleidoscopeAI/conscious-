import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import numpy as np
from gudhi import SimplexTree, RipsComplex
from ripser import ripser
from persim import plot_diagrams
from scipy.spatial.distance import pdist, squareform
import dionysus as d
import networkx as nx
from dataclasses import dataclass

@dataclass
class PersistenceFeatures:
    diagrams: List[np.ndarray]
    bottleneck_distances: np.ndarray
    landscape_features: torch.Tensor
    connected_components: List[List[int]]

class TopologicalLayer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 max_dimension: int = 3,
                 n_landscape_points: int = 100):
        super().__init__()
        self.input_dim = input_dim
        self.max_dimension = max_dimension
        self.n_landscape_points = n_landscape_points
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        self.persistence_encoder = PersistenceEncoder(
            input_dim=input_dim,
            n_landscape_points=n_landscape_points
        )
        
    def compute_persistence(self, x: torch.Tensor) -> PersistenceFeatures:
        # Convert to distance matrix
        points = x.detach().cpu().numpy()
        distances = squareform(pdist(points))
        
        # Compute persistence diagrams
        diagrams = ripser(
            distances,
            maxdim=self.max_dimension,
            distance_matrix=True
        )['dgms']
        
        # Compute bottleneck distances between consecutive diagrams
        bottleneck_distances = np.zeros((len(diagrams)-1,))
        for i in range(len(diagrams)-1):
            bottleneck_distances[i] = d.bottleneck_distance(
                d.Diagram(diagrams[i]),
                d.Diagram(diagrams[i+1])
            )
            
        # Compute persistence landscapes
        landscapes = self._compute_landscapes(diagrams)
        
        # Extract connected components
        components = self._extract_components(distances)
        
        return PersistenceFeatures(
            diagrams=diagrams,
            bottleneck_distances=bottleneck_distances,
            landscape_features=landscapes,
            connected_components=components
        )

    def _compute_landscapes(self, diagrams: List[np.ndarray]) -> torch.Tensor:
        # Compute persistence landscapes
        landscapes = []
        for diagram in diagrams:
            landscape = d.Landscape(diagram)
            landscapes.append(landscape)
        # Convert to torch tensor
        return torch.tensor(landscapes)

    def _extract_components(self, distances: np.ndarray) -> List[List[int]]:
        # Extract connected components using NetworkX
        G = nx.Graph()
        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                if distances[i, j] > 0:
                    G.add_edge(i, j, weight=distances[i, j])
        components = [list(c) for c in nx.connected_components(G)]
        return components