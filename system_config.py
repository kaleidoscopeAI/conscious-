# Configuration for Quantum Conscious Cube System

# System parameters
SYSTEM = {
    "dimension": 3,           # Cube dimension
    "resolution": 32,         # Grid resolution
    "background_threads": 1,  # Number of background processing threads
    "processing_interval": 0.1,  # Time between updates (seconds)
}

# Node parameters
NODE = {
    "default_energy": 0.8,    # Initial node energy
    "default_stability": 0.7, # Initial node stability
    "feature_dim": 64,        # Dimension of feature vectors
    "memory_threshold": 5.0,  # Memory threshold before overflow
    "energy_decay": 0.999,    # Energy decay factor per update
    "replication_threshold": 0.7,  # Energy threshold for replication
}

# Visualization parameters
VISUALIZATION = {
    "enable_3d": True,       # Enable 3D visualization
    "show_tension": True,    # Visualize tension field
    "show_connections": True, # Show node connections
    "auto_refresh": 5,       # Auto-refresh interval (seconds)
}

# Advanced parameters
ADVANCED = {
    "quantum_qubits": 8,      # Qubits per standard node
    "supernode_qubits": 12,   # Qubits per SuperNode
    "memory_vector_size": 256, # Memory vector dimensions
    "enable_gpu": True,       # Use GPU acceleration if available
}
