import hashlib
import yaml
from pathlib import Path

MODEL_REGISTRY = "model_registry.yaml"
MODEL_DIR = Path("models")

def verify_model_versions():
    with open(MODEL_REGISTRY) as f:
        registry = yaml.safe_load(f)
    
    for model_name, spec in registry["models"].items():
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found in {MODEL_DIR}")
        
        # Verify model hash
        hasher = hashlib.sha256()
        with open(model_path / "model.bin", "rb") as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        
        if hasher.hexdigest() != spec["hash"]:
            raise ValueError(f"Model {model_name} hash mismatch")

def update_model_registry():
    # Implementation for automated registry updates
    pass

if __name__ == "__main__":
    verify_model_versions()
