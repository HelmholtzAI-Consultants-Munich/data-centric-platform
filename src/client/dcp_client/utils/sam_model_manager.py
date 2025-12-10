"""
SAM model manager for downloading, loading, and hardware detection.

Handles model selection based on available hardware (MPS, CUDA, CPU)
and manages checkpoint downloads.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Optional
import urllib.request
from urllib.error import URLError


class SAMModelManager:
    """
    Manages SAM model download, loading, and hardware detection.
    """
    
    # Model checkpoint URLs (only models we actually use)
    CHECKPOINTS = {
        "vit_b": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        },
        "mobilesam": {
            "url": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        },
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory to cache model checkpoints (default: <package>/models/)
        """
        if cache_dir is None:
            # Store weights within the package directory
            package_dir = Path(__file__).parent.parent
            cache_dir = package_dir / "models"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_hardware(self) -> Dict[str, bool]:
        """
        Detect available hardware (MPS, CUDA, CPU).
        
        Returns:
            Dictionary with hardware availability flags
        """
        return {
            "mps": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
            "cuda": torch.cuda.is_available(),
            "cpu": True,  # Always available as fallback
        }
    
    def select_model(self, model_type: str = "auto") -> str:
        """
        Select best model for current hardware.
        
        Args:
            model_type: Model type ("auto", "vit_b", "mobilesam")
        
        Returns:
            Selected model type string
        """
        if model_type != "auto":
            if model_type not in self.CHECKPOINTS:
                raise ValueError(
                    f"Invalid model_type: {model_type}. "
                    f"Expected 'auto' or one of {list(self.CHECKPOINTS.keys())}"
                )
            return model_type
        
        # Auto-select based on hardware
        hardware = self.detect_hardware()
        
        if hardware["cuda"] or hardware["mps"]:
            return "vit_b"  # GPU available - use SAM-B for quality
        else:
            return "mobilesam"  # CPU only - use MobileSAM for speed
    
    def _get_checkpoint_filename(self, model_type: str) -> str:
        """Get checkpoint filename for a given model type."""
        if model_type == "mobilesam":
            return "sam_mobilesam.pt"
        return f"sam_{model_type}.pth"
    
    def get_checkpoint_path(self, model_type: str) -> Path:
        """Get checkpoint file path (download if needed)."""
        checkpoint_name = self._get_checkpoint_filename(model_type)
        checkpoint_path = self.cache_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            # Download checkpoint
            checkpoint_path = self.download_checkpoint(model_type)
        
        return checkpoint_path
    
    def download_checkpoint(self, model_type: str) -> Path:
        """
        Download model checkpoint.
        
        Args:
            model_type: Model type ("vit_b", "mobilesam")
        
        Returns:
            Path to downloaded checkpoint file
        
        Raises:
            ValueError: If model_type is invalid
            URLError: If download fails
        """
        if model_type not in self.CHECKPOINTS:
            raise ValueError(f"Invalid model_type: {model_type}. Expected one of {list(self.CHECKPOINTS.keys())}")
        
        url = self.CHECKPOINTS[model_type]["url"]
        checkpoint_name = self._get_checkpoint_filename(model_type)
        checkpoint_path = self.cache_dir / checkpoint_name
        
        print(f"Downloading {model_type} checkpoint from {url}...")
        try:
            urllib.request.urlretrieve(url, checkpoint_path)
        except URLError as e:
            raise URLError(f"Failed to download checkpoint: {e}")
        
        print(f"Checkpoint downloaded to {checkpoint_path}")
        return checkpoint_path
    
    def get_device(self) -> str:
        """
        Get best device for current hardware.
        
        Returns:
            Device string ("cuda", "mps", or "cpu")
        """
        hardware = self.detect_hardware()
        
        if hardware["cuda"]:
            return "cuda"
        elif hardware["mps"]:
            return "mps"
        else:
            return "cpu"
