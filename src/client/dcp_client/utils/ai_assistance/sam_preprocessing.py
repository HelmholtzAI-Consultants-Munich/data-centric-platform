"""
Preprocessing module for SAM (Segment Anything Model) integration.

Handles image preprocessing and format conversion for SAM input.
"""

import numpy as np
import hashlib
from typing import Tuple


def preprocess_image(
    image: np.ndarray, target_dtype: str = "uint8"
) -> Tuple[np.ndarray, str, Tuple[int, int]]:
    """
    Preprocess image for SAM input.
    
    SAM automatically resizes images to 1024×1024 internally, but we need to:
    1. Validate and normalize the input
    2. Store original dimensions for coordinate/mask transformations
    3. Compute preprocessing hash for cache keys
    
    Args:
        image: Input image array (2D or 3D with channel dimension)
        target_dtype: Target dtype for output ("uint8" default)
    
    Returns:
        Tuple of (preprocessed_image, preproc_hash, original_shape)
        where original_shape is (height, width) of the original image
    
    Raises:
        ValueError: If image format is unsupported
    """
    # Store original dimensions and convert to RGB for SAM
    # SAM expects RGB images with shape (H, W, 3)
    if len(image.shape) == 2:
        # Grayscale (H, W) -> RGB (H, W, 3)
        original_shape = (image.shape[0], image.shape[1])
        processed = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            # Single channel (H, W, 1) -> RGB (H, W, 3)
            original_shape = (image.shape[0], image.shape[1])
            gray = image[:, :, 0]
            processed = np.stack([gray, gray, gray], axis=-1)
        elif image.shape[2] == 3:
            # Already RGB (H, W, 3)
            original_shape = (image.shape[0], image.shape[1])
            processed = image.copy()
        elif image.shape[2] == 4:
            # RGBA (H, W, 4) -> RGB (H, W, 3)
            original_shape = (image.shape[0], image.shape[1])
            processed = image[:, :, :3].copy()
        elif image.shape[0] == 1:
            # Single channel (1, H, W) -> RGB (H, W, 3)
            original_shape = (image.shape[1], image.shape[2])
            gray = image[0]
            processed = np.stack([gray, gray, gray], axis=-1)
        elif image.shape[0] == 3:
            # RGB in (C, H, W) format -> (H, W, 3)
            original_shape = (image.shape[1], image.shape[2])
            processed = np.transpose(image, (1, 2, 0))
        elif image.shape[0] == 4:
            # RGBA in (C, H, W) format -> RGB (H, W, 3)
            original_shape = (image.shape[1], image.shape[2])
            processed = np.transpose(image[:3], (1, 2, 0))
        else:
            raise ValueError(
                f"Cannot determine channel dimension for shape {image.shape}. "
                "Expected (H, W), (H, W, C), or (C, H, W) with C in [1, 3, 4]"
            )
    else:
        raise ValueError(
            f"Unsupported image shape: {image.shape}. "
            "Expected 2D (H, W) or 3D (H, W, C) or (C, H, W)"
        )
    
    # Validate dtype
    if processed.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        raise ValueError(
            f"Unsupported dtype: {processed.dtype}. "
            "Expected uint8, uint16, float32, or float64"
        )
    
    # Normalize to 0-255 range
    if processed.dtype == np.uint8:
        # Already in 0-255 range
        normalized = processed.astype(np.float32)
    elif processed.dtype == np.uint16:
        # Scale from 0-65535 to 0-255
        normalized = (processed.astype(np.float32) / 65535.0) * 255.0
    elif processed.dtype in [np.float32, np.float64]:
        # Check if already normalized (0-1) or in 0-255 range
        if processed.max() <= 1.0:
            # Scale from 0-1 to 0-255
            normalized = processed.astype(np.float32) * 255.0
        else:
            # Assume already in 0-255 range, just convert dtype
            normalized = processed.astype(np.float32)
    else:
        normalized = processed.astype(np.float32)
    
    # Clip to valid range
    normalized = np.clip(normalized, 0, 255)
    
    # Convert to target dtype
    if target_dtype == "uint8":
        processed_image = normalized.astype(np.uint8)
    else:
        processed_image = normalized
    
    # Compute preprocessing hash for cache key
    preproc_params = {
        "dtype": str(processed_image.dtype),
        "shape": processed_image.shape,
        "target_dtype": target_dtype,
    }
    preproc_str = str(sorted(preproc_params.items()))
    preproc_hash = hashlib.md5(preproc_str.encode()).hexdigest()[:8]
    
    return processed_image, preproc_hash, original_shape


def napari_rect_to_sam_box(napari_vertices: np.ndarray) -> np.ndarray:
    """
    Convert Napari rectangle vertices to SAM bounding box format.
    
    Napari rectangles are defined by 4 vertices (corners) in (row, col) = (y, x) format.
    SAM expects bounding boxes as [x_min, y_min, x_max, y_max].
    
    Args:
        napari_vertices: Array of shape (4, 2) with rectangle corners in (y, x) format
    
    Returns:
        SAM box format: [x_min, y_min, x_max, y_max] in original image coordinates
    """
    if napari_vertices.shape != (4, 2):
        raise ValueError(
            f"Expected napari_vertices shape (4, 2), got {napari_vertices.shape}"
        )
    
    # Napari stores as (row, col) = (y, x)
    y_coords = napari_vertices[:, 0]  # All y values (rows)
    x_coords = napari_vertices[:, 1]  # All x values (cols)
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    return np.array([x_min, y_min, x_max, y_max])
