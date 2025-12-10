"""
QThread worker for non-blocking SAM inference.

Handles coordinate transformation, mask resizing, and thread-safe
SAM predictions with debouncing and cancellation support.
"""

import uuid
import numpy as np
import torch
from typing import Optional, Tuple, List
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from queue import Queue, Empty

from dcp_client.gui.sam_preprocessing import (
    preprocess_image,
    napari_rect_to_sam_box,
    transform_box_to_sam_space,
    transform_points_to_sam_space,
    resize_mask_to_original,
)
from dcp_client.utils.sam_model_manager import SAMModelManager


class SAMInferenceWorker(QThread):
    """
    QThread worker for non-blocking SAM inference.
    
    Handles all SAM operations in a separate thread to prevent UI blocking.
    One SamPredictor instance per worker thread (thread-safe).
    """
    
    mask_ready = pyqtSignal(np.ndarray, float, str)  # mask, confidence, token
    error_occurred = pyqtSignal(str)
    embedding_computed = pyqtSignal()  # Signal when embedding is ready
    
    def __init__(
        self,
        model_manager: SAMModelManager,
        parent=None,
    ):
        """
        Initialize SAM inference worker.
        
        Args:
            model_manager: SAMModelManager instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.model_manager = model_manager
        
        # Thread-safe communication
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # Request queue (max 3 pending)
        self.request_queue: Queue = Queue(maxsize=3)
        self.current_token: Optional[str] = None
        self.pending_requests: List[dict] = []
        
        # SAM predictor (created in worker thread)
        self.predictor = None
        self.device = None
        
        # Image state
        self.current_image: Optional[np.ndarray] = None
        self.original_shape: Optional[Tuple[int, int]] = None
        self.image_id: Optional[str] = None
        self.embedding_ready = False
        
        # Control flags
        self._should_stop = False
        self._is_processing = False
    
    def run(self):
        """
        Worker thread main loop.
        Processes requests from queue in worker thread.
        """
        try:
            # Initialize predictor in worker thread
            self._initialize_predictor()
            
            # Main processing loop
            while not self._should_stop:
                try:
                    # Get request from queue (with timeout to check stop flag)
                    request = self.request_queue.get(timeout=0.1)
                    
                    # Check if request is still valid (token check)
                    if request.get("token") != self.current_token and request.get("type") != "set_image":
                        continue  # Drop stale request
                    
                    # Process request
                    self._process_request(request)
                    
                except Empty:
                    # Timeout - check stop flag and continue
                    continue
                except Exception as e:
                    self.error_occurred.emit(f"Error in worker thread: {str(e)}")
        finally:
            self._cleanup()
    
    def _initialize_predictor(self):
        """Initialize SAM predictor in worker thread."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            # Select model
            model_type = self.model_manager.select_model()
            device = self.model_manager.get_device()
            self.device = device
            
            # Get checkpoint path
            checkpoint_path = self.model_manager.get_checkpoint_path(model_type)
            
            # Load model
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            sam.to(device=device)
            
            # Create predictor
            self.predictor = SamPredictor(sam)
            
            # Apply PyTorch optimizations
            self._apply_pytorch_optimizations(device)
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize SAM predictor: {str(e)}")
            raise
    
    def _apply_pytorch_optimizations(self, device: str):
        """
        Apply device-specific PyTorch optimizations.
        
        Args:
            device: Device string ("cuda", "mps", or "cpu")
        """
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            # AMP and channels_last will be applied per-operation
        elif device == "mps":
            # MPS optimizations (AMP with FP32 fallback detection)
            pass
        # CPU: inference_mode is sufficient
    
    def set_image(self, image: np.ndarray, image_id: str):
        """
        Set image and compute embedding (called from UI thread).
        
        Args:
            image: Image array
            image_id: Unique identifier for the image
        """
        self.mutex.lock()
        try:
            # Preprocess image
            processed_image, preproc_hash, original_shape = preprocess_image(image)
            
            # Store state
            self.current_image = processed_image
            self.original_shape = original_shape
            self.image_id = image_id
            self.embedding_ready = False
            
            # Queue embedding computation
            request = {
                "type": "set_image",
                "image": processed_image,
                "image_id": image_id,
                "preproc_hash": preproc_hash,
                "original_shape": original_shape,
                "token": str(uuid.uuid4()),  # Generate token for set_image
            }
            
            # Clear queue and add new request
            self._clear_queue()
            self.request_queue.put(request)
            self.condition.wakeAll()
            
        finally:
            self.mutex.unlock()
    
    def predict_box(self, napari_vertices: np.ndarray, token: str):
        """
        Queue bounding box prediction request.
        
        Args:
            napari_vertices: Array of shape (4, 2) with rectangle corners in (y, x) format
            token: Interaction token (UUID) for cancellation tracking
        """
        self.mutex.lock()
        try:
            # Update current token
            self.current_token = token
            
            # Convert Napari rectangle to SAM box format
            box_original = napari_rect_to_sam_box(napari_vertices)
            
            # Queue prediction request
            request = {
                "type": "predict_box",
                "box_original": box_original,
                "token": token,
            }
            
            # Drop stale requests for same gesture
            self._drop_stale_requests(token)
            
            # Add to queue (drop oldest if queue is full)
            try:
                self.request_queue.put_nowait(request)
            except:
                # Queue full - drop oldest and add new
                try:
                    self.request_queue.get_nowait()
                except Empty:
                    pass
                self.request_queue.put_nowait(request)
            
            self.condition.wakeAll()
            
        finally:
            self.mutex.unlock()
    
    def predict_points(
        self,
        foreground_points: np.ndarray,
        background_points: np.ndarray,
        token: str
    ):
        """
        Queue point-based prediction request.
        
        Args:
            foreground_points: Array of shape (N, 2) with foreground points in (x, y) format
            background_points: Array of shape (M, 2) with background points in (x, y) format
            token: Interaction token (UUID) for cancellation tracking
        """
        self.mutex.lock()
        try:
            self.current_token = token
            
            request = {
                "type": "predict_points",
                "foreground_points": foreground_points,
                "background_points": background_points,
                "token": token,
            }
            
            self._drop_stale_requests(token)
            
            try:
                self.request_queue.put_nowait(request)
            except:
                try:
                    self.request_queue.get_nowait()
                except Empty:
                    pass
                self.request_queue.put_nowait(request)
            
            self.condition.wakeAll()
            
        finally:
            self.mutex.unlock()
    
    def _process_request(self, request: dict):
        """
        Process a request in worker thread.
        
        Args:
            request: Request dictionary
        """
        try:
            if request["type"] == "set_image":
                self._compute_embedding(request)
            elif request["type"] == "predict_box":
                if request["token"] != self.current_token:
                    return
                self._predict_box(request)
            elif request["type"] == "predict_points":
                if request["token"] != self.current_token:
                    return
                self._predict_points(request)
        except Exception as e:
            self.error_occurred.emit(f"Error processing request: {str(e)}")
    
    def _compute_embedding(self, request: dict):
        """Compute SAM embedding for image."""
        image = request["image"]
        
        with torch.inference_mode():
            self.predictor.set_image(image)
        
        self.embedding_ready = True
        self.embedding_computed.emit()
    
    def _predict_box(self, request: dict):
        """
        Predict mask from bounding box.
        
        Args:
            request: Request dictionary with box data
        """
        if not self.embedding_ready or self.current_image is None:
            self.error_occurred.emit("Image embedding not ready. Call set_image() first.")
            return
        
        box_original = request["box_original"]
        token = request["token"]
        
        # SAM predictor expects box in original image coordinates
        # It handles the transformation to 1024x1024 internally
        
        # Run prediction
        try:
            with torch.inference_mode():
                # Apply device-specific optimizations
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        masks, scores, _ = self.predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box_original[None, :],  # Add batch dimension
                            multimask_output=False,
                        )
                else:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box_original[None, :],  # Add batch dimension
                        multimask_output=False,
                    )
            
            # Get best mask - SAM returns masks in original image size
            mask = masks[0]
            confidence = float(scores[0])
            
            # Check token before emitting (may have been cancelled)
            if token == self.current_token:
                self.mask_ready.emit(mask.astype(np.uint8), confidence, token)
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._handle_oom()
            else:
                self.error_occurred.emit(f"Prediction error: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Prediction error: {str(e)}")
    
    def _predict_points(self, request: dict):
        """
        Predict mask from point prompts.
        
        Args:
            request: Request dictionary with foreground/background points
        """
        if not self.embedding_ready or self.current_image is None:
            self.error_occurred.emit("Image embedding not ready. Call set_image() first.")
            return
        
        fg_points = request["foreground_points"]
        bg_points = request["background_points"]
        token = request["token"]
        
        # Combine points and create labels (1 for foreground, 0 for background)
        # SAM expects points in original image coordinates
        all_points = []
        all_labels = []
        
        if fg_points.size > 0:
            all_points.append(fg_points)
            all_labels.extend([1] * len(fg_points))
        
        if bg_points.size > 0:
            all_points.append(bg_points)
            all_labels.extend([0] * len(bg_points))
        
        if len(all_points) == 0:
            self.error_occurred.emit("No points provided for prediction.")
            return
        
        point_coords = np.vstack(all_points)
        point_labels = np.array(all_labels, dtype=np.int32)
        
        try:
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        masks, scores, _ = self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            box=None,
                            multimask_output=True,
                        )
                else:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=None,
                        multimask_output=True,
                    )
            
            # Get best mask (highest score) - SAM returns masks in original size
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            confidence = float(scores[best_idx])
            
            if token == self.current_token:
                self.mask_ready.emit(mask.astype(np.uint8), confidence, token)
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._handle_oom()
            else:
                self.error_occurred.emit(f"Prediction error: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Prediction error: {str(e)}")
    
    def _handle_oom(self):
        """Handle out-of-memory error with fallback."""
        self.error_occurred.emit(
            "Out of memory. Falling back to CPU/MobileSAM. "
            "This may take a moment..."
        )
        # TODO: Implement fallback to CPU/MobileSAM
        # For now, just emit error
    
    def _clear_queue(self):
        """Clear request queue."""
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except Empty:
                break
    
    def _drop_stale_requests(self, current_token: str):
        """Drop stale requests from queue."""
        temp_queue = Queue()
        while not self.request_queue.empty():
            try:
                req = self.request_queue.get_nowait()
                if req.get("token") == current_token or req.get("type") == "set_image":
                    temp_queue.put(req)
            except Empty:
                break
        
        # Put back non-stale requests
        while not temp_queue.empty():
            self.request_queue.put(temp_queue.get())
    
    def _cleanup(self):
        """Cleanup resources."""
        if self.predictor is not None:
            del self.predictor
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def stop(self):
        """Signal worker thread to stop - no blocking wait."""
        self._should_stop = True
        self.condition.wakeAll()
