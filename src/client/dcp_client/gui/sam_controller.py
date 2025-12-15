"""SAM Controller for assisted labelling in Napari.

This module handles all SAM (Segment Anything Model) related functionality
for the Napari window, including prompt handling, mask management, and
layer coordination.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
import uuid

import napari
import numpy as np

if TYPE_CHECKING:
    from dcp_client.app import Application

from dcp_client.gui.sam_worker import SAMInferenceWorker
from dcp_client.utils.sam_model_manager import SAMModelManager
from dcp_client.utils.utils import get_path_stem


class SAMController:
    """Controls SAM-assisted labelling in Napari viewer."""
    
    # Layer names
    BOX_PROMPTS = "box-prompts"
    POINT_PROMPTS = "point-prompts"
    SAM_PREVIEW = "sam-preview"
    
    def __init__(self, viewer: napari.Viewer, app: "Application"):
        self.viewer = viewer
        self.app = app
        
        self.sam_worker: Optional[SAMInferenceWorker] = None
        self.sam_model_manager: Optional[SAMModelManager] = None
        self._initialized = False
        
        self._next_label_id = 1
        self._mask_label_history: list[int] = []
        self._box_prompts_len = 0
        self._bg_point_mode = False
        self._box_events_connected = False
    
    def _get_layer(self, name: str) -> Optional[napari.layers.Layer]:
        """Get layer by name, returns None if not found."""
        return next((l for l in self.viewer.layers if l.name == name), None)
    
    def _get_image_layer(self) -> Optional[napari.layers.Image]:
        """Get the first Image layer in viewer."""
        return next((l for l in self.viewer.layers if isinstance(l, napari.layers.Image)), None)
    
    def _get_seg_layer(self) -> Optional[napari.layers.Labels]:
        """Get the segmentation layer (contains '_seg' in name)."""
        return next(
            (l for l in self.viewer.layers if isinstance(l, napari.layers.Labels) and "_seg" in l.name),
            None
        )
    
    def initialize(self) -> bool:
        """Lazy initialize SAM components. Returns True if successful."""
        if self._initialized:
            return True
        
        try:
            self.sam_model_manager = SAMModelManager()
            self.sam_worker = SAMInferenceWorker(self.sam_model_manager)
            
            self.sam_worker.mask_ready.connect(self._on_mask_ready)
            self.sam_worker.error_occurred.connect(lambda msg: None)
            self.sam_worker.embedding_computed.connect(lambda: None)
            
            self.sam_worker.start()
            self._initialized = True
            return True
        except Exception:
            return False
    
    def set_image(self) -> None:
        """Set the current image for SAM embedding computation."""
        if not self.sam_worker:
            return
        
        image_layer = self._get_image_layer()
        if not image_layer:
            return
        
        image_data = image_layer.data
        if image_data.ndim < 2 or image_data.ndim > 3:
            return
        
        self.sam_worker.set_image(image_data, str(uuid.uuid4()))
    
    def stop(self) -> None:
        """Stop the SAM worker thread - signal only, no wait."""
        self._disconnect_box_events()
        if self.sam_worker:
            try:
                self.sam_worker._should_stop = True
                self.sam_worker.condition.wakeAll()
            except Exception:
                pass
            self.sam_worker = None
    
    def enable(self, mode: int, bind_viewer_keys: Callable) -> None:
        """
        Enable assisted labelling with specified mode.
        
        Args:
            mode: 0 for boxes, 1 for points
            bind_viewer_keys: Callback to bind keys at viewer level
        """
        self._ensure_preview_layer()
        bind_viewer_keys()
        
        if mode == 0:
            self._setup_box_prompts()
        else:
            self._setup_point_prompts()
    
    def disable(self) -> None:
        """Disable assisted labelling, hide prompt layers."""
        for name in (self.BOX_PROMPTS, self.POINT_PROMPTS, self.SAM_PREVIEW):
            layer = self._get_layer(name)
            if layer:
                layer.visible = False
                if name != self.SAM_PREVIEW:
                    layer.mode = "pan_zoom"
    
    def _ensure_preview_layer(self) -> napari.layers.Labels:
        """Create or get the sam-preview layer."""
        layer = self._get_layer(self.SAM_PREVIEW)
        if layer:
            layer.visible = True
            return layer
        
        image_layer = self._get_image_layer()
        shape = image_layer.data.shape[:2] if image_layer else (512, 512)
        
        return self.viewer.add_labels(
            np.zeros(shape, dtype=np.int32),
            name=self.SAM_PREVIEW,
            opacity=0.5,
        )
    
    def _setup_box_prompts(self) -> None:
        """Set up box prompts layer and bindings."""
        layer = self._get_layer(self.BOX_PROMPTS)
        if not layer:
            layer = self.viewer.add_shapes(
                name=self.BOX_PROMPTS,
                edge_color="magenta",
                face_color="transparent",
                edge_width=4,
            )
        else:
            layer.visible = True
        
        try:
            layer.mode = "add_rectangle"
        except Exception:
            layer.mode = "add"
        
        point_layer = self._get_layer(self.POINT_PROMPTS)
        if point_layer:
            point_layer.visible = False
        
        self.viewer.layers.selection = {layer}
        self._box_prompts_len = len(layer.data)
        self._connect_box_events(layer)
        self._bind_layer_keys(layer, include_points=False)
    
    def _setup_point_prompts(self) -> None:
        """Set up point prompts layer and bindings."""
        layer = self._get_layer(self.POINT_PROMPTS)
        if not layer:
            layer = self.viewer.add_points(name=self.POINT_PROMPTS, size=10)
        else:
            layer.visible = True
        
        layer.mode = "add"
        
        box_layer = self._get_layer(self.BOX_PROMPTS)
        if box_layer:
            box_layer.visible = False
        
        self.viewer.layers.selection = {layer}
        self._bind_layer_keys(layer, include_points=True)
    
    def _connect_box_events(self, layer: napari.layers.Shapes) -> None:
        """Connect box data change events (always refreshes connection)."""
        self._disconnect_box_events()
        try:
            layer.events.data.connect(self._on_box_data_changed)
            self._box_events_connected = True
        except Exception:
            pass
    
    def _disconnect_box_events(self) -> None:
        """Disconnect box data change events."""
        layer = self._get_layer(self.BOX_PROMPTS)
        if layer:
            try:
                layer.events.data.disconnect(self._on_box_data_changed)
            except Exception:
                pass
        self._box_events_connected = False
    
    def _bind_layer_keys(self, layer, include_points: bool) -> None:
        """Bind keybindings to a prompt layer."""
        try:
            layer.bind_key('Enter', overwrite=True)(self._accept_layer_wrapper)
            layer.bind_key('Escape', overwrite=True)(self._reject_layer_wrapper)
            layer.bind_key('Control-z', overwrite=True)(self._undo_layer_wrapper)
            
            if include_points:
                layer.bind_key('d', overwrite=True)(self._on_points_confirmed)
                layer.bind_key('b', overwrite=True)(self._toggle_bg_mode)
        except Exception:
            pass
    
    def _accept_layer_wrapper(self, layer):
        self.accept_masks()
        return layer
    
    def _reject_layer_wrapper(self, layer):
        self.reject_last_mask()
        return layer
    
    def _undo_layer_wrapper(self, layer):
        self.undo_last_prompt()
        return layer
    
    def _toggle_bg_mode(self, layer):
        """Toggle between foreground (white) and background (red) point mode."""
        point_layer = self._get_layer(self.POINT_PROMPTS)
        if point_layer:
            self._bg_point_mode = not self._bg_point_mode
            point_layer.current_face_color = 'red' if self._bg_point_mode else 'white'
        return layer
    
    # === Event Handlers ===
    
    def _on_box_data_changed(self, event) -> None:
        """Handle new boxes added to the box-prompts layer."""
        layer = event.source
        if layer.name != self.BOX_PROMPTS:
            return
        
        new_len = len(layer.data)
        if new_len > self._box_prompts_len:
            for idx in range(self._box_prompts_len, new_len):
                box = np.asarray(layer.data[idx])
                if box.ndim == 2 and box.shape[1] >= 2:
                    box_2d = box[:, -2:]  # Take last 2 columns (y, x)
                    if self.sam_worker and self._initialized:
                        self.sam_worker.predict_box(box_2d, str(uuid.uuid4()))
        
        self._box_prompts_len = new_len
    
    def _on_points_confirmed(self, layer):
        """Called when user presses 'd' to confirm points."""
        if layer.name != self.POINT_PROMPTS:
            return layer
        
        points = np.asarray(layer.data)
        if len(points) == 0:
            return layer
        
        colors = np.asarray(layer.face_color)
        # Handle both uniform (1D) and per-point (2D) color arrays
        if colors.ndim == 1:
            # Uniform color for all points - broadcast to (N, 4)
            colors = np.tile(colors, (len(points), 1))
        is_white = np.all(np.isclose(colors[:, :3], [1, 1, 1]), axis=1)
        
        fg_points = points[is_white]
        bg_points = points[~is_white]
        
        def to_sam_coords(arr):
            if arr.size == 0:
                return arr.reshape(0, 2)
            arr_2d = arr[:, -2:] if arr.shape[1] > 2 else arr
            return np.flip(arr_2d, axis=1)
        
        if self.sam_worker and self._initialized:
            self.sam_worker.predict_points(
                to_sam_coords(fg_points),
                to_sam_coords(bg_points),
                str(uuid.uuid4())
            )
            layer.data = np.empty((0, 2))
        
        return layer
    
    def _on_mask_ready(self, mask: np.ndarray, confidence: float, token: str) -> None:
        """Handle SAM mask prediction result."""
        current_selection = set(self.viewer.layers.selection)
        
        preview_layer = self._get_layer(self.SAM_PREVIEW)
        if not preview_layer:
            image_layer = self._get_image_layer()
            shape = image_layer.data.shape[:2] if image_layer else mask.shape[:2]
            preview_layer = self.viewer.add_labels(
                np.zeros(shape, dtype=np.int32),
                name=self.SAM_PREVIEW,
                opacity=0.5,
            )
            if current_selection:
                self.viewer.layers.selection = current_selection
        
        existing = preview_layer.data.copy()
        if mask.shape[:2] != existing.shape[:2]:
            return
        
        existing[mask > 0] = self._next_label_id
        self._mask_label_history.append(self._next_label_id)
        self._next_label_id += 1
        
        preview_layer.data = existing
        preview_layer.visible = True
        preview_layer.refresh()
    
    # === Mask Actions ===
    
    def _get_dominant_label(self, region_values: np.ndarray, sam_region_size: int, threshold: float = 0.3) -> int | None:
        """Return existing label if overlap > threshold, else None."""
        positives = region_values[region_values > 0]
        if positives.size == 0:
            return None
        counts = np.bincount(positives)
        dominant_idx = counts.argmax()
        if counts[dominant_idx] / sam_region_size > threshold:
            return int(dominant_idx)
        return None
    
    def accept_masks(self) -> None:
        """Accept all preview masks and merge into segmentation layer."""
        preview_layer = self._get_layer(self.SAM_PREVIEW)
        if not preview_layer or np.max(preview_layer.data) == 0:
            return
        
        preview_data = preview_layer.data
        preview_labels = np.unique(preview_data)
        preview_labels = preview_labels[preview_labels > 0]
        
        if len(preview_labels) == 0:
            return
        
        seg_layer = self._get_seg_layer()
        if not seg_layer:
            seg_name = get_path_stem(self.app.cur_selected_img) + "_seg"
            h, w = preview_data.shape[:2]
            if self.app.num_classes > 1:
                seg_data_init = np.zeros((2, h, w), dtype=np.int32)
            else:
                seg_data_init = np.zeros((h, w), dtype=np.int32)
            seg_layer = self.viewer.add_labels(seg_data_init, name=seg_name)
        
        seg_data = seg_layer.data.copy()
        
        if seg_data.ndim > 2:
            active_channel = self.viewer.dims.current_step[0] if self.viewer.dims.current_step else 0
            target = seg_data[active_channel]
            
            preview_resized = self._resize_if_needed(preview_data, target.shape)
            next_label = int(np.max(target)) + 1
            
            for lbl in preview_labels:
                region = preview_resized == lbl
                region_size = region.sum()
                dominant = self._get_dominant_label(target[region], region_size)
                
                if dominant:
                    seg_data[active_channel][seg_data[active_channel] == dominant] = 0
                    assign_label = dominant
                else:
                    assign_label = next_label
                    next_label += 1
                
                seg_data[active_channel][region] = assign_label
                if active_channel == 0 and seg_data.shape[0] > 1:
                    seg_data[1][region] = -1
        else:
            preview_resized = self._resize_if_needed(preview_data, seg_data.shape)
            next_label = int(np.max(seg_data)) + 1
            
            for lbl in preview_labels:
                region = preview_resized == lbl
                region_size = region.sum()
                dominant = self._get_dominant_label(seg_data[region], region_size)
                
                if dominant:
                    seg_data[seg_data == dominant] = 0
                    assign_label = dominant
                else:
                    assign_label = next_label
                    next_label += 1
                
                seg_data[region] = assign_label
        
        seg_layer.data = seg_data
        seg_layer.refresh()
        
        self.viewer.layers.remove(preview_layer)
        self._reset_state()
        self._clear_prompts()
    
    def reject_last_mask(self) -> None:
        """Remove the last preview mask and its prompt."""
        self._remove_last_preview_mask()
        self._remove_last_prompt()
    
    def undo_last_prompt(self) -> None:
        """Undo the last prompt and its mask (alias for reject_last_mask)."""
        self._remove_last_preview_mask()
        self._remove_last_prompt()
    
    def _remove_last_preview_mask(self) -> None:
        """Remove the last mask from preview layer."""
        if not self._mask_label_history:
            return
        
        preview_layer = self._get_layer(self.SAM_PREVIEW)
        if preview_layer:
            last_label = self._mask_label_history.pop()
            data = preview_layer.data.copy()
            data[data == last_label] = 0
            preview_layer.data = data
            preview_layer.refresh()
    
    def _remove_last_prompt(self) -> None:
        """Remove the last box or point prompt."""
        box_layer = self._get_layer(self.BOX_PROMPTS)
        if box_layer and len(box_layer.data) > 0:
            box_layer.data = box_layer.data[:-1]
            self._box_prompts_len = len(box_layer.data)
        
        point_layer = self._get_layer(self.POINT_PROMPTS)
        if point_layer and len(point_layer.data) > 0:
            point_layer.data = point_layer.data[:-1]
    
    def _clear_prompts(self) -> None:
        """Clear all prompt layers."""
        box_layer = self._get_layer(self.BOX_PROMPTS)
        if box_layer and len(box_layer.data) > 0:
            box_layer.data = []
            self._box_prompts_len = 0
        
        point_layer = self._get_layer(self.POINT_PROMPTS)
        if point_layer and len(point_layer.data) > 0:
            point_layer.data = np.empty((0, 2))
    
    def _reset_state(self) -> None:
        """Reset mask tracking state."""
        self._mask_label_history = []
        self._next_label_id = 1
    
    def _resize_if_needed(self, data: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize data to match target shape if needed."""
        if data.shape == target_shape:
            return data
        
        from scipy.ndimage import zoom
        scale = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
        return zoom(data, scale, order=0).astype(data.dtype)

