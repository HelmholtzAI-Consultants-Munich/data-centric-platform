from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional

from skimage.io import imread, imsave
if TYPE_CHECKING:
    from numpy.typing import NDArray

from .mlclient import MLClient


@dataclass  # Builtin library, helps reduce boilerplate code, see https://www.youtube.com/watch?v=vBH6GRJ1REM for tutorial
class Application:
    mlclient: MLClient
    server_ip: str = '0.0.0.0'
    server_port: int = 7010
    img_filename: str = ''
    eval_data_path: str = ''
    train_data_path: str = ''
    inprogr_data_path: str = ''
        

    def load_image_and_seg(self) -> Tuple[NDArray, Optional[NDArray]]:
        self.potential_seg_name = Path(self.img_filename).stem + '_seg.tiff' #+Path(self.img_filename).suffix
        img_path = os.path.join(self.eval_data_path, self.img_filename)
        if os.path.exists(img_path):
            seq_path = os.path.join(self.eval_data_path, self.potential_seg_name)
        else: 
            img_path = os.path.join(self.train_data_path, self.img_filename)
            seq_path = os.path.join(self.train_data_path, self.potential_seg_name)

        img = imread(img_path)
        seg = imread(seq_path) if os.path.exists(seq_path) else None
        return img, seg
    
    def set_seg_name(self, name: str) -> None:
        self.seg_name = name


    def save_seg(self, seg: NDArray) -> None:
        os.replace(os.path.join(self.eval_data_path, self.img_filename), os.path.join(self.train_data_path, self.img_filename))
        seg_name = Path(self.img_filename).stem+ '_seg.tiff' #+Path(self.img_filename).suffix
        imsave(os.path.join(self.train_data_path, seg_name), seg)
        if os.path.exists(os.path.join(self.eval_data_path, seg_name)): 
            os.remove(os.path.join(self.eval_data_path, seg_name))
        
    def save2(self, seg):
        os.replace(os.path.join(self.eval_data_path, self.img_filename), os.path.join(self.inprogr_data_path, self.img_filename))
        seg_name = Path(self.img_filename).stem + '_' + self.seg_name + '.tiff' #+Path(self.img_filename).suffix
        imsave(os.path.join(self.inprogr_data_path, seg_name), seg)
        if os.path.exists(os.path.join(self.eval_data_path, self.potential_seg_name)): 
            os.replace(os.path.join(self.eval_data_path, self.potential_seg_name), os.path.join(self.inprogr_data_path, self.potential_seg_name))

    def run_train(self) -> str:
        if not self.mlclient.is_connected:
            self.mlclient.connect(ip=self.server_ip, port=self.server_port)
        return self.mlclient.run_train(path=self.train_data_path)
    
    def run_inference(self) -> str:
        if not self.mlclient.is_connected:
            self.mlclient.connect(ip=self.server_ip, port=self.server_port)
        return self.mlclient.run_inference(path=self.eval_data_path)
        