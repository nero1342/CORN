import io
from pathlib import Path
from typing import Any, Callable, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, transform: Optional[Callable] = None, **kwargs) -> None:
        self.transform = transform
    
    def _read_image(self, image: Any, read_mode = "cv2") -> np.ndarray:
        """Read image from source.

        Args:
            image (Any): Image source. Could be str, Path or bytes.

        Returns:
            np.ndarray: Loaded image.
        """

        if read_mode == "pillow":
            # if not isinstance(image, (str, Path)):
            #     image = io.BytesIO(image)
            # image = np.asarray(Image.open(image).convert("RGB"))
            image = np.asarray(image)
        elif read_mode == "cv2":
            if not isinstance(image, (str, Path)):
                image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError("use pillow or cv2")
        return image

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError()
    