import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import logging
import multiprocessing
import sys

# Third-party imports
import numpy as np
import cv2
import rawpy

from ..config import logger, log_critical


class EnhancedRawConverter:
    """Enhanced RAW image converter with multiprocessing support."""
    def __init__(self):
        self.max_workers = min(multiprocessing.cpu_count(), 16)

    def convert_raw_image(self, input_path: str, output_path: str, max_dims=(3000, 2000)) -> bool:
        """Convert and resize a RAW image while preserving aspect ratio and compress to PNG."""
        try:
            with rawpy.imread(input_path) as raw:
                width = raw.sizes.width
                height = raw.sizes.height

                rgb = raw.postprocess(
                    output_color=rawpy.ColorSpace.sRGB,
                    gamma=(2.2, 4.5),
                    no_auto_bright=False,
                    use_auto_wb=True,
                    use_camera_wb=True,
                    bright=1.0,
                    user_flip=-1,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                    highlight_mode=rawpy.HighlightMode.Clip
                )

                rgb_normalized = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
                rgb_8bit = rgb_normalized.astype(np.uint8)

                # Resize while keeping aspect ratio
                max_width, max_height = max_dims
                scale = min(max_width / width, max_height / height)
                new_size = (int(width * scale), int(height * scale))
                rgb_8bit = cv2.resize(rgb_8bit, new_size, interpolation=cv2.INTER_AREA)

                if output_path.lower().endswith('.png'):
                    cv2.imwrite(output_path, cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                else:
                    raise ValueError("Output format must be PNG")

                return True

        except Exception as e:
            print(f"Error converting {input_path}: {str(e)}")
            return False