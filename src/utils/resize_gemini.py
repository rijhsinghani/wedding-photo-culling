import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
from typing import Optional
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import time
import sys

# Third-party imports
import rawpy
from PIL import Image

from ..config import logger, log_critical

def resize_for_gemini(image_path: str, max_dimension: int = 1024, quality: int = 85) -> Optional[str]:
    try:
        # Add raw file handling
        if image_path.lower().endswith(('.arw', '.cr2', '.nef')):
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(output_color=rawpy.ColorSpace.sRGB)
                img = Image.fromarray(rgb)
        else:
            img = Image.open(image_path)
            
        # Create temp directory in same location as input file
        temp_dir = os.path.join(os.path.dirname(image_path), 'temp_gemini')
        os.makedirs(temp_dir, exist_ok=True)
        
        output_path = os.path.join(
            temp_dir, 
            f"gemini_{os.path.splitext(os.path.basename(image_path))[0]}_{int(time.time())}.jpg"
        )
        
        ratio = min(max_dimension / float(img.size[0]), max_dimension / float(img.size[1]))
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=quality, optimize=True)
        
        return output_path
    except Exception as e:
        logger.error(f"Failed to resize image for analysis: {image_path}")
        return None