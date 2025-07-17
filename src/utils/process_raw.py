import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
from typing import Dict, List, Tuple
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Third-party imports

from src.core.enhanced_raw_converter import EnhancedRawConverter


def _process_raw_file(args: Tuple[str, str, str, str]) -> Dict:
    """Process a single RAW file (used by multiprocessing)."""
    input_path, output_path, rel_path, file = args
    converter = EnhancedRawConverter()
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = converter.convert_raw_image(input_path, output_path)
        
        if success:
            return {
                'status': 'converted',
                'data': {
                    'raw_path': input_path,
                    'converted_path': output_path,
                    'relative_path': rel_path,
                    'filename': file
                }
            }
        else:
            return {
                'status': 'failed',
                'data': input_path
            }
    except Exception as e:
        return {
            'status': 'failed',
            'data': input_path,
            'error': str(e)
        }
    





