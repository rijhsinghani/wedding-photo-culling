import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
import shutil
from typing import Dict
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import logging
import multiprocessing
import sys
# Third-party imports
from tqdm import tqdm


from src.utils.process_raw import _process_raw_file

from ..config import logger, log_critical


def preprocess_raw_images(input_dir: str, output_dir: str) -> Dict:
    """Convert all RAW images in input directory and subdirectories to PNG format using multiprocessing."""
    raw_extensions = {'.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2', '.dng', '.raw'}
    standard_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    
    results = {
        'converted': [],
        'failed': [],
        'skipped': []
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all files for processing
    raw_files = []
    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(current_output_dir, exist_ok=True)
        
        for file in files:
            file_ext = os.path.splitext(file.lower())[1]
            input_path = os.path.join(root, file)
            
            if file_ext in raw_extensions:
                output_path = os.path.join(
                    current_output_dir, 
                    os.path.splitext(file)[0] + '.png'
                )
                raw_files.append((input_path, output_path, rel_path, file))
            elif file_ext in standard_extensions:
                output_path = os.path.join(current_output_dir, file)
                try:
                    shutil.copy2(input_path, output_path)
                    results['skipped'].append(input_path)
                except Exception as e:
                    print(f"Error copying {input_path}: {str(e)}")
    
    # Process RAW files with multiprocessing
    if raw_files:
        max_workers = min(multiprocessing.cpu_count(), 16)
        print(f"\nProcessing {len(raw_files)} RAW files using {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_raw_file, args) for args in raw_files]
            
            with tqdm(total=len(raw_files), desc="Converting RAW files") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result['status'] == 'converted':
                            results['converted'].append(result['data'])
                        else:
                            results['failed'].append(result['data'])
                    except Exception as e:
                        print(f"Error processing RAW file: {str(e)}")
                    pbar.update(1)
    
    # Print summary
    print(f"\nConversion Summary:")
    print(f"Successfully converted: {len(results['converted'])} files")
    print(f"Failed conversions: {len(results['failed'])} files")
    print(f"Skipped (non-RAW) files: {len(results['skipped'])} files")
    
    return results