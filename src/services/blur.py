import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import multiprocessing
max_worker = min(multiprocessing.cpu_count(), 16)

# Third-party imports

import numpy as np
import cv2
from tqdm import tqdm
import threading


from src.core.processing_cache import ProcessingCache
from src.core.blur_detector import BlurDetector
from src.utils.util import setup_directories, get_existing_results, should_exclude_image, save_json_report, get_all_image_files


from ..config import logger, log_critical


# Assume these functions and objects are imported from your project:
# - get_all_image_files, setup_directories, save_json_report, logger
# - BlurDetector, ProcessingCache

def process_blur(input_dir: str, output_dir: str, raw_results: dict, config: dict, 
                 cache: 'ProcessingCache' = None) -> dict:
    if cache is None:
        cache = ProcessingCache(output_dir)
        
    directories = setup_directories(output_dir)
    detector = BlurDetector(threshold=config.get('thresholds', {}).get('blur_threshold', 25))
    
    # Use blurry directory from setup_directories
    
    results = {
        'blurry': [],
        'clear': [],
        'stats': {
            'total_processed': 0,
            'blur_detected': 0,
            'overexposed_detected': 0,
            'processing_errors': 0,
            'skipped': 0
        }
    }
    
    image_files = get_all_image_files(input_dir, config['supported_formats'])
    
    # Lock to synchronize shared result updates.
    results_lock = threading.Lock()
    
    def process_image(image_path: str):
        # Check if the image should be processed using the cache.
        if not cache.should_process_image(image_path, 'poor_quality'):
            with results_lock:
                results['stats']['skipped'] += 1
            return

        try:
            original_path = image_path
            if raw_results and isinstance(raw_results.get('converted'), list):
                for converted in raw_results['converted']:
                    if isinstance(converted, dict):
                        if Path(converted.get('converted_path', '')).stem == Path(image_path).stem:
                            original_path = converted.get('raw_path', image_path)
                            break

            # Check for exposure issues.
            image = cv2.imread(image_path)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                overexposed_pixels = np.sum(gray > 240)
                total_pixels = gray.size
                overexposed_ratio = overexposed_pixels / total_pixels
                
                # If severely overexposed, copy to poor_quality.
                if overexposed_ratio > 0.30:
                    rel_path = os.path.relpath(os.path.dirname(image_path), input_dir)
                    dest_dir = os.path.join(directories['blurry'], rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    dest_path = os.path.join(dest_dir, os.path.basename(original_path))
                    shutil.copy2(str(original_path), dest_path)
                    
                    with results_lock:
                        results['blurry'].append({
                            'original_path': str(original_path),
                            'new_path': dest_path,
                            'reason': f"Severe overexposure ({overexposed_ratio:.2%} overexposed)",
                            'exposure_ratio': float(overexposed_ratio),
                            'issue_type': 'exposure'
                        })
                        results['stats']['overexposed_detected'] += 1
                    cache.add_processed_image(image_path, 'blurry')
                    return

            # Regular blur detection.
            blur_result = detector.detect_blur(image_path)
            if "error" in blur_result:
                with results_lock:
                    results['stats']['processing_errors'] += 1
                return

            with results_lock:
                results['stats']['total_processed'] += 1
            
            if blur_result["is_blurry"] and blur_result["confidence"] >= 90:
                rel_path = os.path.relpath(os.path.dirname(image_path), input_dir)
                dest_dir = os.path.join(directories['blurry'], rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                
                dest_path = os.path.join(dest_dir, os.path.basename(original_path))
                shutil.copy2(str(original_path), dest_path)
                
                with results_lock:
                    results['blurry'].append({
                        'original_path': str(original_path),
                        'new_path': dest_path,
                        'blur_scores': blur_result,
                        'confidence': float(blur_result["confidence"]),
                        'reason': blur_result["reason"],
                        'issue_type': 'blur'
                    })
                    results['stats']['blur_detected'] += 1
                cache.add_processed_image(image_path, 'poor_quality')
            else:
                with results_lock:
                    results['clear'].append({
                        'path': str(original_path),
                        'blur_scores': blur_result,
                        'reason': blur_result["reason"]
                    })

        except Exception as e:
            with results_lock:
                results['stats']['processing_errors'] += 1
            logger.error(f"Error processing {image_path}: {str(e)}")
    
    # Process all images concurrently.
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = [executor.submit(process_image, image_path) for image_path in image_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Detecting blur"):
            pass

    # Save report at root level
    save_json_report(output_dir, results, config, 'blur_report.json')
    
    # Print summary.
    print(f"\nBlur Detection Summary:")
    print(f"Total processed: {results['stats']['total_processed']}")
    print(f"Blurry images detected: {results['stats']['blur_detected']}")
    print(f"Overexposed images detected: {results['stats']['overexposed_detected']}")
    print(f"Processing errors: {results['stats']['processing_errors']}")
    print(f"Skipped: {results['stats']['skipped']}")
    
    return results



