import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import threading
from tqdm import tqdm
import multiprocessing
max_worker = min(multiprocessing.cpu_count(), 16)
from src.core.processing_cache import ProcessingCache
from src.core.eye_detector import EyeDetector
from src.utils.util import setup_directories, save_json_report

from ..config import logger, log_critical



def process_closed_eyes(input_dir: str, output_dir: str, raw_results: dict, config: dict, 
                        cache: 'ProcessingCache' = None) -> dict:
    """Enhanced process_closed_eyes function with Gemini integration."""
    print("\nProcessing closed eyes detection...")
    if cache is None:
        cache = ProcessingCache(output_dir)
        
    directories = setup_directories(output_dir)
    detector = EyeDetector()
    
    results = {
        'closed_eyes': [],
        'open_eyes': [],
        'no_face_detected': [],
        'stats': {
            'total_processed': 0,
            'closed_eyes_detected': 0,
            'processing_errors': 0,
            'skipped': 0,
            'no_face': 0
        }
    }
    
    # Collect all image files from subdirectories.
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_files.append(os.path.join(root, file))
                
    # Lock to synchronize updates to results.
    results_lock = threading.Lock()

    def process_image(img_path):
        img_name = os.path.basename(img_path)
        # Check if image should be processed.
        if not cache.should_process_image(img_name, 'closed_eyes'):
            with results_lock:
                results['stats']['skipped'] += 1
            return

        # Check for original RAW file.
        original_image_path = None
        if raw_results and raw_results.get('converted'):
            for converted in raw_results['converted']:
                raw_path = Path(converted['raw_path'])
                img_path_obj = Path(img_path)
                if raw_path.stem == img_path_obj.stem:
                    original_image_path = raw_path
                    break
        if not original_image_path:
            original_image_path = img_path

        try:
            # Perform eye detection.
            has_closed_eyes, confidence, reason = detector.detect_closed_eyes(img_path)
            with results_lock:
                results['stats']['total_processed'] += 1

            # Process based on result.
            if has_closed_eyes:
                # Maintain directory structure.
                rel_path = os.path.relpath(os.path.dirname(img_path), input_dir)
                dest_dir = os.path.join(directories['closed_eyes'], rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(str(original_image_path)))
                shutil.copy2(str(original_image_path), dest_path)
                with results_lock:
                    results['closed_eyes'].append({
                        'original_path': str(original_image_path),
                        'new_path': dest_path,
                        'confidence': float(confidence),
                        'reason': reason
                    })
                    results['stats']['closed_eyes_detected'] += 1
                    cache.add_processed_image(img_name, 'closed_eyes')
            elif reason in ["No faces detected", "Face too small - likely venue/decor shot", "No significant faces - skip eye detection"]:
                with results_lock:
                    results['no_face_detected'].append({
                        'path': str(original_image_path),
                        'reason': reason
                    })
                    results['stats']['no_face'] += 1
            else:
                with results_lock:
                    results['open_eyes'].append({
                        'path': str(original_image_path),
                        'confidence': float(confidence),
                        'reason': reason
                    })
        except Exception as e:
            with results_lock:
                results['stats']['processing_errors'] += 1
            print(f"Error processing {img_name}: {str(e)}")

    # Process images concurrently.
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = [executor.submit(process_image, img_path) for img_path in image_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing images for closed eyes"):
            pass

    # Save JSON report at root level.
    save_json_report(directories['closed_eyes'], results, config, 'closed_eyes_report.json')
    
    # Print summary.
    print(f"\nClosed Eyes Detection Summary:")
    print(f"Total processed: {results['stats']['total_processed']}")
    print(f"Closed eyes detected: {results['stats']['closed_eyes_detected']}")
    print(f"Images without faces: {results['stats']['no_face']}")
    print(f"Processing errors: {results['stats']['processing_errors']}")
    print(f"Skipped: {results['stats']['skipped']}")
    
    return results




