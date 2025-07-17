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
import sys

# Third-party imports
from skimage.metrics import structural_similarity as ssim
from dotenv import load_dotenv
load_dotenv()

from src.core.processing_cache import ProcessingCache
from src.core.image_preprocessor import ImagePreprocessor
from src.main import print_detailed_summary
from src.utils.util import setup_directories, save_json_report, get_all_image_files,cleanup_temp_directory

from src.services.preprocess import preprocess_raw_images
from src.services.best_quality import process_best_quality
from src.services.blur import process_blur
from src.services.closed_eyes import process_closed_eyes
from src.services.duplicates import process_duplicates
from src.services.focus import process_focus

import logging
import multiprocessing
max_worker = min(multiprocessing.cpu_count(), 16)

from ..config import logger, log_critical



#updated

# Define helper functions
def run_process_blur(args):
    return process_blur(*args)

def run_process_duplicates(args):
    return process_duplicates(*args)

def run_process_closed_eyes(args):
    return process_closed_eyes(*args)

def run_process_focus(args):
    return process_focus(*args)

def run_process_best_quality(args):
    return process_best_quality(*args)


def process_all(input_dir: str, output_dir: str, raw_results: Dict, config: Dict) -> Dict:
    logging.basicConfig(
        filename=os.path.join(output_dir, 'processing.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    cache = ProcessingCache(output_dir)
    temp_dirs = []
    processed_files = set()
    best_images = set()
    excluded_files = set()
    venue_shots = []
    
    directories = setup_directories(output_dir)
    preprocessor = ImagePreprocessor()
    
    # Parallelized RAW file scanning
    print("\nScanning for RAW files...")
    raw_extensions = {'.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf'}
    raw_files_found = list(Path(input_dir).rglob("*"))
    raw_files_found = [str(f) for f in raw_files_found if f.suffix.lower() in raw_extensions]
    
    if raw_files_found:
        temp_dir = os.path.join(output_dir, 'temp_converted')
        temp_dirs.append(temp_dir)
        raw_results = preprocess_raw_images(input_dir, output_dir)
        input_dir = temp_dir
    
    # Parallelized venue shot detection
    print("\nScanning for venue/decoration shots...")
    image_files = list(Path(input_dir).rglob("*"))
    image_files = [str(f) for f in image_files if f.suffix.lower() in config['supported_formats']]
    
    def process_venue(file_path):
        is_venue, confidence, reason = preprocessor.is_venue_shot(file_path)
        if is_venue and confidence >= 90:
            return file_path, confidence, reason
        return None
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        venue_results = list(executor.map(process_venue, image_files))
    
    for result in venue_results:
        if result:
            file_path, confidence, reason = result
            venue_shots.append({
                'original_path': file_path,
                'confidence': confidence,
                'reason': reason
            })
            excluded_files.add(Path(file_path).name)
    
   
    tasks = {
        'blur': (input_dir, output_dir, raw_results, config, cache),
        'duplicates': (input_dir, output_dir, raw_results, config, cache),
        'closed_eyes': (input_dir, output_dir, raw_results, config, cache),
        'focus': (input_dir, output_dir, raw_results, config, cache, excluded_files, best_images),
        'best_quality': (input_dir, output_dir, raw_results, config, cache, excluded_files, best_images),
    }

    # Function mapping
    task_functions = {
        'blur': run_process_blur,
        'duplicates': run_process_duplicates,
        'closed_eyes': run_process_closed_eyes,
        'focus': run_process_focus,
        'best_quality': run_process_best_quality,
    }

    results = {}

    with ProcessPoolExecutor(max_workers=max_worker) as executor:
        future_to_task = {
            executor.submit(task_functions[task], args): task for task, args in tasks.items()
        }
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                results[task_name] = future.result()
            except Exception as e:
                logger.error(f"Error in {task_name}: {str(e)}")
    
    # Update excluded files from results
    for group in results.get('duplicates', {}).get('duplicates', {}).values():
        if isinstance(group, dict):
            best_image = group.get('best_image')
            if best_image:
                best_images.add(os.path.basename(best_image))
            for dup_info in group.get('duplicates', {}).values():
                excluded_files.add(os.path.basename(str(dup_info.get('original_path', ''))))
    
    # Save report
    final_results = {
        'processing_results': results,
        'stats': {
            'raw_files': {
                'found': len(raw_files_found),
                'converted': len(raw_results['converted']) if raw_results else 0,
                'failed': len(raw_results['failed']) if raw_results else 0
            },
            'venue_shots': len(venue_shots),
            'duplicate_groups': len(results.get('duplicates', {}).get('duplicates', {})),
            'blurry_detected': len(results.get('blur', {}).get('blurry', [])),
            'closed_eyes': len(results.get('closed_eyes', {}).get('closed_eyes', [])),
            'in_focus': len(results.get('focus', {}).get('in_focus', [])),
            'out_focus': len(results.get('focus', {}).get('out_focus', [])),
            'best_quality': len(results.get('best_quality', {}).get('best_quality', []))
        }
    }
    
    save_json_report(output_dir, final_results, config)
    print("\nProcessing completed successfully!")
    return final_results



