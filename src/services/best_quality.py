import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
mp.set_start_method('spawn', force=True)
import multiprocessing
max_worker = min(multiprocessing.cpu_count(), 16)
# Third-party imports
from tqdm import tqdm
import gc
from src.core.gemini_image_analyzer import GeminiImageAnalyzer
from src.core.processing_cache import ProcessingCache

from src.utils.util import setup_directories, get_existing_results, should_exclude_image, save_json_report, get_all_image_files

from ..config import logger, log_critical
try:
    multiprocessing.set_start_method('fork')  # Fix for macOS
except:
    pass

try:
    multiprocessing.active_children()

    # multiprocessing.resource_tracker.unregister("/dev/shm", "semaphore")
except:
    pass


# std upt

def process_best_quality(input_dir: str, output_dir: str, raw_results: dict, config: dict, 
                         cache: 'ProcessingCache' = None,
                         exclude_files: set = None,
                         allowed_duplicates: set = None) -> dict:
    if cache is None:
        cache = ProcessingCache(output_dir)
    if exclude_files is None:
        exclude_files = set()
    if allowed_duplicates is None:
        allowed_duplicates = set()
        
    directories = setup_directories(output_dir)
    analyzer = GeminiImageAnalyzer()
    
    best_quality_min_score = config['thresholds'].get('best_quality_min_score', 50)
    gemini_confidence_min = config['thresholds'].get('gemini_confidence_min', 90)
    
    results = {
        'best_quality': [],
        'rejected': [],
        'stats': {
            'total_processed': 0,
            'best_selected': 0,
            'duplicates_handled': 0,
            'quality_issues': 0,
            'processing_errors': 0,
            'skipped': 0,
            'excluded': {
                'duplicate': 0,
                'blur': 0,
                'focus': 0,
                'eyes': 0,
                'quality': 0,
                'other': 0
            }
        }
    }
    
    # Ensure all exclusion categories exist
    exclusion_categories = ['duplicate', 'blur', 'focus', 'eyes', 'quality', 'other']
    for category in exclusion_categories:
        if category not in results['stats']['excluded']:
            results['stats']['excluded'][category] = 0

    print("\n[cyan]Scanning directories for supported files...")
    processed_files = set()
    image_files = []
    in_focus_files = []
    
    # First collect from input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                full_path = os.path.join(root, file)
                if full_path not in processed_files:
                    image_files.append(full_path)
                    processed_files.add(full_path)
    
    # Then check in_focus directory
    in_focus_dir = os.path.join(output_dir, 'in_focus')
    if os.path.exists(in_focus_dir):
        for root, _, files in os.walk(in_focus_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.arw', '.cr2', '.nef')):
                    full_path = os.path.join(root, file)
                    base_name = os.path.splitext(os.path.basename(file))[0]
                    is_duplicate = any(os.path.splitext(os.path.basename(p))[0] == base_name 
                                       for p in processed_files)
                    if not is_duplicate:
                        in_focus_files.append(full_path)
                        processed_files.add(full_path)
    
    # Combine unique files
    all_files = list(set(image_files + in_focus_files))
    
    print(f"\n[cyan]Processing {len(all_files)} unique files...")
    if not all_files:
        print("No images found to process")
        return results
        
    print(f"[cyan]Total files found: {len(all_files)}")
    
    previous_results = get_existing_results(output_dir)
    # Reset processed_files for image name checks during processing
    processed_files = set()
    
    # Lock for synchronizing updates to shared data structures
    lock = threading.Lock()

    # Worker function to process one image
    def process_image(img_path):
        try:
            img_name = os.path.basename(img_path)
            with lock:
                if img_name in processed_files:
                    results['stats']['skipped'] += 1
                    return

            is_in_focus_file = in_focus_dir in img_path
            if not is_in_focus_file:
                should_exclude, reason = should_exclude_image(img_name, previous_results, allowed_duplicates)
                if should_exclude:
                    with lock:
                        if reason not in results['stats']['excluded']:
                            results['stats']['excluded'][reason] = 0
                        results['stats']['excluded'][reason] += 1
                        results['rejected'].append({
                            'path': str(img_path),
                            'reason': f"Failed quality check: {reason}",
                            'score': 0
                        })
                        processed_files.add(img_name)
                    return

            original_image_path = img_path
            if raw_results and isinstance(raw_results.get('converted'), list):
                for converted in raw_results['converted']:
                    if isinstance(converted, dict):
                        if Path(converted.get('converted_path', '')).stem == Path(img_path).stem:
                            original_image_path = converted.get('raw_path', img_path)
                            break

            if is_in_focus_file:
                in_focus_report_path = os.path.join(output_dir, 'in_focus', 'report.json')
                if os.path.exists(in_focus_report_path):
                    with open(in_focus_report_path, 'r') as f:
                        focus_report = json.load(f)
                        for focus_img in focus_report.get('processed_images', {}).get('in_focus', []):
                            if os.path.basename(focus_img.get('new_path', '')) == img_name:
                                original_image_path = focus_img.get('original_path', img_path)
                                break

            analysis = analyzer.analyze_single_image(img_path)
            if not analysis:
                with lock:
                    results['stats']['processing_errors'] += 1
                return

            with lock:
                results['stats']['total_processed'] += 1
            decision = analysis.get('decision', '')
            score = analysis.get('score', 0)

            if is_in_focus_file and score >= best_quality_min_score:
                decision = 'BEST'

            if decision == 'BEST' and score >= best_quality_min_score:
                if analysis.get('gemini_status') == 'REJECT':
                    with lock:
                        results['stats']['excluded']['quality'] += 1
                        results['rejected'].append({
                            'path': str(img_path),
                            'reason': analysis.get('reason', 'Rejected by Gemini analysis'),
                            'score': score,
                            'gemini_status': decision,
                            'confidence': float(score)
                        })
                        processed_files.add(img_name)
                    return

                rel_path = os.path.relpath(os.path.dirname(img_path),
                                           input_dir if not is_in_focus_file else in_focus_dir)
                dest_dir = os.path.join(directories['best_quality'], rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(original_image_path))

                if os.path.exists(str(original_image_path)):
                    shutil.copy2(str(original_image_path), dest_path)
                    with lock:
                        results['best_quality'].append({
                            'original_path': str(original_image_path),
                            'new_path': dest_path,
                            'score': score,
                            'reason': analysis.get('reason', ''),
                            'gemini_status': analysis.get('decision', ''),
                            'confidence': float(analysis.get('score', 0))
                        })
                        results['stats']['best_selected'] += 1
                        cache.add_processed_image(img_name, 'best_quality')
                        processed_files.add(img_name)
            else:
                with lock:
                    results['stats']['excluded']['quality'] += 1
                    results['rejected'].append({
                        'path': str(img_path),
                        'reason': analysis.get('reason', 'Failed quality criteria'),
                        'score': score,
                        'gemini_status': decision,
                        'confidence': float(score)
                    })
                    processed_files.add(img_name)
        except Exception as e:
            with lock:
                results['stats']['processing_errors'] += 1
            logger.error(f"Error processing {img_path}: {str(e)}")
            if 'eyes' in str(e):
                with lock:
                    if 'eyes' not in results['stats']['excluded']:
                        results['stats']['excluded']['eyes'] = 0
                    results['stats']['excluded']['eyes'] += 1
                    gc.collect()  # Free memory

    # Use ThreadPoolExecutor to process images concurrently
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = [executor.submit(process_image, img_path) for img_path in all_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing images for best quality"):
            pass  # futures update the results internally

    # Save report at root level  
    save_json_report(output_dir, results, config, 'best_quality_report.json')
    
    # Print summary
    print(f"\nBest Quality Detection Summary:")
    print(f"Total processed: {results['stats']['total_processed']}")
    print(f"Best quality selected: {results['stats']['best_selected']}")
    print("\nExcluded Images:")
    print(f"- Duplicates: {results['stats']['excluded']['duplicate']}")
    print(f"- Blurry: {results['stats']['excluded']['blur']}")
    print(f"- Focus issues: {results['stats']['excluded']['focus']}")
    print(f"- Eye issues: {results['stats']['excluded']['eyes']}")
    print(f"- Quality issues: {results['stats']['excluded']['quality']}")
    print(f"- Other exclusions: {results['stats']['excluded']['other']}")
    print(f"Processing errors: {results['stats']['processing_errors']}")
    print(f"Skipped: {results['stats']['skipped']}")
    
    return results




