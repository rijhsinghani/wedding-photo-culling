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
import logging
import multiprocessing
max_worker = min(multiprocessing.cpu_count(), 16)
# Third-party imports
from tqdm import tqdm
import threading



from src.core.processing_cache import ProcessingCache
from src.core.focus_detector import FocusDetector
from src.utils.util import setup_directories, get_existing_results, should_exclude_image, save_json_report

from ..config import logger, log_critical






#org

def process_focus(input_dir: str, output_dir: str, raw_results: Dict, config: Dict, 
                cache: Optional[ProcessingCache] = None,
                exclude_files: Set[str] = None,
                allowed_duplicates: Set[str] = None) -> Dict:
    if cache is None:
        cache = ProcessingCache(output_dir)
    if exclude_files is None:
        exclude_files = set()
    if allowed_duplicates is None:
        allowed_duplicates = set()
       
    directories = setup_directories(output_dir)
    detector = FocusDetector(config)
    
    focus_threshold = config['thresholds'].get('focus_threshold', 45)
    in_focus_confidence = config['thresholds'].get('in_focus_confidence', 85)
    
    results = {
        'in_focus': [],
        'out_focus': [],
        'blurry': [],
        'no_subject': [],
        'stats': {
            'total_processed': 0,
            'in_focus_detected': 0,
            'out_focus_detected': 0,
            'blurry_detected': 0,
            'no_subject': 0,
            'processing_errors': 0,
            'skipped': 0,
            'excluded': {
                'duplicate': 0,
                'closed_eyes': 0,
                'blurry': 0,
                'eyes': 0,
                'focus': 0,
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
    image_files = []
    for root, _, files in os.walk(input_dir):
        files_in_dir = []
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                img_name = os.path.basename(file)
                if img_name not in exclude_files or img_name in allowed_duplicates:
                    full_path = os.path.join(root, file)
                    files_in_dir.append(full_path)
        
        if files_in_dir:
            rel_path = os.path.relpath(root, input_dir)
            print(f"[green]Found {len(files_in_dir)} files in: {rel_path}")
            image_files.extend(files_in_dir)
    
    if not image_files:
        print("No images found to process")
        return results
        
    print(f"[cyan]Total files found: {len(image_files)}")

    previous_results = get_existing_results(output_dir)
    processed_files = set()

    # Create best_quality directory
    best_quality_dir = os.path.join(output_dir, 'best_quality')
    poor_quality_dir = os.path.join(output_dir, 'poor_quality')  # New combined directory
    os.makedirs(best_quality_dir, exist_ok=True)
    os.makedirs(poor_quality_dir, exist_ok=True)

    for img_path in tqdm(image_files, desc="Analyzing focus"):
        img_name = os.path.basename(img_path)
        
        try:
            should_exclude, reason = should_exclude_image(img_name, previous_results, allowed_duplicates)
            if should_exclude:
                if reason not in results['stats']['excluded']:
                    results['stats']['excluded'][reason] = 0
                results['stats']['excluded'][reason] += 1
                continue

            if img_name in processed_files:
                results['stats']['skipped'] += 1
                continue

            original_image_path = img_path
            if raw_results and isinstance(raw_results.get('converted'), list):
                for converted in raw_results['converted']:
                    if isinstance(converted, dict):
                        if Path(converted.get('converted_path', '')).stem == Path(img_path).stem:
                            original_image_path = converted.get('raw_path', img_path)
                            break

            focus_result = detector.analyze_image(img_path)
            if not focus_result or not isinstance(focus_result, dict):
                results['stats']['processing_errors'] += 1
                continue

            results['stats']['total_processed'] += 1
            rel_path = os.path.relpath(os.path.dirname(img_path), input_dir)

            focus_score = focus_result.get('focus_score', 0)
            gemini_confidence = focus_result.get('confidence', 0)

            if (focus_result.get('status') == "in_focus" and 
                focus_result.get('gemini_status') == "IN_FOCUS" and 
                focus_score >= focus_threshold and 
                gemini_confidence >= in_focus_confidence):
                
                dest_dir = os.path.join(directories['in_focus'], rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(original_image_path))
                
                if os.path.exists(str(original_image_path)):
                    shutil.copy2(str(original_image_path), dest_path)
                    results['in_focus'].append({
                        'original_path': str(original_image_path),
                        'new_path': dest_path,
                        'focus_score': focus_score,
                        'confidence': gemini_confidence,
                        'detail': focus_result.get('detail', '')
                    })
                    results['stats']['in_focus_detected'] += 1
                    cache.add_processed_image(img_name, 'in_focus')
                    processed_files.add(img_name)

                    best_quality_dest_dir = os.path.join(best_quality_dir, rel_path)
                    os.makedirs(best_quality_dest_dir, exist_ok=True)
                    best_quality_dest_path = os.path.join(best_quality_dest_dir, 
                                                        os.path.basename(original_image_path))
                    
                    shutil.copy2(str(original_image_path), best_quality_dest_path)
                    cache.add_processed_image(img_name, 'best_quality')
            
            elif focus_result.get('status') == "off_focus":
                # Save to poor_quality directory instead
                dest_dir = os.path.join(poor_quality_dir, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(original_image_path))
                
                if os.path.exists(str(original_image_path)):
                    shutil.copy2(str(original_image_path), dest_path)
                    results['out_focus'].append({
                        'original_path': str(original_image_path),
                        'new_path': dest_path,
                        'focus_score': focus_score,
                        'confidence': gemini_confidence,
                        'detail': focus_result.get('detail', ''),
                        'issue_type': 'focus'  # Added to track issue type
                    })
                    results['stats']['out_focus_detected'] += 1
                    cache.add_processed_image(img_name, 'poor_quality')  # Updated cache key
                    processed_files.add(img_name)
            
            else:
                if focus_result.get('status') == "blur":
                    # Save to poor_quality directory
                    dest_dir = os.path.join(poor_quality_dir, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, os.path.basename(original_image_path))
                    
                    if os.path.exists(str(original_image_path)):
                        shutil.copy2(str(original_image_path), dest_path)
                        results['blurry'].append({
                            'original_path': str(original_image_path),
                            'new_path': dest_path,
                            'detail': focus_result.get('detail', ''),
                            'issue_type': 'blur'  # Added to track issue type
                        })
                        results['stats']['blurry_detected'] += 1
                        cache.add_processed_image(img_name, 'poor_quality')  # Updated cache key
                else:
                    results['no_subject'].append({
                        'path': str(original_image_path),
                        'detail': focus_result.get('detail', '')
                    })
                    results['stats']['no_subject'] += 1
                processed_files.add(img_name)

        except Exception as e:
            results['stats']['processing_errors'] += 1
            logger.error(f"Error processing {img_name}: {str(e)}")
            if 'eyes' in str(e):
                if 'eyes' not in results['stats']['excluded']:
                    results['stats']['excluded']['eyes'] = 0
                results['stats']['excluded']['eyes'] += 1
            continue

    # Save reports to both directories for backwards compatibility
    save_json_report(directories['in_focus'], results, config)
    save_json_report(poor_quality_dir, results, config)

    log_file = os.path.join(output_dir, 'focus_analysis.log')
    logger.addHandler(logging.FileHandler(log_file))
    logger.setLevel(logging.INFO)
    
    logger.info("Starting focus analysis with configuration:")
    logger.info(f"Focus threshold: {config['thresholds'].get('focus_threshold', 40)}")
    logger.info(f"In-focus confidence: {config['thresholds'].get('in_focus_confidence', 75)}")
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            results['debug_log'] = f.read()
            
    return results
