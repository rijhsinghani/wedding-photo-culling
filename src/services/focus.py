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
import cv2



from src.core.processing_cache import ProcessingCache
from src.core.focus_detector import FocusDetector
from src.core.batch_gemini_processor import BatchGeminiProcessor
from src.utils.util import setup_directories, get_existing_results, should_exclude_image, save_json_report

from ..config import logger, log_critical






def _process_focus_result(focus_result, img_path, original_image_path, results, directories, 
                         cache, processed_files, focus_threshold, in_focus_confidence):
    """Process a single focus result and update results/files accordingly"""
    img_name = os.path.basename(img_path)
    # Get the relative path from the directory structure
    img_dir = os.path.dirname(img_path)
    # Find the base directory (should be temp_converted or similar)
    base_dir = img_dir
    while os.path.basename(base_dir) not in ['temp_converted', 'converted', 'output'] and base_dir != '/':
        parent = os.path.dirname(base_dir)
        if parent == base_dir:  # Reached root
            break
        base_dir = parent
    rel_path = os.path.relpath(img_dir, base_dir) if base_dir != img_dir else '.'
    
    if not focus_result or not isinstance(focus_result, dict):
        results['stats']['processing_errors'] += 1
        logger.warning(f"Focus analysis failed for {img_name}: Invalid result")
        return
    
    # Validate that required keys exist
    required_keys = ['status', 'gemini_status', 'focus_score', 'confidence']
    missing_keys = [key for key in required_keys if key not in focus_result]
    if missing_keys:
        results['stats']['processing_errors'] += 1
        logger.error(f"Focus result missing keys for {img_name}: {missing_keys}. Actual keys: {list(focus_result.keys())}")
        return
    
    results['stats']['total_processed'] += 1
    
    focus_score = focus_result.get('focus_score', 0)
    gemini_confidence = focus_result.get('confidence', 0)
    status = focus_result.get('status', 'unknown')
    gemini_status = focus_result.get('gemini_status', 'UNKNOWN')
    
    if (status == "in_focus" and 
        gemini_status == "IN_FOCUS" and 
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
            
            best_quality_dest_dir = os.path.join(directories['best_quality'], rel_path)
            os.makedirs(best_quality_dest_dir, exist_ok=True)
            best_quality_dest_path = os.path.join(best_quality_dest_dir, 
                                                os.path.basename(original_image_path))
            
            shutil.copy2(str(original_image_path), best_quality_dest_path)
            cache.add_processed_image(img_name, 'best_quality')
    
    elif status == "off_focus":
        # Save to blurry directory
        dest_dir = os.path.join(directories['blurry'], rel_path)
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
                'issue_type': 'focus'
            })
            results['stats']['out_focus_detected'] += 1
            cache.add_processed_image(img_name, 'blurry')
            processed_files.add(img_name)
    
    else:
        # Handle blur, no_subject, error, and other cases
        if status == "blur":
            # Save to blurry directory
            dest_dir = os.path.join(directories['blurry'], rel_path)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(original_image_path))
            
            if os.path.exists(str(original_image_path)):
                shutil.copy2(str(original_image_path), dest_path)
                results['blurry'].append({
                    'original_path': str(original_image_path),
                    'new_path': dest_path,
                    'detail': focus_result.get('detail', ''),
                    'issue_type': 'blur'
                })
                results['stats']['blurry_detected'] += 1
                cache.add_processed_image(img_name, 'blurry')
        elif status == "no_subject":
            results['no_subject'].append({
                'path': str(original_image_path),
                'detail': focus_result.get('detail', '')
            })
            results['stats']['no_subject'] += 1
        elif status == "error":
            # This is a processing error, increment error counter
            results['stats']['processing_errors'] += 1
            logger.error(f"Focus analysis error for {img_name}: {focus_result.get('detail', 'Unknown error')}")
        else:
            # Log unknown status for debugging
            logger.warning(f"Unknown focus status '{status}' for {img_name}")
            results['stats']['processing_errors'] += 1
    
    processed_files.add(img_name)


#org

def process_focus(input_dir: str, output_dir: str, raw_results: Dict, config: Dict, 
                cache: Optional[ProcessingCache] = None,
                exclude_files: Set[str] = None,
                allowed_duplicates: Set[str] = None,
                use_batch_processing: bool = None) -> Dict:
    if cache is None:
        cache = ProcessingCache(output_dir)
    if exclude_files is None:
        exclude_files = set()
    if allowed_duplicates is None:
        allowed_duplicates = set()
    
    # Use config setting if use_batch_processing is not explicitly provided
    if use_batch_processing is None:
        use_batch_processing = config.get('focus_processing', {}).get('use_batch_processing', True)
       
    directories = setup_directories(output_dir)
    detector = FocusDetector(config)
    
    # Initialize batch processor if enabled
    batch_processor = None
    if use_batch_processing and config.get('batch_gemini_processing', {}).get('enabled', True):
        batch_config = config.get('batch_gemini_processing', {})
        batch_processor = BatchGeminiProcessor(
            batch_size=batch_config.get('batch_size', 5),
            max_workers=batch_config.get('max_workers', 3),
            use_queue_manager=batch_config.get('use_queue_manager', True)
        )
    
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

    # Directories are already created by setup_directories

    # Use batch processing if enabled and we have multiple images
    if use_batch_processing and batch_processor and len(image_files) > 1:
        print(f"[cyan]Using batch processing with batch size {batch_processor.batch_size}")
        
        # Prepare images for batch processing
        images_to_process = []
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            should_exclude, reason = should_exclude_image(img_name, previous_results, allowed_duplicates)
            if not should_exclude and img_name not in processed_files:
                images_to_process.append(img_path)
        
        # Process in batches
        if images_to_process:
            batch_result = batch_processor.process_directory(
                os.path.dirname(images_to_process[0]),
                detector.gemini_prompt,
                extensions=('.png', '.jpg', '.jpeg')
            )
            
            # Process batch results
            for img_path, batch_focus_result in batch_result.results.items():
                img_name = os.path.basename(img_path)
                
                # Transform batch result to match expected format
                # The batch processor returns: status, confidence, reason
                # We need: focus_score, gemini_status, confidence, detail, status
                
                # Get technical focus score from detector
                image = cv2.imread(img_path)
                focus_score = detector.calculate_focus_score(image) if image is not None else 0
                
                # Transform the result
                focus_result = {
                    'filename': img_name,
                    'focus_score': focus_score,
                    'gemini_status': batch_focus_result.get('status', 'ERROR'),
                    'confidence': batch_focus_result.get('confidence', 0),
                    'detail': batch_focus_result.get('reason', ''),
                    'status': 'in_focus' if (
                        focus_score >= focus_threshold and 
                        batch_focus_result.get('confidence', 0) >= in_focus_confidence and
                        batch_focus_result.get('status', '') == 'IN_FOCUS'
                    ) else 'off_focus'
                }
                
                # Map original paths
                original_image_path = img_path
                if raw_results and isinstance(raw_results.get('converted'), list):
                    for converted in raw_results['converted']:
                        if isinstance(converted, dict):
                            if Path(converted.get('converted_path', '')).stem == Path(img_path).stem:
                                original_image_path = converted.get('raw_path', img_path)
                                break
                
                # Process the result (same logic as sequential processing)
                _process_focus_result(
                    focus_result, img_path, original_image_path, 
                    results, directories, cache, processed_files,
                    focus_threshold, in_focus_confidence
                )
        
        # Cleanup batch processor cache
        batch_processor.cleanup_cache()
        
    else:
        # Original sequential processing
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
                
                # Use the helper function to process the result
                _process_focus_result(
                    focus_result, img_path, original_image_path,
                    results, directories, cache, processed_files,
                    focus_threshold, in_focus_confidence
                )

            except KeyError as e:
                results['stats']['processing_errors'] += 1
                # Log the specific key that's missing with full traceback
                import traceback
                logger.error(f"KeyError processing {img_name}: Missing key {str(e)}. Focus result keys: {list(focus_result.keys()) if 'focus_result' in locals() else 'N/A'}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                continue
            except Exception as e:
                results['stats']['processing_errors'] += 1
                logger.error(f"Error processing {img_name}: {type(e).__name__}: {str(e)}")
                continue

    # Save report at root level
    save_json_report(output_dir, results, config, 'focus_report.json')

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
