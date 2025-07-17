import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from typing import Dict, List, Tuple, Optional, Any, Set
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
import shutil
from pathlib import Path
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import threading
from tqdm import tqdm
import multiprocessing
max_worker = min(multiprocessing.cpu_count(), 16)

# Third-party imports
from src.core.processing_cache import ProcessingCache
from src.core.duplicate_detector_optimized import OptimizedDuplicateDetector as DuplicateDetector
from src.utils.util import setup_directories, save_json_report, get_all_image_files

from ..config import logger, log_critical


# Assume the following are imported from elsewhere in your project:
# - get_all_image_files, setup_directories, save_json_report, logger
# - DuplicateDetector, ProcessingCache

def process_duplicates(input_dir: str, output_dir: str, raw_results: dict, config: dict, 
                       cache: 'ProcessingCache' = None,
                       exclude_files: set = None) -> dict:
    if cache is None:
        cache = ProcessingCache(output_dir)
    if exclude_files is None:
        exclude_files = set()
        
    directories = setup_directories(output_dir)
    threshold = config.get('thresholds', {}).get('duplicate_hash_threshold', 8)
    detector = DuplicateDetector(threshold=threshold)
    
    results = {
        'duplicates': {},
        'unique': [],
        'stats': {
            'total_processed': 0,
            'duplicate_groups': 0,
            'unique_images': 0,
            'processing_errors': 0,
            'skipped': 0,
            'excluded': 0
        }
    }
    
    try:
        # Get all image files excluding those in exclude_files
        image_files = [f for f in get_all_image_files(input_dir, config['supported_formats'])
                       if os.path.basename(f) not in exclude_files]
        
        if not image_files:
            return results
            
        print(f"\nAnalyzing images in: {input_dir}")
        print(f"\nProcessing {len(image_files)} images...")

        # Find duplicate groups using the optimized detector.
        duplicate_groups = detector.find_duplicates_optimized(input_dir, 0)
        if not duplicate_groups:
            return results

        # Process each duplicate group concurrently.
        results_lock = threading.Lock()

        def process_group(group_id, group_info):
            try:
                # Convert temporary file paths to original paths.
                best_image = path_mapping.get(group_info['best_image'])
                if not best_image:
                    return

                similar_images = []
                for similar in group_info.get('similar_images', []):
                    orig_path = path_mapping.get(similar)
                    if orig_path:
                        similar_images.append(orig_path)
                if not similar_images:
                    return

                # Create a directory for the duplicate group.
                group_dir = os.path.join(directories['duplicates'], group_id)
                os.makedirs(group_dir, exist_ok=True)

                # Resolve RAW file for the best image if available.
                original_best_image = best_image
                if raw_results and raw_results.get('converted'):
                    for converted in raw_results['converted']:
                        if Path(converted['converted_path']).stem == Path(best_image).stem:
                            original_best_image = converted['raw_path']
                            break

                # Copy the best image.
                best_dest = os.path.join(group_dir, os.path.basename(original_best_image))
                shutil.copy2(str(original_best_image), best_dest)

                # Process and copy similar images.
                duplicates_info = {}
                for similar in similar_images:
                    original_path = similar
                    if raw_results and raw_results.get('converted'):
                        for converted in raw_results['converted']:
                            if Path(converted['converted_path']).stem == Path(similar).stem:
                                original_path = converted['raw_path']
                                break

                    dest_path = os.path.join(group_dir, os.path.basename(original_path))
                    shutil.copy2(str(original_path), dest_path)
                    duplicates_info[os.path.basename(similar)] = {
                        'original_path': str(original_path),
                        'new_path': dest_path,
                        'quality_score': group_info['quality_scores'].get(os.path.basename(similar), 0)
                    }

                with results_lock:
                    results['duplicates'][group_id] = {
                        'best_image': os.path.basename(best_image),
                        'duplicates': duplicates_info,
                        'quality_scores': {os.path.basename(k): v 
                                           for k, v in group_info['quality_scores'].items()}
                    }
                    results['stats']['duplicate_groups'] += 1

            except Exception as e:
                logger.error(f"Error processing group {group_id}: {str(e)}")
                with results_lock:
                    results['stats']['processing_errors'] += 1

        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            group_futures = [executor.submit(process_group, group_id, group_info)
                             for group_id, group_info in duplicate_groups.items()]
            for _ in tqdm(as_completed(group_futures), total=len(group_futures), 
                          desc="Processing duplicate groups"):
                pass

        # Update overall statistics.
        all_processed_files = set()
        for group_info in results['duplicates'].values():
            all_processed_files.add(group_info['best_image'])
            all_processed_files.update(group_info['duplicates'].keys())
        results['stats']['total_processed'] = len(all_processed_files)
        results['stats']['unique_images'] = len(all_processed_files) - sum(
            len(group['duplicates']) for group in results['duplicates'].values()
        )

        # Save the JSON report.
        save_json_report(directories['duplicates'], results, config)
        return results

    except Exception as e:
        logger.error(f"Error in duplicate detection: {str(e)}")
        raise
    finally:
        # Clean up the temporary directory.
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {str(e)}")


