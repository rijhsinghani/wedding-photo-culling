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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import stat
import time
import sys
# Third-party imports
import numpy as np

from src.core.enhanced_raw_converter import EnhancedRawConverter
from src.core.image_preprocessor import ImagePreprocessor

from ..config import logger


#Functions of Operations - utils

def setup_directories(output_dir: str) -> Dict[str, str]:
    """Create necessary output directories."""
    directories = {
        'best_quality': os.path.join(output_dir, 'best_quality'),
        'poor_quality': os.path.join(output_dir, 'poor_quality'), 
        'closed_eyes': os.path.join(output_dir, 'closed_eyes'),
        'in_focus': os.path.join(output_dir, 'in_focus'),
        'duplicates': os.path.join(output_dir, 'duplicates'),
        'other_files': os.path.join(output_dir, 'other_files')  
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def save_json_report(folder_path: str, images_info: Dict, config: Dict):
    """Save processing results to a JSON file with proper type conversion."""
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def convert_dict(d):
        if not isinstance(d, dict):
            return convert_numpy_types(d)
        return {k: convert_dict(v) for k, v in d.items()}

    report = {
        'process_date': datetime.now().isoformat(),
        'configuration': config,
        'processed_images': images_info,
        'stats': {
            'total_processed': len(images_info.get('processed', [])),
            'venue_shots': len(images_info.get('other_files', [])),
        }
    }
    
    json_path = os.path.join(folder_path, 'report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4, default=convert_numpy_types)

def get_previously_processed_images(output_dir: str, category: str) -> Set[str]:
    """Get list of images already processed for a category from JSON report."""
    report_path = os.path.join(output_dir, category, 'report.json')
    if not os.path.exists(report_path):
        return set()
        
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
            processed = set()
            if 'processed_images' in report:
                results = report['processed_images']
                for key in ['blurry', 'clear', 'closed_eyes', 'open_eyes', 'in_focus', 'out_focus', 'best_quality']:
                    if key in results:
                        for item in results[key]:
                            if isinstance(item, dict):
                                file_path = item.get('original_path', item.get('path', ''))
                                if file_path:
                                    processed.add(os.path.basename(file_path))
            return processed
    except Exception:
        return set()

def get_all_image_files(input_dir: str, supported_formats: Dict) -> List[str]:
    """Recursively get all supported image files from input directory and its subdirectories."""
    image_files = []
    all_formats = set(ext.lower() for formats in supported_formats.values() 
                     for ext in formats)
    
    print("\n[cyan]Scanning directories for supported files...")
    for root, _, files in os.walk(input_dir):
        files_in_dir = []
        for file in files:
            if os.path.splitext(file.lower())[1] in all_formats:
                full_path = os.path.join(root, file)
                files_in_dir.append(full_path)
        
        if files_in_dir:
            rel_path = os.path.relpath(root, input_dir)
            print(f"[green]Found {len(files_in_dir)} files in: {rel_path}")
            image_files.extend(files_in_dir)
    
    print(f"[cyan]Total files found: {len(image_files)}")
    return image_files


def is_image_disqualified(img_name: str, previous_results: Dict) -> Tuple[bool, str]:
    """
    Check if an image should be disqualified based on previous results.
    Returns (is_disqualified: bool, reason: str)
    """
    # Check duplicate first
    duplicate_results = previous_results.get('duplicate_results', {}).get('duplicates', {})
    is_duplicate = False
    is_best_in_group = False
    
    for group_info in duplicate_results.values():
        best_image = group_info.get('best_image')
        if best_image and os.path.basename(str(best_image)) == img_name:
            is_best_in_group = True
        
        duplicates = group_info.get('duplicates', {})
        for dup_info in duplicates.values():
            if os.path.basename(str(dup_info.get('original_path', ''))) == img_name:
                is_duplicate = True

    if is_duplicate and not is_best_in_group:
        return True, "duplicate"

    # Check closed eyes
    eyes_results = previous_results.get('eyes_results', {}).get('closed_eyes', [])
    if any(img_name in str(img.get('original_path', '')) for img in eyes_results):
        return True, "closed_eyes"

    # Check blur
    blur_results = previous_results.get('blur_results', {}).get('blurry', [])
    if any(img_name in str(img.get('original_path', '')) for img in blur_results):
        return True, "blurry"

    # Check focus
    focus_results = previous_results.get('focus_results', {}).get('out_focus', [])
    if any(img_name in str(img.get('original_path', '')) for img in focus_results):
        return True, "focus"

    return False, ""

def get_quality_files_sets(output_dir: str, exclude_duplicates: bool = True) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str]]:
    """Get sets of files from previous quality checks."""
    closed_eyes_files = set()
    duplicate_files = set()
    best_images = set()
    blurry_files = set()
    focus_issues = set()
    
    # Get closed eyes files
    eyes_path = os.path.join(output_dir, 'closed_eyes', 'report.json')
    if os.path.exists(eyes_path):
        with open(eyes_path, 'r') as f:
            eyes_report = json.load(f)
            if isinstance(eyes_report.get('processed_images', {}).get('closed_eyes'), list):
                closed_eyes_files.update(
                    os.path.basename(str(img.get('original_path', ''))) 
                    for img in eyes_report['processed_images']['closed_eyes']
                )
                
    # Get duplicate files and best images
    if exclude_duplicates:
        dup_path = os.path.join(output_dir, 'duplicates', 'report.json')
        if os.path.exists(dup_path):
            with open(dup_path, 'r') as f:
                dup_report = json.load(f)
                if isinstance(dup_report.get('processed_images', {}).get('duplicates'), dict):
                    for group_info in dup_report['processed_images']['duplicates'].values():
                        if isinstance(group_info, dict):
                            best_image = group_info.get('best_image')
                            if best_image:
                                best_images.add(os.path.basename(best_image))
                            if isinstance(group_info.get('duplicates'), dict):
                                duplicate_files.update(
                                    os.path.basename(str(dup.get('original_path', ''))) 
                                    for dup in group_info['duplicates'].values()
                                )
                                
    # Get blurry files
    blur_path = os.path.join(output_dir, 'blurry', 'report.json')
    if os.path.exists(blur_path):
        with open(blur_path, 'r') as f:
            blur_report = json.load(f)
            if isinstance(blur_report.get('processed_images', {}).get('blurry'), list):
                blurry_files.update(
                    os.path.basename(str(img.get('original_path', ''))) 
                    for img in blur_report['processed_images']['blurry']
                )
                
    # Get focus issues
    focus_path = os.path.join(output_dir, 'in_focus', 'report.json')
    if os.path.exists(focus_path):
        with open(focus_path, 'r') as f:
            focus_report = json.load(f)
            if isinstance(focus_report.get('processed_images', {}).get('out_focus'), list):
                focus_issues.update(
                    os.path.basename(str(img.get('original_path', ''))) 
                    for img in focus_report['processed_images']['out_focus']
                )
                
    return closed_eyes_files, duplicate_files, best_images, blurry_files, focus_issues

def should_exclude_image(img_name: str, previous_results: Dict, allowed_duplicates: Set[str]) -> Tuple[bool, str]:
    """Enhanced exclusion logic with proper error handling"""
    try:
        basename = os.path.basename(img_name)
        name_without_ext = os.path.splitext(basename)[0]
        
        check_names = {
            basename,
            name_without_ext + '.png',
            name_without_ext + '.ARW'
        }
        
        # Enhanced validation
        if not isinstance(previous_results, dict):
            logger.warning("previous_results is not a dictionary")
            return False, ""
            
        if 'closed_eyes_files' in previous_results and any(name in previous_results['closed_eyes_files'] for name in check_names):
            return True, 'eyes'
            
        if 'blurry_files' in previous_results and any(name in previous_results['blurry_files'] for name in check_names):
            return True, 'blur'
            
        if 'duplicate_files' in previous_results and any(name in previous_results['duplicate_files'] for name in check_names):
            if not any(name in (allowed_duplicates or set()) for name in check_names):
                return True, 'duplicate'
                
        if 'out_of_focus_files' in previous_results and any(name in previous_results['out_of_focus_files'] for name in check_names):
            return True, 'focus'
            
        return False, ''
        
    except Exception as e:
        logger.error(f"Error in should_exclude_image: {str(e)}")
        return False, ''

def get_existing_results(output_dir: str) -> Dict:
    results = {
        'duplicate_files': set(),
        'best_duplicate_images': set(),
        'blurry_files': set(),
        'closed_eyes_files': set(),
        'out_of_focus_files': set()
    }
    
    # Get duplicate information first
    dup_path = os.path.join(output_dir, 'duplicates', 'report.json')
    if os.path.exists(dup_path):
        try:
            with open(dup_path, 'r') as f:
                dup_report = json.load(f)
                dup_data = dup_report.get('processed_images', {}).get('duplicates', {})
                for group_info in dup_data.values():
                    best_image = group_info.get('best_image')
                    if best_image:
                        results['best_duplicate_images'].add(best_image)
                    for filename, dup_info in group_info.get('duplicates', {}).items():
                        results['duplicate_files'].add(filename)
                        results['duplicate_files'].add(os.path.basename(dup_info.get('original_path', '')))
        except Exception as e:
            print(f"Error reading duplicates report: {str(e)}")

    # Get blurry results
    blur_path = os.path.join(output_dir, 'blurry', 'report.json')
    if os.path.exists(blur_path):
        try:
            with open(blur_path, 'r') as f:
                blur_report = json.load(f)
                for img in blur_report.get('processed_images', {}).get('blurry', []):
                    orig_path = img.get('original_path', '')
                    if orig_path:
                        results['blurry_files'].add(os.path.basename(orig_path))
                        # Also add the PNG version if it exists
                        png_name = os.path.splitext(os.path.basename(orig_path))[0] + '.png'
                        results['blurry_files'].add(png_name)
        except Exception as e:
            print(f"Error reading blur report: {str(e)}")

    # Get closed eyes results
    eyes_path = os.path.join(output_dir, 'closed_eyes', 'report.json')
    if os.path.exists(eyes_path):
        try:
            with open(eyes_path, 'r') as f:
                eyes_report = json.load(f)
                for img in eyes_report.get('processed_images', {}).get('closed_eyes', []):
                    orig_path = img.get('original_path', '')
                    if orig_path:
                        results['closed_eyes_files'].add(os.path.basename(orig_path))
                        # Also add the PNG version
                        png_name = os.path.splitext(os.path.basename(orig_path))[0] + '.png'
                        results['closed_eyes_files'].add(png_name)
        except Exception as e:
            print(f"Error reading closed eyes report: {str(e)}")

    # Get out of focus results
    focus_path = os.path.join(output_dir, 'in_focus', 'report.json')
    if os.path.exists(focus_path):
        try:
            with open(focus_path, 'r') as f:
                focus_report = json.load(f)
                for img in focus_report.get('processed_images', {}).get('out_focus', []):
                    orig_path = img.get('original_path', '')
                    if orig_path:
                        results['out_of_focus_files'].add(os.path.basename(orig_path))
                        # Also add the PNG version
                        png_name = os.path.splitext(os.path.basename(orig_path))[0] + '.png'
                        results['out_of_focus_files'].add(png_name)
        except Exception as e:
            print(f"Error reading focus report: {str(e)}")
            
    return results

def cleanup_temp_directory(temp_dir: str):
    """
    Safely remove temporary directory without terminating processes.
    """
    if not temp_dir or not os.path.exists(temp_dir):
        return

    max_retries = 3
    retry_delay = 1  # seconds

    def handle_error(func, path, exc_info):
        """Handle permission errors during deletion."""
        try:
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {e}")
            try:
                if os.path.exists(path):
                    new_name = f"{path}_old_{int(time.time())}"
                    os.rename(path, new_name)
            except:
                pass

    try:
        print("\n[yellow]Cleaning up temporary directory...")
        
        # First attempt: Try gentle cleanup
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    if os.path.exists(file_path):
                        os.chmod(file_path, stat.S_IWRITE)
                        os.unlink(file_path)
                except:
                    continue
                    
            for name in dirs:
                dir_path = os.path.join(root, name)
                try:
                    if os.path.exists(dir_path):
                        os.chmod(dir_path, stat.S_IWRITE)
                        os.rmdir(dir_path)
                except:
                    continue

        # Second attempt: Use shutil if files remain
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, onerror=handle_error)

        if not os.path.exists(temp_dir):
            print("[green]Successfully cleaned up temporary directory")
        else:
            # If directory still exists, rename it
            try:
                new_name = f"{temp_dir}_old_{int(time.time())}"
                os.rename(temp_dir, new_name)
                print(f"[yellow]Renamed temporary directory to: {new_name}")
            except Exception as e:
                logger.warning(f"Could not rename temporary directory: {e}")
                print("[yellow]Note: Temporary files will be cleaned up on next run")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        print("[yellow]Note: Temporary files will be cleaned up on next run")

def safe_file_path(file_path: str) -> bool:
    """Safely check if a file exists and is accessible."""
    try:
        return os.path.exists(file_path) and os.path.isfile(file_path)
    except (OSError, ValueError):
        return False



def detect_and_move_venue_shots(input_dir: str, output_dir: str, raw_results: Dict = None) -> Tuple[List[Dict], Set[str]]:
    venues_dir = os.path.join(output_dir, 'other_files')
    os.makedirs(venues_dir, exist_ok=True)
    
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    preprocessor = ImagePreprocessor()
    venue_shots = []
    excluded_files = set()
    temp_dirs = []
    processed_paths = set()

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.arw', '.cr2', '.nef')):
                image_path = os.path.join(root, file)
                
                if image_path not in processed_paths:
                    is_venue, confidence, reason = preprocessor.is_venue_shot(image_path)
                    print(f"Checking {file}: ", end='')
                    
                    if is_venue and confidence >= 90:
                        processed_paths.add(image_path)
                        print(f"VENUE (confidence: {confidence:.1f}%)")
                        
                        original_path = image_path
                        if raw_results:
                            for converted in raw_results.get('converted', []):
                                if Path(converted['converted_path']).stem == Path(image_path).stem:
                                    original_path = converted['raw_path']
                                    break

                        # Create proper relative path structure
                        rel_path = os.path.relpath(os.path.dirname(original_path), input_dir)
                        dest_dir = os.path.join(venues_dir, rel_path)
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        # Create unique destination path
                        dest_path = os.path.join(dest_dir, os.path.basename(original_path))
                        # Ensure we're not copying to same location
                        if os.path.abspath(original_path) != os.path.abspath(dest_path):
                            try:
                                shutil.copy2(str(original_path), dest_path)
                                venue_shots.append({
                                    'original_path': str(original_path),
                                    'new_path': dest_path,
                                    'confidence': float(confidence),
                                    'reason': reason
                                })
                                excluded_files.add(os.path.basename(file))
                                
                            except Exception as e:
                                logger.error(f"Error copying venue shot {original_path}: {str(e)}")
                                continue
                        else:
                            logger.warning(f"Skipping copy as source and destination are same: {original_path}")
                    else:
                        print("Contains people")

    if temp_dirs:
        for temp_dir in temp_dirs:
            cleanup_temp_directory(temp_dir)

    print(f"\nMoved {len(venue_shots)} venue/decoration shots to other_files")
    return venue_shots, excluded_files


