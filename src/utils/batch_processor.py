"""
Batch Processing Utility for Large Image Collections
Handles memory-efficient processing of images in configurable batches
"""

import os
import gc
import logging
from typing import List, Dict, Any, Callable
from pathlib import Path
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Manages batch processing of large image collections."""
    
    def __init__(self, batch_size: int = 25, clear_cache: bool = True):
        self.batch_size = batch_size
        self.clear_cache = clear_cache
        self.progress_file = "batch_progress.json"
        
    def save_progress(self, batch_num: int, processed_files: List[str], results: Dict):
        """Save processing progress for resume capability."""
        progress = {
            'last_batch': batch_num,
            'processed_files': processed_files,
            'partial_results': results,
            'timestamp': os.path.getmtime('.')
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
            
    def load_progress(self) -> Dict:
        """Load previous progress if available."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                logger.warning("Failed to load progress file")
        return {}
        
    def clear_progress(self):
        """Clear progress file after successful completion."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            
    def process_in_batches(self, 
                          items: List[Any], 
                          process_func: Callable,
                          merge_func: Callable = None,
                          description: str = "Processing") -> Dict:
        """
        Process items in batches with progress tracking.
        
        Args:
            items: List of items to process
            process_func: Function to process each batch
            merge_func: Function to merge batch results
            description: Progress bar description
            
        Returns:
            Merged results from all batches
        """
        total_items = len(items)
        num_batches = (total_items + self.batch_size - 1) // self.batch_size
        
        # Check for previous progress
        progress = self.load_progress()
        start_batch = progress.get('last_batch', 0)
        processed_files = set(progress.get('processed_files', []))
        accumulated_results = progress.get('partial_results', {})
        
        if start_batch > 0:
            logger.info(f"Resuming from batch {start_batch + 1}/{num_batches}")
            
        # Process each batch
        with tqdm(total=num_batches, initial=start_batch, desc=description) as pbar:
            for batch_num in range(start_batch, num_batches):
                # Get batch items
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_items)
                batch_items = items[start_idx:end_idx]
                
                # Skip already processed items
                batch_items = [item for item in batch_items 
                             if str(item) not in processed_files]
                
                if not batch_items:
                    pbar.update(1)
                    continue
                    
                logger.info(f"Processing batch {batch_num + 1}/{num_batches} "
                          f"({len(batch_items)} items)")
                
                try:
                    # Process batch
                    batch_results = process_func(batch_items)
                    
                    # Merge results
                    if merge_func:
                        accumulated_results = merge_func(accumulated_results, batch_results)
                    else:
                        # Default merge: update dictionary
                        if isinstance(batch_results, dict):
                            accumulated_results.update(batch_results)
                        else:
                            accumulated_results[f'batch_{batch_num}'] = batch_results
                    
                    # Update processed files
                    processed_files.update(str(item) for item in batch_items)
                    
                    # Save progress
                    self.save_progress(batch_num + 1, list(processed_files), 
                                     accumulated_results)
                    
                    # Clear memory if requested
                    if self.clear_cache:
                        gc.collect()
                        if hasattr(process_func, '__self__'):
                            # Clear any caches in the processor object
                            processor = process_func.__self__
                            if hasattr(processor, 'feature_cache'):
                                processor.feature_cache.clear()
                            if hasattr(processor, 'comparison_cache'):
                                processor.comparison_cache.clear()
                                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num + 1}: {str(e)}")
                    # Continue with next batch
                    
                pbar.update(1)
                
        # Clear progress file on successful completion
        self.clear_progress()
        
        return accumulated_results
        
    def split_by_directory(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Split files by directory for organized processing."""
        dir_groups = {}
        for path in file_paths:
            dir_name = os.path.dirname(path)
            if dir_name not in dir_groups:
                dir_groups[dir_name] = []
            dir_groups[dir_name].append(path)
        return dir_groups