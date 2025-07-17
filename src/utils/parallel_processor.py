"""
Parallel Processing Utility for Independent Operations
Optimizes workflow by running non-dependent tasks concurrently
"""

import os
import time
import logging
from typing import Dict, List, Callable, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Manages parallel execution of independent processing tasks."""
    
    def __init__(self, max_workers: int = None):
        if max_workers is None:
            # Use half of CPU cores to leave room for other processes
            max_workers = max(1, multiprocessing.cpu_count() // 2)
        self.max_workers = max_workers
        self.results_lock = Lock()
        
    def run_parallel_tasks(self, tasks: Dict[str, Tuple[Callable, tuple, dict]]) -> Dict[str, Any]:
        """
        Run multiple independent tasks in parallel.
        
        Args:
            tasks: Dictionary of task_name -> (function, args, kwargs)
            
        Returns:
            Dictionary of task_name -> result
        """
        results = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task_name, (func, args, kwargs) in tasks.items():
                logger.info(f"Submitting task: {task_name}")
                future = executor.submit(func, *args, **kwargs)
                future_to_task[future] = task_name
                
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    with self.results_lock:
                        results[task_name] = result
                    logger.info(f"Completed task: {task_name}")
                except Exception as e:
                    logger.error(f"Error in task {task_name}: {str(e)}")
                    with self.results_lock:
                        results[task_name] = {'error': str(e)}
                        
        elapsed = time.time() - start_time
        logger.info(f"Parallel execution completed in {elapsed:.2f} seconds")
        
        return results
        
    def run_parallel_on_items(self, 
                            items: List[Any],
                            process_func: Callable,
                            max_concurrent: int = None) -> Dict[str, Any]:
        """
        Process multiple items in parallel using the same function.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            max_concurrent: Maximum concurrent executions
            
        Returns:
            Dictionary mapping item -> result
        """
        if max_concurrent is None:
            max_concurrent = self.max_workers
            
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_item = {
                executor.submit(process_func, item): item 
                for item in items
            }
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results[str(item)] = result
                except Exception as e:
                    logger.error(f"Error processing {item}: {str(e)}")
                    results[str(item)] = None
                    
        return results
        

class OptimizedWorkflow:
    """Orchestrates optimized parallel processing workflow."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.parallel_processor = ParallelProcessor()
        
    def process_quality_flow_parallel(self, 
                                    input_dir: str, 
                                    output_dir: str, 
                                    raw_results: Dict) -> Dict:
        """
        Optimized quality processing with parallel execution.
        
        Processing flow:
        1. Duplicates detection (independent)
        2. Blur + Focus detection (parallel)
        3. Eyes detection (uses face data from duplicates)
        4. Best quality (uses all previous results)
        """
        from src.services.duplicates import process_duplicates
        from src.services.blur import process_blur
        from src.services.focus import process_focus
        from src.services.closed_eyes import process_closed_eyes
        from src.services.best_quality import process_best_quality
        
        results = {}
        
        # Step 1: Process duplicates first (provides face data for other steps)
        logger.info("Step 1: Processing duplicates...")
        duplicate_results = process_duplicates(input_dir, output_dir, raw_results, self.config)
        results['duplicate_results'] = duplicate_results
        
        # Step 2: Run blur and focus detection in parallel
        logger.info("Step 2: Running blur and focus detection in parallel...")
        parallel_tasks = {
            'blur': (process_blur, (input_dir, output_dir, raw_results, self.config), {}),
            'focus': (process_focus, (input_dir, output_dir, raw_results, self.config), {})
        }
        
        parallel_results = self.parallel_processor.run_parallel_tasks(parallel_tasks)
        results['blur_results'] = parallel_results.get('blur', {})
        results['focus_results'] = parallel_results.get('focus', {})
        
        # Step 3: Process eyes (can use face data from duplicates)
        logger.info("Step 3: Processing closed eyes...")
        eyes_results = process_closed_eyes(input_dir, output_dir, raw_results, self.config)
        results['eyes_results'] = eyes_results
        
        # Step 4: Best quality assessment (uses all previous results)
        logger.info("Step 4: Processing best quality selection...")
        # Pass previous results to best quality processor
        enhanced_config = self.config.copy()
        enhanced_config['previous_results'] = {
            'duplicates': duplicate_results,
            'blur': results['blur_results'],
            'focus': results['focus_results'],
            'eyes': eyes_results
        }
        
        quality_results = process_best_quality(input_dir, output_dir, raw_results, enhanced_config)
        results['quality_results'] = quality_results
        
        return results
        
    def process_batch_parallel(self, batch_items: List[str], process_funcs: Dict[str, Callable]) -> Dict:
        """
        Process a batch of items with multiple functions in parallel.
        
        Args:
            batch_items: List of file paths
            process_funcs: Dictionary of analysis_name -> function
            
        Returns:
            Combined results from all analyses
        """
        batch_results = {}
        
        # Run different analyses in parallel
        parallel_tasks = {}
        for analysis_name, func in process_funcs.items():
            parallel_tasks[analysis_name] = (func, (batch_items,), {})
            
        analysis_results = self.parallel_processor.run_parallel_tasks(parallel_tasks)
        
        # Combine results by file
        for analysis_name, results in analysis_results.items():
            if isinstance(results, dict):
                for file_path, result in results.items():
                    if file_path not in batch_results:
                        batch_results[file_path] = {}
                    batch_results[file_path][analysis_name] = result
                    
        return batch_results