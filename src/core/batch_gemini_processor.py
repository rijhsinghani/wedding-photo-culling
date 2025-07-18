"""
Batch Gemini Processor for improved performance
"""
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading

import google.generativeai as genai
from PIL import Image

from src.utils.resize_gemini import resize_for_gemini
from src.utils.retry_handler import retry_gemini_api, RetryConfig
from src.core.api_queue_manager import get_default_queue_manager

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Represents a batch of images to process"""
    image_paths: List[str]
    prompt: str
    batch_id: str
    

@dataclass  
class BatchResult:
    """Results from batch processing"""
    results: Dict[str, Dict[str, Any]]
    total_time: float
    successful: int
    failed: int


class BatchGeminiProcessor:
    """
    Processes multiple images in batches for better performance
    """
    def __init__(
        self,
        model_name: str = 'gemini-1.5-pro',
        batch_size: int = 5,
        max_workers: int = 3,
        use_queue_manager: bool = True
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_queue_manager = use_queue_manager
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        # Queue manager for rate limiting
        self.queue_manager = get_default_queue_manager() if use_queue_manager else None
        
        # Cache for resized images
        self.resize_cache = {}
        self.cache_lock = threading.Lock()
        
        # Statistics
        self.total_processed = 0
        self.total_time = 0.0
    
    def _resize_and_cache(self, image_path: str) -> Optional[str]:
        """Resize image and cache the result"""
        with self.cache_lock:
            if image_path in self.resize_cache:
                return self.resize_cache[image_path]
        
        resized_path = resize_for_gemini(image_path)
        
        if resized_path:
            with self.cache_lock:
                self.resize_cache[image_path] = resized_path
        
        return resized_path
    
    @retry_gemini_api
    def _process_single_image(
        self,
        image_path: str,
        prompt: str
    ) -> Dict[str, Any]:
        """Process a single image with Gemini"""
        try:
            # Resize image
            resized_path = self._resize_and_cache(image_path)
            if not resized_path:
                return {
                    'status': 'ERROR',
                    'confidence': 0,
                    'reason': 'Failed to resize image'
                }
            
            # Load image
            with Image.open(resized_path) as img:
                # Make API request
                if self.queue_manager:
                    # Use queue manager for rate limiting
                    with self.queue_manager.request_context(
                        self.model.generate_content,
                        [prompt, img]
                    ) as response:
                        if response.success:
                            api_response = response.result
                        else:
                            raise response.error
                else:
                    # Direct API call
                    api_response = self.model.generate_content([prompt, img])
            
            # Parse response
            if not api_response or not api_response.text:
                return {
                    'status': 'ERROR',
                    'confidence': 0,
                    'reason': 'Empty API response'
                }
            
            # Parse the response (assuming same format as focus detector)
            response_text = api_response.text.strip()
            parts = response_text.split('|')
            
            if len(parts) != 3:
                return {
                    'status': 'ERROR',
                    'confidence': 0,
                    'reason': f'Invalid response format: {response_text}'
                }
            
            status, confidence, reason = parts
            
            # Convert confidence
            try:
                confidence = float(confidence)
                if confidence <= 1.0:
                    confidence *= 100
            except ValueError:
                confidence = 0
            
            return {
                'status': status.strip().upper(),
                'confidence': confidence,
                'reason': reason.strip(),
                'filename': os.path.basename(image_path)
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'status': 'ERROR',
                'confidence': 0,
                'reason': f'Processing failed: {str(e)}',
                'filename': os.path.basename(image_path)
            }
    
    def process_batch(
        self,
        image_paths: List[str],
        prompt: str,
        parallel: bool = True
    ) -> BatchResult:
        """
        Process a batch of images
        
        Args:
            image_paths: List of image file paths
            prompt: The prompt to use for all images
            parallel: Whether to process in parallel
            
        Returns:
            BatchResult with all results
        """
        start_time = time.time()
        results = {}
        successful = 0
        failed = 0
        
        if parallel and len(image_paths) > 1:
            # Process in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(
                        self._process_single_image,
                        path,
                        prompt
                    ): path
                    for path in image_paths
                }
                
                # Collect results
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results[path] = result
                        
                        if result['status'] != 'ERROR':
                            successful += 1
                        else:
                            failed += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to process {path}: {e}")
                        results[path] = {
                            'status': 'ERROR',
                            'confidence': 0,
                            'reason': f'Exception: {str(e)}'
                        }
                        failed += 1
        else:
            # Process sequentially
            for path in image_paths:
                result = self._process_single_image(path, prompt)
                results[path] = result
                
                if result['status'] != 'ERROR':
                    successful += 1
                else:
                    failed += 1
        
        total_time = time.time() - start_time
        
        # Update statistics
        self.total_processed += len(image_paths)
        self.total_time += total_time
        
        logger.info(
            f"Batch processed: {successful} successful, {failed} failed "
            f"in {total_time:.2f}s ({len(image_paths)/total_time:.2f} images/sec)"
        )
        
        return BatchResult(
            results=results,
            total_time=total_time,
            successful=successful,
            failed=failed
        )
    
    def process_directory(
        self,
        directory: str,
        prompt: str,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
    ) -> BatchResult:
        """
        Process all images in a directory in batches
        
        Args:
            directory: Directory containing images
            prompt: The prompt to use
            extensions: File extensions to process
            
        Returns:
            Combined BatchResult
        """
        # Find all image files
        image_paths = []
        for file in os.listdir(directory):
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(directory, file))
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process in batches
        all_results = {}
        total_successful = 0
        total_failed = 0
        start_time = time.time()
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            logger.info(
                f"Processing batch {i//self.batch_size + 1}/"
                f"{(len(image_paths) + self.batch_size - 1)//self.batch_size}"
            )
            
            batch_result = self.process_batch(batch_paths, prompt)
            
            # Combine results
            all_results.update(batch_result.results)
            total_successful += batch_result.successful
            total_failed += batch_result.failed
        
        total_time = time.time() - start_time
        
        return BatchResult(
            results=all_results,
            total_time=total_time,
            successful=total_successful,
            failed=total_failed
        )
    
    def cleanup_cache(self):
        """Clean up cached resized images"""
        with self.cache_lock:
            for resized_path in self.resize_cache.values():
                try:
                    if os.path.exists(resized_path):
                        os.remove(resized_path)
                except Exception as e:
                    logger.warning(f"Failed to remove {resized_path}: {e}")
            
            self.resize_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_time = (
            self.total_time / self.total_processed
            if self.total_processed > 0
            else 0
        )
        
        return {
            'total_processed': self.total_processed,
            'total_time': self.total_time,
            'average_time_per_image': avg_time,
            'images_per_second': 1 / avg_time if avg_time > 0 else 0,
            'cache_size': len(self.resize_cache)
        }