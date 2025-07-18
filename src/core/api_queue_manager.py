"""
API Queue Manager for rate limiting and request management
"""
import time
import queue
import threading
import logging
from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class APIRequest:
    """Represents a queued API request"""
    id: str
    func: Callable
    args: Tuple
    kwargs: Dict
    timestamp: datetime
    priority: int = 0
    
    def __lt__(self, other):
        """Enable priority queue sorting (higher priority first)"""
        return self.priority > other.priority


@dataclass
class APIResponse:
    """Represents an API response"""
    request_id: str
    result: Any
    success: bool
    error: Optional[Exception] = None
    duration: float = 0.0


class APIQueueManager:
    """
    Manages API requests with rate limiting and concurrency control
    """
    def __init__(
        self,
        max_concurrent_requests: int = 5,
        requests_per_minute: int = 60,
        queue_size: int = 1000
    ):
        self.max_concurrent = max_concurrent_requests
        self.rpm_limit = requests_per_minute
        self.queue_size = queue_size
        
        # Request tracking
        self.request_queue = queue.PriorityQueue(maxsize=queue_size)
        self.response_futures = {}
        self.active_requests = 0
        self.active_lock = threading.Lock()
        
        # Rate limiting
        self.request_times = []
        self.rate_limit_lock = threading.Lock()
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_duration = 0.0
        
        # Worker management
        self.workers = []
        self.shutdown_event = threading.Event()
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads to process the queue"""
        for i in range(self.max_concurrent):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"APIWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Worker thread main loop"""
        while not self.shutdown_event.is_set():
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=1.0)
                
                # Check rate limit
                self._wait_for_rate_limit()
                
                # Process the request
                self._process_request(request)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits"""
        with self.rate_limit_lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [
                t for t in self.request_times 
                if now - t < 60
            ]
            
            # Check if we're at the limit
            if len(self.request_times) >= self.rpm_limit:
                # Calculate how long to wait
                oldest_request = self.request_times[0]
                wait_time = 60 - (now - oldest_request) + 0.1
                
                if wait_time > 0:
                    logger.info(
                        f"Rate limit reached. Waiting {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
            
            # Record this request
            self.request_times.append(now)
    
    def _process_request(self, request: APIRequest):
        """Process a single API request"""
        start_time = time.time()
        response = None
        
        with self.active_lock:
            self.active_requests += 1
            self.total_requests += 1
        
        try:
            # Execute the request
            result = request.func(*request.args, **request.kwargs)
            
            response = APIResponse(
                request_id=request.id,
                result=result,
                success=True,
                duration=time.time() - start_time
            )
            
            with self.active_lock:
                self.successful_requests += 1
                self.total_duration += response.duration
            
            logger.debug(
                f"Request {request.id} completed in {response.duration:.2f}s"
            )
            
        except Exception as e:
            response = APIResponse(
                request_id=request.id,
                result=None,
                success=False,
                error=e,
                duration=time.time() - start_time
            )
            
            with self.active_lock:
                self.failed_requests += 1
            
            logger.error(
                f"Request {request.id} failed: {type(e).__name__}: {str(e)}"
            )
        
        finally:
            with self.active_lock:
                self.active_requests -= 1
            
            # Store response
            if request.id in self.response_futures:
                self.response_futures[request.id].put(response)
    
    def submit_request(
        self,
        func: Callable,
        *args,
        request_id: Optional[str] = None,
        priority: int = 0,
        **kwargs
    ) -> str:
        """
        Submit a request to the queue
        
        Args:
            func: The function to call
            args: Positional arguments for the function
            request_id: Optional request ID (generated if not provided)
            priority: Request priority (higher = processed sooner)
            kwargs: Keyword arguments for the function
            
        Returns:
            Request ID
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        request = APIRequest(
            id=request_id,
            func=func,
            args=args,
            kwargs=kwargs,
            timestamp=datetime.now(),
            priority=priority
        )
        
        # Create future for response
        from queue import Queue
        response_queue = Queue(maxsize=1)
        self.response_futures[request_id] = response_queue
        
        # Add to queue
        self.request_queue.put(request)
        
        logger.debug(f"Submitted request {request_id} with priority {priority}")
        
        return request_id
    
    def get_response(
        self,
        request_id: str,
        timeout: Optional[float] = None
    ) -> Optional[APIResponse]:
        """
        Get the response for a request
        
        Args:
            request_id: The request ID
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            APIResponse object or None if timeout
        """
        if request_id not in self.response_futures:
            raise ValueError(f"Unknown request ID: {request_id}")
        
        response_queue = self.response_futures[request_id]
        
        try:
            response = response_queue.get(timeout=timeout)
            del self.response_futures[request_id]
            return response
        except:
            # Timeout occurred
            return None
    
    @contextmanager
    def request_context(self, func: Callable, *args, **kwargs):
        """
        Context manager for making a request and getting response
        
        Example:
            with queue_manager.request_context(api_func, arg1, arg2) as response:
                if response.success:
                    print(response.result)
        """
        request_id = self.submit_request(func, *args, **kwargs)
        try:
            response = self.get_response(request_id, timeout=300)  # 5 min timeout
            yield response
        finally:
            # Cleanup if needed
            if request_id in self.response_futures:
                del self.response_futures[request_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.active_lock:
            avg_duration = (
                self.total_duration / self.successful_requests
                if self.successful_requests > 0
                else 0
            )
            
            return {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'active_requests': self.active_requests,
                'queued_requests': self.request_queue.qsize(),
                'average_duration': avg_duration,
                'requests_per_minute': len(self.request_times)
            }
    
    def shutdown(self, timeout: float = 30.0):
        """Shutdown the queue manager"""
        logger.info("Shutting down API queue manager...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout / len(self.workers))
        
        # Log final stats
        stats = self.get_stats()
        logger.info(f"Final queue statistics: {stats}")


# Global instance for convenience
_default_queue_manager = None


def get_default_queue_manager() -> APIQueueManager:
    """Get or create the default queue manager"""
    global _default_queue_manager
    if _default_queue_manager is None:
        _default_queue_manager = APIQueueManager()
    return _default_queue_manager