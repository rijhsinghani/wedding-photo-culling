"""
Multiprocessing helper for cross-platform compatibility.
Handles macOS-specific issues with multiprocessing.
"""

import os
import sys
import platform
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def get_optimal_workers():
    """Get optimal number of workers based on system."""
    cpu_count = mp.cpu_count()
    # On macOS, limit workers to avoid issues
    if platform.system() == 'Darwin':
        return min(cpu_count - 1, 4)
    return min(cpu_count - 1, 8)

def get_executor(use_threads=False, max_workers=None):
    """Get appropriate executor based on platform and requirements."""
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    if use_threads or platform.system() == 'Darwin':
        # Use threads on macOS to avoid fork issues
        return ThreadPoolExecutor(max_workers=max_workers)
    else:
        # Use processes on other platforms
        return ProcessPoolExecutor(max_workers=max_workers)

def init_multiprocessing():
    """Initialize multiprocessing with appropriate method."""
    if platform.system() == 'Darwin':
        # Force spawn method on macOS to avoid issues
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass
    
# Initialize on import
init_multiprocessing()