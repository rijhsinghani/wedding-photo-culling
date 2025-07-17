import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
import json
from typing import Dict
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import sys
# Third-party imports
from skimage.metrics import structural_similarity as ssim
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()


class ProcessingCache:
    def __init__(self, output_dir, max_cache_size: int = 1000):
        self.output_dir = output_dir
        self.max_cache_size = max_cache_size
        self.cache_cleanup_threshold = 0.9
        self.processed_files = {
            'duplicates': set(),
            'poor_quality': set(),  # Combined set
            'closed_eyes': set(),
            'in_focus': set(),
            'best_quality': set()
        }
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize cache from existing files and reports"""
        for category in self.processed_files.keys():
            category_dir = os.path.join(self.output_dir, category)
            if os.path.exists(category_dir):
                # Check report file
                json_path = os.path.join(category_dir, 'report.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            report = json.load(f)
                            if 'processed_images' in report:
                                results = report['processed_images']
                                self._add_from_results(results, category)
                    except Exception as e:
                        print(f"Error reading {category} report: {str(e)}")
                
                # Check directory contents
                for file in os.listdir(category_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.arw', '.cr2', '.nef')):
                        self.processed_files[category].add(file)

    def _add_from_results(self, results: Dict, category: str):
        """Add processed files from results"""
        for key in ['blurry', 'clear', 'closed_eyes', 'open_eyes', 'in_focus', 'out_focus', 'best_quality']:
            if key in results:
                for item in results[key]:
                    if isinstance(item, dict):
                        file_path = item.get('original_path', item.get('path', ''))
                        if file_path:
                            self.processed_files[category].add(os.path.basename(file_path))

    def should_process_image(self, img_path: str, category: str) -> bool:
        filename = os.path.basename(img_path)
        return filename not in self.processed_files[category]

    def add_processed_image(self, img_path: str, category: str):
        filename = os.path.basename(img_path)
        self.processed_files[category].add(filename)
