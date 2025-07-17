
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
from typing import Dict, Optional
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import sys
# Third-party imports
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()


from ..config import logger, log_critical



class QualityAssessor:
    def __init__(self):
        self.quality_weights = {
            'resolution': 0.15,
            'face_detection': 0.20,
            'contrast': 0.15,
            'sharpness': 0.20,
            'noise': 0.15,
            'exposure': 0.15
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def assess_image_quality(self, image_path: str) -> Optional[Dict]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            exposure_score = 1.0 - abs(127.5 - mean_brightness) / 127.5

            # Enhanced quality assessment
            quality_scores = {
                'resolution': self._get_resolution_score(image),
                'face_detection': self._get_face_score(gray),
                'contrast': self._get_contrast_score(gray),
                'sharpness': self._get_sharpness_score(gray),
                'noise': self._get_noise_score(gray),
                'exposure': exposure_score
            }

            # Apply quality boosts
            quality_scores = {k: min(v * 1.15, 1.0) for k, v in quality_scores.items()}

            overall_score = sum(score * self.quality_weights[metric] 
                              for metric, score in quality_scores.items())

            return {
                'quality_scores': quality_scores,
                'overall_quality_score': min(overall_score * 1.1, 1.0),
                'exposure_score': exposure_score
            }

        except Exception as e:
            logger.error(f"Error in quality assessment: {str(e)}")
            return None

    def _get_resolution_score(self, image: np.ndarray) -> float:
        height, width = image.shape[:2]
        pixels = height * width
        target_pixels = 2000 * 3000
        return min(pixels / target_pixels, 1.0)

    def _get_face_score(self, gray: np.ndarray) -> float:
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return min(len(faces), 3) / 3 if len(faces) > 0 else 0.5

    def _get_contrast_score(self, gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist.flatten() / hist.sum()
        contrast = np.sqrt(((np.arange(256) - np.sum(hist * np.arange(256))) ** 2) * hist).sum()
        return min(contrast / 80.0, 1.0)

    def _get_sharpness_score(self, gray: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        lap_score = laplacian.var() / 500.0
        edge_score = np.sqrt(sobelx**2 + sobely**2).mean() / 100.0
        
        return min((lap_score * 0.6 + edge_score * 0.4), 1.0)

    def _get_noise_score(self, gray: np.ndarray) -> float:
        noise_sigma = estimate_sigma(gray, channel_axis=None)
        return max(1.0 - (noise_sigma / 10.0), 0.0)
