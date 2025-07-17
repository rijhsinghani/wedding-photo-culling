import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
from typing import Tuple

import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)


# Third-party imports
import numpy as np
import cv2
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

from src.utils.resize_gemini import resize_for_gemini

from ..config import logger, log_critical
# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI-based preprocessing will be limited.")
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)





class ImagePreprocessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.yolo_model = YOLO('yolov8n.pt')  # Load small YOLO model
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.venue_prompt = """
        Analyze this wedding photo to strictly determine if it only shows venue/decoration:

        Classify as VENUE if ALL of these are true:
        - Only shows decorations, tables, chairs, flowers
        - Only venue setup and details
        - Only decor elements visible
        - No human presence at all

        Classify as NOT_VENUE if ANY of these are true:
        - Contains any people or human elements
        - Shows any human activity
        - Contains any human objects/items

        Format Response EXACTLY as:
        VENUE|95|No people or human elements, only decor and venue details
        or 
        NOT_VENUE|90|Contains human elements or activity
        """

    def detect_people(self, image_path: str) -> bool:
        """Use YOLOv8 to detect people in image"""
        try:
            results = self.yolo_model(image_path)
            
            # Check if any detection is a person
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.cls == 0:  # Class 0 is person in YOLO
                        return True
            
            return False
                    
        except Exception as e:
            logger.error(f"Error in people detection: {e}")
            return False

    def _detect_decor_elements(self, image: np.ndarray) -> bool:
        """Detect venue/decor specific elements."""
        try:
            # Convert to HSV color space for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for common decor elements (white/cream colors)
            white_lower = np.array([0, 0, 180])
            white_upper = np.array([180, 30, 255])
            
            # Create masks
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            
            # Calculate white/cream percentage
            white_percent = (cv2.countNonZero(white_mask) / 
                           (image.shape[0] * image.shape[1])) * 100
            
            # Check for decorative patterns using edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_percent = (cv2.countNonZero(edges) / 
                          (image.shape[0] * image.shape[1])) * 100
            
            # Venue shots typically have:
            # - High percentage of white/cream colors (decor)
            # - Moderate amount of edges (patterns/decorations)
            return white_percent > 25 and 5 < edge_percent < 35
                    
        except Exception as e:
            logger.error(f"Error in decor detection: {e}")
            return False

    def is_venue_shot(self, image_path: str) -> Tuple[bool, float, str]:
        """Determine if image is a venue shot without people."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, 0, "Failed to read image"

            # First check for people using YOLO
            has_people = self.detect_people(image_path)
            if has_people:
                return False, 100.0, "Contains people"

            # Check for decor elements
            is_decor = self._detect_decor_elements(image)
            if is_decor:
                return True, 100.0, "Venue/decor shot detected"

            # If still unsure, verify with Gemini
            resized = resize_for_gemini(image_path)
            if not resized:
                return False, 0, "Failed to process image"

            from PIL import Image
            with Image.open(resized) as img:
                response = self.model.generate_content([
                    self.venue_prompt,
                    img
                ])
            
            os.remove(resized)
            
            # Parse response
            classification, confidence, reason = response.text.strip().split('|')
            confidence = float(confidence)
            
            is_venue = (classification == 'VENUE' and confidence >= 85)
            return is_venue, confidence, reason

        except Exception as e:
            logger.error(f"Error in venue detection: {e}")
            return False, 0, str(e)
