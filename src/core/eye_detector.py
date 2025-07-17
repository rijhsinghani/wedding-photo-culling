
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
from typing import Dict, Tuple, Any
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import logging
import sys

# Third-party imports
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification
from skimage.metrics import structural_similarity as ssim
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()


from src.utils.resize_gemini import resize_for_gemini
from src.core.image_preprocessor import ImagePreprocessor

from ..config import logger, log_critical

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI-based eye detection will be limited.")
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)




class EyeDetector:
    """Enhanced eye detection with Gemini integration."""
    def __init__(self):
        # Initialize existing models
        self.model_path = "MichalMlodawski/open-closed-eye-classification-mobilev2"
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = MobileNetV2ForImageClassification.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Enhanced prompt for Gemini
        self.gemini_prompt = """
        Analyze this facial image for eye status with extreme precision:

        Critical Requirements:
        1. Main Subject Focus:
           - Only analyze eyes of main subject(s) in focus
           - Ignore background people
           
        2. Eye Status Analysis:
           - Account for eye makeup (kajal, eyeliner, etc)
           - Distinguish between:
             * Actually closed eyes
             * Heavy eye makeup
             * Cultural eye appearances
             * Natural eye shapes
           
        3. Group Photo Handling:
           - Identify main subjects
           - Only analyze focused subjects
           - Ignore background people
           
        Format Response as:
        CLOSED|confidence|reason (only if 100% certain main subject's eyes are closed)
        or
        OPEN|confidence|reason (if main subject's eyes are open)
        or
        UNCLEAR|0|reason (if cannot determine)

        Only mark as CLOSED if absolutely certain the main subject's eyes are fully closed.
        Be extra careful with:
        - Traditional eye makeup
        - Cultural eye appearances
        - Group photos with multiple people
        """

    def analyze_with_gemini(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using Gemini for eye status verification."""
        try:
            # Resize image for Gemini
            resized_path = resize_for_gemini(image_path)
            if not resized_path:
                return {"status": "error", "confidence": 0, "reason": "Failed to process image"}

            # Get Gemini's analysis
            with Image.open(resized_path) as img:
                response = self.gemini_model.generate_content([
                    self.gemini_prompt,
                    img
                ])
            
            # Clean up resized image
            os.remove(resized_path)
            
            # Parse response
            status, confidence, reason = response.text.strip().split('|')
            return {
                "status": status,
                "confidence": float(confidence),
                "reason": reason
            }
            
        except Exception as e:
            return {"status": "error", "confidence": 0, "reason": str(e)}

    def detect_faces(self, image: np.ndarray) -> bool:
        """Detect if image contains any faces."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.22, minNeighbors=5)
        return len(faces) > 0

    def detect_closed_eyes(self, image_path: str) -> Tuple[bool, float, str]:
        try:
            # First check if it's a venue shot
            preprocessor = ImagePreprocessor()
            is_venue, confidence, reason = preprocessor.is_venue_shot(image_path)
            
            if is_venue and confidence >= 75:
                return False, 0, "Venue shot - no face detection needed"
                
            # Rest of existing eye detection logic
            image = cv2.imread(image_path)
            if image is None:
                return False, 0, "Failed to read image"
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.22, minNeighbors=5)
            
            if len(faces) == 0:
                return False, 0, "No faces detected"

            # Calculate image area for context detection
            image_height, image_width = gray.shape
            image_area = image_width * image_height
            
            # Sort faces by size and filter out very small faces
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
            # Check if this is a portrait/group photo vs venue/decor shot
            largest_face_area = faces[0][2] * faces[0][3]
            face_to_image_ratio = largest_face_area / image_area
            
            # Skip eye detection for:
            # - Faces smaller than 5% of image (background people)
            # - Wide/establishing shots
            if face_to_image_ratio < 0.05:
                return False, 0, "Face too small - likely venue/decor shot"
                
            # For group photos, check if multiple significant faces
            significant_faces = [f for f in faces if (f[2] * f[3]) / image_area > 0.03]
            if len(significant_faces) == 0:
                return False, 0, "No significant faces - skip eye detection"
            
            main_face = faces[0]
            
            # Get ML model prediction first
            x, y, w, h = main_face
            face_roi = image[y:y+h, x:x+w]
            face_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            inputs = self.image_processor(images=face_img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                closed_prob = float(probs[0][0].item() * 100)  # Convert to float explicitly

            # Get Gemini verification
            gemini_result = self.analyze_with_gemini(image_path)
            
            # Decision logic with explicit float comparisons
            if closed_prob >= 85.0:  # ML model strongly indicates closed eyes
                if gemini_result["status"] == "CLOSED":
                    return True, max(closed_prob, float(gemini_result["confidence"])), "Eyes closed (confirmed by both ML and Gemini)"

                else:
                    return True, closed_prob, "Eyes closed (detected by ML model)"
                    
            return False, float(100 - closed_prob), "Eyes open"

        except Exception as e:
            logger.error(f"Error in eye detection: {e}")
            return False, 0, str(e)
