
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os

from typing import Dict,Tuple, Optional, Any
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import sys

# Third-party imports
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

from src.utils.resize_gemini import resize_for_gemini

from ..config import logger, log_critical

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI-based focus detection will be limited.")
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)



class FocusDetector:
    """Enhanced focus detector with Gemini integration for wedding photography."""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config else {}
        self.threshold = 35
        self.sample_regions = [(0.4, 0.6), (0.3, 0.7)]
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        self.gemini_prompt = """
        Analyze this wedding photo for focus quality:
        
        Focus Analysis Requirements:
        1. Main Subject Focus:
           - Main subject must be tack sharp
           - Facial features must be crystal clear
           - Fine details must be distinct
           - No motion blur or camera shake
           
        2. Overall Image Quality:
           - Professional-grade sharpness
           - Clear depth of field control
           - No technical focus issues
           
        3. Focus Priority:
           - Primary subject must be perfectly focused
           - Background focus is secondary
           
        Format Response EXACTLY as one of these:
        IN_FOCUS|confidence_0_to_100|detailed_reason
        OFF_FOCUS|confidence_0_to_100|detailed_reason

        IMPORTANT: Mark as IN_FOCUS if the image meets ALL of these criteria:
        - Main subjects are sharp and clear
        - Facial features are distinct
        - No motion blur or camera shake
        - Professional-grade sharpness
        - Good overall focus quality

        Example good response:
        IN_FOCUS|95|All subjects sharp and clear, faces distinct, excellent focus quality

        Example bad response:
        OFF_FOCUS|60|Main subject slightly soft, some motion blur visible
        """

    def analyze_with_gemini(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using Gemini for focus verification with enhanced logging."""
        try:
            resized_path = resize_for_gemini(image_path)
            if not resized_path:
                return {"status": "error", "confidence": 0, "reason": "Failed to process image"}

            from PIL import Image
            with Image.open(resized_path) as img:
                response = self.gemini_model.generate_content([
                    self.gemini_prompt,
                    img
                ])
            
            os.remove(resized_path)
            
            # Parse response with logging
            response_text = response.text.strip()
            logger.info(f"Gemini raw response: {response_text}")
            
            parts = response_text.split('|')
            if len(parts) != 3:
                logger.error(f"Invalid response format: {response_text}")
                return {"status": "error", "confidence": 0, "reason": "Invalid response format"}
                
            status, confidence, reason = parts
            
            # Convert confidence to float and normalize to 0-100
            try:
                confidence = float(confidence)
                if confidence <= 1.0:
                    confidence *= 100
            except ValueError:
                logger.error(f"Invalid confidence value: {confidence}")
                confidence = 0

            # Force status to uppercase for consistency
            status = status.strip().upper()
            
            logger.info(f"Processed Gemini response - Status: {status}, Confidence: {confidence}")
            
            return {
                "status": status,
                "confidence": confidence,
                "reason": reason.strip()
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {str(e)}")
            return {"status": "error", "confidence": 0, "reason": str(e)}

    def detect_faces(self, image: np.ndarray) -> bool:
        """Detect if image contains faces."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0

    def _analyze_region(self, gray: np.ndarray, region: Tuple[float, float]) -> Tuple[float, float]:
        """Analyze focus in a specific region of the image."""
        h, w = gray.shape
        start_x = int(w * region[0])
        end_x = int(w * region[1])
        start_y = int(h * region[0])
        end_y = int(h * region[1])
        
        roi = gray[start_y:end_y, start_x:end_x]
        
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_score = np.mean(edge_magnitude) * 0.8
        
        local_var = np.var(roi.astype(np.float32)) * 1.2
        
        return edge_score, local_var

    def calculate_focus_score(self, image: np.ndarray) -> float:
        """Calculate normalized focus score."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Laplacian variance for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score1 = laplacian.var()
        
        # FFT score for frequency analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1
        f_shift = f_shift * (1 - mask)
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        score2 = np.abs(img_back).mean()

        # Normalize and combine scores
        normalized_score = (score1 / 1000 * 0.6 + score2 / 100 * 0.4) * 100
        
        # Scale the score to a more reasonable range
        return min(100, max(50, normalized_score))

    def analyze_image(self, image_path: str) -> Dict:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Check for faces
            has_faces = self.detect_faces(image)
            if not has_faces:
                return {
                    'filename': os.path.basename(image_path),
                    'focus_score': 0,
                    'status': "no_subject",
                    'detail': "No faces detected in image"
                }

            # Calculate technical focus score
            focus_score = self.calculate_focus_score(image)
            
            # Get Gemini analysis
            gemini_result = self.analyze_with_gemini(image_path)
            
            # Add additional rejection criteria
            has_rejection_terms = any(term in gemini_result['reason'].lower() for term in [
                'back turned',
                'backs turned',
                'distracting',
                'unclear moment',
                'no clear subject',
                'poor composition',
                'obscures',  # Added for cases like RPV00016
                'foreground obstruction'  # Added for cases like RPV00016
            ])

            # Enhanced decision logic
            config_focus_threshold = self.config.get('thresholds', {}).get('focus_threshold', 50)
            config_confidence_threshold = self.config.get('thresholds', {}).get('in_focus_confidence', 85)
            
            result = {
                'filename': os.path.basename(image_path),
                'focus_score': float(focus_score),
                'gemini_status': gemini_result['status'],
                'confidence': float(gemini_result['confidence']),
                'detail': gemini_result['reason']
            }

            # Modified decision logic to be more strict
            if (focus_score >= config_focus_threshold and 
                gemini_result['confidence'] >= config_confidence_threshold and 
                gemini_result['status'] == 'IN_FOCUS' and 
                not has_rejection_terms and
                'foreground' not in gemini_result['reason'].lower()):  # Added check
                
                result['status'] = "in_focus"
            else:
                result['status'] = "off_focus"

            return result

        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return None