
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
import logging
import signal

# Third-party imports
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

from src.utils.resize_gemini import resize_for_gemini
from src.utils.retry_handler import retry_gemini_api, RetryConfig, retry_on_exception

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
        
        # Add simple caching for repeated analysis
        self._analysis_cache = {}
        
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

    @retry_gemini_api
    def _make_gemini_request(self, prompt: str, img) -> Any:
        """Make the actual Gemini API request with retry logic"""
        return self.gemini_model.generate_content([prompt, img])
    
    def analyze_with_gemini(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using Gemini for focus verification with timeout handling and retry."""
        import time
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Gemini API timeout")
        
        try:
            resized_path = resize_for_gemini(image_path)
            if not resized_path:
                return {"status": "ERROR", "confidence": 0, "reason": "Failed to process image"}

            from PIL import Image
            
            # Set timeout for API call (10 seconds max)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            
            start_time = time.time()
            with Image.open(resized_path) as img:
                # Use the retry-wrapped method
                response = self._make_gemini_request(self.gemini_prompt, img)
            
            # Clear the alarm
            signal.alarm(0)
            api_time = time.time() - start_time
            
            if api_time > 5:  # Log slow API calls
                logger.warning(f"Slow Gemini API call: {api_time:.2f}s for {os.path.basename(image_path)}")
            
            os.remove(resized_path)
            
            # Parse response with logging
            if not response or not response.text:
                logger.error("Empty response from Gemini API")
                return {"status": "ERROR", "confidence": 0, "reason": "Empty API response"}
                
            response_text = response.text.strip()
            # Only log responses in debug mode to reduce noise
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Gemini raw response: {response_text}")
            
            parts = response_text.split('|')
            if len(parts) != 3:
                logger.error(f"Invalid response format: {response_text}")
                return {"status": "ERROR", "confidence": 0, "reason": "Invalid response format"}
                
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
            
        except TimeoutError:
            logger.error(f"Gemini API timeout for {image_path}")
            return {
                "status": "ERROR", 
                "confidence": 0, 
                "reason": "Gemini API timeout - image processing too slow"
            }
        except Exception as e:
            # Clear any pending alarm
            try:
                signal.alarm(0)
            except:
                pass
            logger.error(f"Error in Gemini analysis for {image_path}: {str(e)}")
            return {
                "status": "ERROR", 
                "confidence": 0, 
                "reason": f"Gemini analysis failed: {str(e)}"
            }
        finally:
            # Ensure cleanup
            try:
                if 'resized_path' in locals() and os.path.exists(resized_path):
                    os.remove(resized_path)
            except:
                pass

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
            # Check cache first
            cache_key = os.path.basename(image_path)
            if cache_key in self._analysis_cache:
                logger.debug(f"Using cached result for {cache_key}")
                return self._analysis_cache[cache_key]
            
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
            
            # Check if gemini_result is valid
            if not gemini_result or not isinstance(gemini_result, dict):
                logger.error(f"Invalid gemini_result: {gemini_result}")
                return None
            
            # Add additional rejection criteria
            reason = gemini_result.get('reason', '')
            has_rejection_terms = any(term in reason.lower() for term in [
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
                'gemini_status': gemini_result.get('status', 'ERROR'),
                'confidence': float(gemini_result.get('confidence', 0)),
                'detail': reason
            }

            # Modified decision logic to be more strict
            if (focus_score >= config_focus_threshold and 
                gemini_result.get('confidence', 0) >= config_confidence_threshold and 
                gemini_result.get('status', '') == 'IN_FOCUS' and 
                not has_rejection_terms and
                'foreground' not in reason.lower()):  # Added check
                
                result['status'] = "in_focus"
            else:
                result['status'] = "off_focus"

            # Cache the result
            self._analysis_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            # Return a valid error result instead of None
            error_result = {
                'filename': os.path.basename(image_path),
                'focus_score': 0.0,
                'gemini_status': 'ERROR',
                'confidence': 0.0,
                'detail': f'Error during analysis: {str(e)}',
                'status': 'error'
            }
            # Cache the error result to avoid repeated failures
            cache_key = os.path.basename(image_path)
            self._analysis_cache[cache_key] = error_result
            return error_result