import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
from typing import Dict,Any
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import logging
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
    logger.warning("GEMINI_API_KEY not found in environment variables. AI-based blur detection will be limited.")

GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel(GEMINI_MODEL)


class BlurDetector:
    def __init__(self, threshold: float = 25.0):
        self.threshold = threshold

        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        self.gemini_prompt = """
        Analyze this image for clarity and visibility with extreme precision:

        MUST MARK AS BLURRY if ANY of these issues exist:
        - Motion blur or camera shake
        - Out of focus or soft focus
        - Severe overexposure making details unclear
        - Severe underexposure making subjects hard to see
        - Poor visibility due to technical issues
        - Lack of detail clarity due to exposure problems
        - Blown out highlights obscuring features
        
        Technical Evaluation:
        1. Overall Clarity:
           - Check for motion blur
           - Check focus sharpness
           - Evaluate exposure impact on visibility
           
        2. Detail Analysis:
           - Verify if facial features are clearly visible
           - Check if exposure is obscuring important details
           - Assess if highlights or shadows are destroying detail
           
        Format Response as:
        BLURRY|confidence|detailed_reason
        or
        CLEAR|confidence|detailed_reason

        Mark as BLURRY if:
        - Any critical details are lost due to exposure
        - Faces or subjects are unclear due to lighting
        - Technical issues impact visibility
        - Exposure problems hide important features
        """

    def variance_of_laplacian(self, image: np.ndarray) -> float:
        """Calculate Laplacian variance for blur detection."""
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def fft_blur_detector(self, image: np.ndarray, size: int = 60, thresh: int = 10) -> float:
        """Detect blur using FFT analysis."""
        (h, w) = image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))

        fft = np.fft.fft2(image)
        fftShift = np.fft.fftshift(fft)

        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)

        magnitude = 20 * np.log(np.abs(recon))
        return np.mean(magnitude)

    def analyze_with_gemini(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using Gemini for blur verification."""
        try:
            # Resize image for Gemini
            resized_path = resize_for_gemini(image_path)
            if not resized_path:
                return {"status": "error", "confidence": 0, "reason": "Failed to process image"}

            # Get Gemini's analysis
            response = self.gemini_model.generate_content([
                self.gemini_prompt,
                genai.upload_file(resized_path)
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

    def detect_blur(self, image_path: str) -> Dict:
        """Enhanced blur detection combining traditional methods with Gemini analysis."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return {"error": f"Failed to read image: {image_path}"}

            # Traditional blur detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lap_var = self.variance_of_laplacian(gray)
            fft_score = self.fft_blur_detector(gray)

            # Normalize scores
            lap_score = min(lap_var / 1000, 1) * 100
            fft_score = min(fft_score / 20, 1) * 100
            
            combined_score = (lap_score * 0.5 + fft_score * 0.5)

            # Get Gemini's analysis
            gemini_result = self.analyze_with_gemini(image_path)
            
            # Initialize result dictionary
            result = {
                "laplacian_score": float(lap_score),
                "fft_score": float(fft_score),
                "combined_score": float(combined_score),
                "is_blurry": False,
                "confidence": 0,
                "reason": ""
            }

            # Decision logic combining traditional methods and Gemini
            if gemini_result["status"] != "error":
                if gemini_result["status"] == "BLURRY" and gemini_result["confidence"] >= 90:
                    if combined_score < 40:  # Traditional methods also indicate blur
                        result["is_blurry"] = True
                        result["confidence"] = gemini_result["confidence"]
                        result["reason"] = gemini_result["reason"]
                        result["unclear"] = gemini_result["status"] == "UNCLEAR"
                        return result
                        
                    else:
                        # Additional verification for borderline cases
                        result["is_blurry"] = combined_score < 30
                        result["confidence"] = min(combined_score, gemini_result["confidence"])
                        result["reason"] = "Borderline case: " + gemini_result["reason"]
                elif gemini_result["status"] == "CLEAR" and gemini_result["confidence"] >= 80:
                    result["is_blurry"] = False
                    result["confidence"] = gemini_result["confidence"]
                    result["reason"] = gemini_result["reason"]
                else:
                    # Fall back to traditional methods with stricter threshold
                    result["is_blurry"] = combined_score < 35
                    result["confidence"] = combined_score
                    result["reason"] = "Based on traditional blur detection"
            else:
                # Fall back to traditional methods
                result["is_blurry"] = combined_score < self.threshold
                result["confidence"] = combined_score
                result["reason"] = "Based on traditional blur detection only"

            return result

        except Exception as e:
            return {"error": str(e)}
