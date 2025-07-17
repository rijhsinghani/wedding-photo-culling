import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import logging
import sys
# Third-party imports
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import imagehash
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

from src.utils.resize_gemini import resize_for_gemini

from ..config import logger, log_critical
# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI-based quality assessment will be disabled.")
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class GeminiImageAnalyzer:
    def __init__(self):
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
        else:
            self.model = None
        self.hash_threshold = 20
        self.image_hashes = {}
        self.prompt = """
        Analyze this wedding photo with absolute precision for top-tier photo selection:

        MANDATORY REJECTION CRITERIA [Any of these fails the image]:
        - Backs turned to camera for main subjects
        - Distracting foreground elements
        - No clear storytelling moment
        - Poor composition that detracts from subjects
        - No clear main subject
        - Unclear context or significance
        - People walking through frame
        - Major distractions in background
        
        Key Verification Checks [Mandatory]:
        - Eyes: Ensure all subjects have clearly open eyes, no blinks or squints
        - Focus: Every key subject must be in sharp focus, no soft edges on faces
        - Blur: Verify zero camera shake or motion blur, crisp details throughout
        - Exposure: Properly exposed faces with good highlight and shadow detail
        - Positioning: Natural, comfortable poses, facing camera
        - Expression: Genuine emotional moments, no forced or uncomfortable looks
        - Composition: Clear main subject(s), no distracting elements
        - Context: Clear storytelling moment or significance

        Technical Excellence Criteria [40%]:
        - Focus precision on main subjects
        - Lighting quality on faces and key features
        - Color accuracy and skin tone rendering
        - Image resolution and detail clarity
        - Technical sharpness and depth of field
        - Overall composition and framing

        Subject Quality Assessment [30%]:
        - Real emotional engagement
        - Eye contact and direction
        - Group interaction dynamics
        - Key moment significance
        - Body language and pose comfort
        - Expression authenticity

        Emotional Impact [30%]:
        - Moment significance
        - Story-telling value
        - Memory preservation quality
        - Group dynamics
        - Overall emotional resonance

        Format Response as:
        BEST|98|Clear storytelling moment, technically excellent, perfect expressions, ideal lighting
        or
        REJECT|60|Backs turned to camera, distracting elements, unclear moment significance

        Scoring Guidelines:
        95-100: Exceptional quality - keep as highlight
        85-94: Very good quality - keep
        75-84: Good quality with minor issues - review
        Below 75: Reject - significant issues present

        CRITICAL: Any image failing the MANDATORY REJECTION CRITERIA must be marked REJECT with score below 75, regardless of technical quality.
        """

    def compute_image_hash(self, image_path):
        if image_path not in self.image_hashes:
            try:
                with Image.open(image_path).convert("RGB") as img:
                    self.image_hashes[image_path] = imagehash.average_hash(img)
            except Exception as e:
                print(f"Error computing hash for {image_path}: {str(e)}")
                return None
        return self.image_hashes[image_path]

    def verify_image_uniqueness(self, img_path, processed_images):
        img_hash = self.compute_image_hash(img_path)
        if img_hash is None:
            return False
            
        for proc_img in processed_images:
            proc_hash = self.compute_image_hash(proc_img)
            if proc_hash and abs(img_hash - proc_hash) <= self.hash_threshold:
                return False
        return True

    def analyze_image_group(self, image_paths, focus_results=None, blur_results=None, eyes_results=None):
        group_scores = []
        for img_path in image_paths:
            if self._has_quality_issues(img_path, focus_results, blur_results, eyes_results):
                continue
                
            analysis = self.analyze_single_image(img_path)
            if analysis and analysis['decision'] == 'BEST' and analysis['score'] >= 85:
                group_scores.append(analysis)
                
        return max(group_scores, key=lambda x: x['score']) if group_scores else None

    def _has_quality_issues(self, img_path, focus_results, blur_results, eyes_results):
        img_name = os.path.basename(img_path)
        
        if focus_results and isinstance(focus_results, dict):
            out_focus = focus_results.get('out_focus', [])
            if isinstance(out_focus, list):
                if any(img_name in str(img.get('original_path', '')) for img in out_focus):
                    return True
        
        if blur_results and isinstance(blur_results, dict):
            blurry = blur_results.get('blurry', [])
            if isinstance(blurry, list):
                if any(img_name in str(img.get('original_path', '')) for img in blurry):
                    return True
        
        if eyes_results and isinstance(eyes_results, dict):
            closed_eyes = eyes_results.get('closed_eyes', [])
            if isinstance(closed_eyes, list):
                if any(img_name in str(img.get('original_path', '')) for img in closed_eyes):
                    return True
        
        return False


    def analyze_single_image(self, img_path):
        """Analyze a single image with improved error handling and quality criteria."""
        try:
            if not os.path.exists(img_path):
                logger.error(f"Image file not found: {img_path}")
                return None
                
            resized = resize_for_gemini(img_path)
            if not resized:
                logger.error(f"Failed to resize image for analysis: {img_path}")
                return None
            
            try:
                with Image.open(resized) as img:
                    response = self.model.generate_content([self.prompt, img])
                
                try:
                    os.remove(resized)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {resized}: {str(e)}")

                if not response or not response.text:
                    logger.error(f"Empty response from Gemini for {img_path}")
                    return None
                    
                try:
                    parts = response.text.strip().split('|')
                    if len(parts) != 3:
                        logger.error(f"Unexpected response format for {img_path}: {response.text}")
                        return None
                        
                    decision, score, analysis = parts
                    score = float(score)
                    
                    # Enhanced quality control
                    if any(x in analysis.lower() for x in [
                        'back to camera', 
                        'backs turned', 
                        'distracting', 
                        'unclear moment',
                        'no clear subject',
                        'walking through',
                        'poor composition'
                    ]):
                        decision = 'REJECT'
                        score = min(score, 74.0)  # Force score below acceptance threshold
                    
                    return {
                        'image_path': img_path,
                        'decision': decision.strip(),
                        'score': score,
                        'reason': analysis.strip()
                    }
                    
                except ValueError as e:
                    logger.error(f"Failed to parse Gemini response for {img_path}: {str(e)}")
                    return None
                    
            except Exception as e:
                logger.error(f"Gemini API error for {img_path}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error analyzing {img_path}: {str(e)}")
            return None



    def verify_final_quality(self, img_path):
        """Enhanced final quality verification."""
        try:
            resized = resize_for_gemini(img_path)
            if not resized:
                return False
                
            final_prompt = """
            Final Quality Check:
            
            REJECT if ANY of these are true:
            - Backs turned to camera
            - Distracting elements in frame
            - No clear story/moment
            - Poor composition
            - No clear main subject
            - People walking through frame
            
            Key Requirements:
            1. Verify:
            - Subject is in focus
            - No significant blur
            - No major technical flaws
            - Clear storytelling moment
            - Good composition
            
            2. Accept if:
            - Image is technically sound
            - Subject is clear and well-composed
            - Moment has clear significance
            
            Format: PASS|score|reason or FAIL|score|reason
            """
            
            with Image.open(resized) as img:
                response = self.model.generate_content([final_prompt, img])
            os.remove(resized)
            
            if not response or not response.text:
                return False
                
            try:
                decision, score, reason = response.text.strip().split('|')
                # More stringent quality control
                if any(x in reason.lower() for x in [
                    'back to camera', 
                    'distracting', 
                    'unclear moment',
                    'no clear subject'
                ]):
                    return False
                return decision.strip() == 'PASS' and float(score) >= 75
            except ValueError:
                return False
                
        except Exception:
            return False
