
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
import json
import shutil
from typing import Dict, List, Tuple, Optional, Any, Set
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import time
from tqdm import tqdm
from collections import defaultdict
# Third-party imports
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from imagededup.methods import PHash
import imagehash
import multiprocessing
max_worker = min(multiprocessing.cpu_count(), 16)
import logging
from src.core.quality_assessor import QualityAssessor
from ..config import logger, log_critical


class DuplicateDetector:
    def __init__(self, threshold: int = 10):
        self.threshold = threshold
        self.phash = PHash()
        self.quality_assessor = QualityAssessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.sift = cv2.SIFT_create()
        self.haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.face_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(device=self.device)
        self.image_features_cache = {}
        self.feature_extractor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Additional verification thresholds
        self.face_similarity_threshold = 0.70  # Increased from 0.65
        self.feature_similarity_threshold = 0.70  # Increased from 0.65
        self.background_similarity_threshold = 0.60  # Increased from 0.55
        self.face_verification_threshold = 0.75  # New threshold for strict face verification



    def _count_faces(self, image_path: str) -> int:
        try:
            # Read the image and convert it to grayscale
            img = cv2.imread(image_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using the Haar Cascade
            faces = self.haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.22, minNeighbors=5)
            
            # Return the number of faces detected
            return len(faces)
        
        except Exception as e:
            print(f"Error counting faces: {str(e)}")
            return 0

        

    def _compute_face_embeddings(self, image_path: str) -> List[torch.Tensor]:
        try:
            img = Image.open(image_path).convert('RGB')
            faces = self.mtcnn(img)
            if faces is None:
                return []
            if not isinstance(faces, list):
                faces = [faces]
            embeddings = []
            for face in faces:
                if face is not None:
                    face = face.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.face_embedder(face)
                    embeddings.append(embedding.cpu())
            return embeddings
        except Exception as e:
            logger.error(f"Error computing face embeddings: {str(e)}")
            return []

    def _compute_face_features(self, image_path: str) -> np.ndarray:
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            face_features = []
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                keypoints, descriptors = self.sift.detectAndCompute(face_roi, None)
                if descriptors is not None:
                    face_features.append(descriptors)
            return np.vstack(face_features) if face_features else None
        except Exception as e:
            logger.error(f"Error computing face features: {str(e)}")
            return None

    def _compute_background_features(self, image_path: str) -> np.ndarray:
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            return descriptors
        except Exception as e:
            logger.error(f"Error computing background features: {str(e)}")
            return None

    def _get_image_features(self, image_path: str) -> Dict:
        if image_path in self.image_features_cache:
            return self.image_features_cache[image_path]
        
        features = {
            'face': self._compute_face_features(image_path),
            'face_embeddings': self._compute_face_embeddings(image_path),
            'background': self._compute_background_features(image_path),
            'phash': imagehash.average_hash(Image.open(image_path)),
            'dhash': imagehash.dhash(Image.open(image_path)),  # Additional hash
            'whash': imagehash.whash(Image.open(image_path))   # Additional hash
        }
        self.image_features_cache[image_path] = features
        return features

    def _compute_face_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        if emb1 is None or emb2 is None:
            return 0.0
        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
        return float(cos_sim.item())

    def _verify_same_faces(self, faces1: List[torch.Tensor], faces2: List[torch.Tensor]) -> bool:
        """
        Additional verification step for face matching.
        
        Args:
            faces1 (List[torch.Tensor]): List of face embeddings from first image
            faces2 (List[torch.Tensor]): List of face embeddings from second image
            
        Returns:
            bool: True if faces match, False otherwise
        """
        if not faces1 or not faces2:
            return False
        
        # Check if number of faces match
        if len(faces1) != len(faces2):
            return False
        
        # Track matched faces
        matched_faces = 0
        similarity_threshold = 0.85  # Adjust as needed
        
        # Compare each face from faces1 with faces2
        for face1 in faces1:
            best_match_score = 0
            for face2 in faces2:
                similarity = self._compute_face_similarity(face1, face2)
                if similarity > best_match_score:
                    best_match_score = similarity
            
            if best_match_score >= similarity_threshold:
                matched_faces += 1
        
        # Return True if most faces match
        return matched_faces >= len(faces1) * 0.8  # 80% match threshold
    
    
  #org
    def find_duplicates(self, input_dir: str, output_dir: str = None) -> Dict[str, Dict]:
        temp_dir = None
        try:
            input_dir = os.path.abspath(input_dir)
            logger.info(f"\nAnalyzing images in: {input_dir}")
            
            start_group_num = self._get_last_group_number(output_dir) if output_dir else 0
            print(f"Starting from group number: {start_group_num + 1}")
            
            image_files = []
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                print("No images found in directory")
                return {}
                
            print(f"\nProcessing {len(image_files)} images...")
            processed_groups = {}
            
            # Using original temp directory logic that was working
            temp_parent = os.path.dirname(os.path.dirname(input_dir))
            temp_dir = os.path.join(temp_parent, f'_temp_dup_detection_{int(time.time())}')
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)

            try:
                print("\nGenerating image features...")
                path_mapping = {}
                feature_dict = {}
                    
                for idx, src_path in enumerate(image_files):
                    try:
                        file_ext = os.path.splitext(src_path)[1]
                        temp_name = f"img_{idx}{file_ext}"
                        temp_path = os.path.join(temp_dir, temp_name)
                        shutil.copy2(src_path, temp_path)
                        path_mapping[temp_name] = src_path
                    except Exception as e:
                        logger.error(f"Error copying {src_path}: {str(e)}")
                        continue

                for idx, (temp_name, src_path) in enumerate(path_mapping.items()):
                    try:
                        temp_path = os.path.join(temp_dir, temp_name)
                        img = Image.open(temp_path)
                        
                        features = {
                            'phash': imagehash.average_hash(img),
                            'dhash': imagehash.dhash(img),
                            'whash': imagehash.whash(img),
                            'faces': self._compute_face_embeddings(temp_path),
                            'face_count' : self._count_faces(temp_path),
                            'face_features': self._compute_face_features(temp_path),
                            'background': self._compute_background_features(temp_path)
                        }
                        
                        feature_dict[temp_name] = features.copy()
                        if (idx + 1) % 5 == 0:
                            print(f"Processed {idx + 1}/{len(image_files)} images")
                            
                    except Exception as e:
                        logger.error(f"Error processing {temp_path}: {str(e)}")
                        continue

                

                print("\nFinding initial duplicate groups...")
                initial_groups = []
                processed = set()

                for name1, features1 in feature_dict.items():
                    if name1 in processed:
                        continue
                        
                    current_group = {name1}
                    for name2, features2 in feature_dict.items():
                        if name1 != name2 and name2 not in processed:
                            try:
                                # 1️⃣ Hash Similarity
                                hash_similarity = self._compute_hash_similarity(features1, features2)

                                same_persons = False
                                if features1['face_count']==features2['face_count']:
                                    # print("same no. of persons")
                                    same_persons=True

                                # 2️⃣ Face Similarity with Debugs
                                face_similarity = 0
                                matched_faces = 0  # Counter for face matches
                                # print(f"f1: {len(features1['faces'])} f2: {len(features1['faces'])}")
                                if features1['faces'] and features2['faces']:
                                    face_sims = []
                                    for emb1 in features1['faces']:
                                        for emb2 in features2['faces']:
                                            sim = self._compute_face_similarity(emb1, emb2)
                                            face_sims.append(sim)
                                            if sim >= 0.70:
                                                matched_faces += 1  # Count significant face matches

                                    # Return True if most faces match
                                    # if matched_faces >= len(features1['faces']) * 0.7: # 80% match threshold
                                    #     print("match 70 %")
                                    if face_sims:
                                        face_similarity = max(face_sims)

                                # 3️⃣ Feature Matching Similarity
                                feature_similarity = 0
                                if (features1['face_features'] is not None and 
                                    features2['face_features'] is not None):
                                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                                    matches = bf.match(features1['face_features'], 
                                                    features2['face_features'])
                                    feature_similarity = len(matches) / max(
                                        features1['face_features'].shape[0],
                                        features2['face_features'].shape[0]
                                    )

                                # 4️⃣ Combined Similarity Score
                                combined_score = (
                                    hash_similarity * 0.4 +
                                    face_similarity * 0.3 +
                                    feature_similarity * 0.3
                                )

                                # Debug Logging for Detailed Insights
                                # print(f"\nComparing '{name1}' with '{name2}':")
                                # print(f"  ➡️ Hash Similarity       : {hash_similarity:.3f}")
                                # print(f"  ➡️ Face Similarity       : {face_similarity:.3f} (Matched Faces: {matched_faces})")
                                # print(f"  ➡️ Feature Similarity    : {feature_similarity:.3f}")
                                # print(f"  ➡️ Combined Similarity   : {combined_score:.3f}")

                                # Grouping Condition with Debugging
                                if (hash_similarity >= 0.80 or
                                    face_similarity >= 0.70 or
                                    feature_similarity >= 0.70 or
                                    combined_score >= 0.70) and same_persons:
                                    # print(f"  ✅ Grouping '{name1}' and '{name2}' (Reason: Score thresholds met)")
                                    current_group.add(name2)
                                # else:
                                #     print(f"  ❌ Not Grouping (Reason: Score thresholds NOT met)")

                            except Exception as e:
                                print(f"  ⚠️ Error comparing '{name1}' and '{name2}': {str(e)}")
                                continue

                    if len(current_group) > 1:
                        # print(f"\n✅ Formed Initial Group: {current_group}")
                        initial_groups.append(current_group)
                        processed.update(current_group)


                print(f"\nMerging similar initial groups...{len(initial_groups)}")
                final_groups = []
                sorted_groups = sorted(initial_groups, key=len, reverse=True)
                all_features = feature_dict.copy()
                
                while sorted_groups:
                    current = sorted_groups.pop(0)
                    merged = True
                    while merged:
                        merged = False
                        for other in sorted_groups[:]:
                            try:
                                if self._should_merge_groups(current, other, all_features):
                                    current.update(other)
                                    sorted_groups.remove(other)
                                    merged = True
                                    break
                            except Exception as e:
                                continue
                    final_groups.append(current)

                print(f"\nProcessing {len(final_groups)} final groups...")
                for idx, group in enumerate(final_groups, start=start_group_num + 1):
                    try:
                        original_paths = {path_mapping[name] for name in group}
                        quality_scores = {}
                        
                        for img_path in original_paths:
                            try:
                                quality_result = self.quality_assessor.assess_image_quality(img_path)
                                if quality_result:
                                    quality_scores[img_path] = quality_result['overall_quality_score']
                            except Exception as e:
                                continue
                        
                        if quality_scores:
                            best_image = max(quality_scores.items(), key=lambda x: x[1])[0]
                            similar_images = [img for img in original_paths if img != best_image]
                            
                            group_id = f"group_{idx}"
                            processed_groups[group_id] = {
                                'best_image': os.path.basename(best_image),
                                'similar_images': [os.path.basename(img) for img in similar_images],
                                'quality_scores': {os.path.basename(k): v 
                                            for k, v in quality_scores.items()}
                            }
                            print(f"Processed group {idx}")
                    except Exception as e:
                        continue

            except Exception as e:
                logger.error(f"Error in duplicate processing: {str(e)}")
                return processed_groups
            finally:
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        logger.error(f"Error cleaning up temp directory: {str(e)}")

            return processed_groups

        except Exception as e:
            logger.error(f"Error in duplicate detection: {str(e)}")
            return {}

    def _get_last_group_number(self, output_dir: str) -> int:
        """
        Get the last used group number from existing reports with enhanced validation.
        
        Args:
            output_dir (str): Path to the output directory
            
        Returns:
            int: Last used group number or 0 if none found
        """
        try:
            duplicates_dir = os.path.join(output_dir, 'duplicates')
            if not os.path.exists(duplicates_dir):
                return 0
            
            last_group_num = 0
            
            # Method 1: Check report.json first
            report_path = os.path.join(duplicates_dir, 'report.json')
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                        # Check report structure
                        if (isinstance(report, dict) and 
                            'processed_images' in report and 
                            isinstance(report['processed_images'], dict) and
                            'duplicates' in report['processed_images']):
                            
                            duplicates = report['processed_images']['duplicates']
                            if duplicates:
                                group_numbers = []
                                for group_name in duplicates.keys():
                                    if group_name.startswith('group_'):
                                        try:
                                            num = int(group_name.split('_')[1])
                                            group_numbers.append(num)
                                        except (IndexError, ValueError):
                                            continue
                                if group_numbers:
                                    last_group_num = max(group_numbers)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON format in report file")
                except Exception as e:
                    logger.warning(f"Error reading report file: {str(e)}")
            
            # Method 2: Check physical directories as backup
            try:
                dir_group_numbers = []
                for item in os.listdir(duplicates_dir):
                    full_path = os.path.join(duplicates_dir, item)
                    if os.path.isdir(full_path) and item.startswith('group_'):
                        try:
                            num = int(item.split('_')[1])
                            dir_group_numbers.append(num)
                        except (IndexError, ValueError):
                            continue
                
                if dir_group_numbers:
                    dir_max = max(dir_group_numbers)
                    # Take the maximum between both methods
                    last_group_num = max(last_group_num, dir_max)
            except Exception as e:
                logger.warning(f"Error checking directory groups: {str(e)}")
            
            # Validate final number
            if not isinstance(last_group_num, int) or last_group_num < 0:
                return 0
                
            return last_group_num
            
        except Exception as e:
            logger.error(f"Error in _get_last_group_number: {str(e)}")
            return 0

    def _compute_hash_similarity(self, img1_features: Dict, img2_features: Dict) -> float:
        """Compute weighted average of multiple perceptual hashes"""
        phash_diff = abs(img1_features['phash'] - img2_features['phash'])
        dhash_diff = abs(img1_features['dhash'] - img2_features['dhash'])
        whash_diff = abs(img1_features['whash'] - img2_features['whash'])
        
        # Weighted average of hash differences
        avg_hash_diff = (phash_diff * 0.4 + dhash_diff * 0.3 + whash_diff * 0.3)
        return 1 - (avg_hash_diff / 64)  # Normalize to 0-1 range

    def _are_images_similar(self, img1_path: str, img2_path: str) -> Tuple[bool, str, float]:
        try:
            img1_features = self._get_image_features(img1_path)
            img2_features = self._get_image_features(img2_path)
            
            # Compute hash similarity first (fastest check)
            hash_similarity = self._compute_hash_similarity(img1_features, img2_features)
            if hash_similarity >= 0.85:  # Very high hash similarity
                return True, 'hash', hash_similarity
            
            # Check face similarities
            face_similarities = []
            for emb1 in img1_features['face_embeddings']:
                for emb2 in img2_features['face_embeddings']:
                    sim = self._compute_face_similarity(emb1, emb2)
                    face_similarities.append(sim)
            
            if face_similarities:
                max_face_sim = max(face_similarities)
                if max_face_sim >= self.face_similarity_threshold:
                    # Additional verification for high confidence
                    if self._verify_same_faces(img1_features['face_embeddings'], 
                                             img2_features['face_embeddings']):
                        return True, 'face', max_face_sim

            # Check SIFT features if face matching wasn't conclusive
            if (img1_features['face'] is not None and 
                img2_features['face'] is not None):
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(img1_features['face'], img2_features['face'])
                feature_similarity = len(matches) / max(
                    img1_features['face'].shape[0],
                    img2_features['face'].shape[0]
                )
                if feature_similarity >= self.feature_similarity_threshold:
                    return True, 'feature', feature_similarity

            # Check background as last resort
            if (img1_features['background'] is not None and 
                img2_features['background'] is not None):
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(img1_features['background'], 
                                 img2_features['background'])
                bg_similarity = len(matches) / max(
                    img1_features['background'].shape[0],
                    img2_features['background'].shape[0]
                )
                if bg_similarity >= self.background_similarity_threshold:
                    return True, 'background', bg_similarity

            return False, 'none', 0.0
        except Exception as e:
            logger.error(f"Error comparing images: {str(e)}")
            return False, 'error', 0.0

    def _compute_group_similarity(self, group1: Set[str], group2: Set[str]) -> float:
        """Compute overall similarity between two groups"""
        if not group1 or not group2:
            return 0.0
            
        # Get all face embeddings for both groups
        g1_embeddings = []
        g2_embeddings = []
        for img in group1:
            features = self._get_image_features(img)
            g1_embeddings.extend(features['face_embeddings'])
        for img in group2:
            features = self._get_image_features(img)
            g2_embeddings.extend(features['face_embeddings'])
            
        if not g1_embeddings or not g2_embeddings:
            return 0.0
            
        # Compute face similarities
        face_similarities = []
        for emb1 in g1_embeddings:
            for emb2 in g2_embeddings:
                sim = self._compute_face_similarity(emb1, emb2)
                face_similarities.append(sim)
                
        # Return highest similarity found
        return max(face_similarities) if face_similarities else 0.0

    def _should_merge_groups(self, group1: Set[str], group2: Set[str]) -> bool:
        """Enhanced group merging decision logic with stricter verification"""
        try:
            # First check face counts
            g1_face_count = self._get_group_face_count(group1)
            g2_face_count = self._get_group_face_count(group2)
            
            # If face counts differ significantly, don't merge
            if abs(g1_face_count - g2_face_count) > 1:
                return False

            # Check overall group similarity
            similarity_score = self._compute_group_similarity(group1, group2)
            if similarity_score >= 0.85:  # Increased threshold
                # Verify with face patterns
                return self._verify_face_patterns(group1, group2)
                
            # Detailed verification
            face_match_count = 0
            total_comparisons = 0
            
            # Sample images from each group
            sample_size = min(3, min(len(group1), len(group2)))
            group1_sample = list(group1)[:sample_size]
            group2_sample = list(group2)[:sample_size]
            
            for img1 in group1_sample:
                features1 = self._get_image_features(img1)
                faces1 = features1['face_embeddings']
                
                if not faces1:
                    continue
                    
                for img2 in group2_sample:
                    features2 = self._get_image_features(img2)
                    faces2 = features2['face_embeddings']
                    
                    if not faces2:
                        continue
                        
                    total_comparisons += 1
                    
                    # Enhanced face verification
                    if self._verify_same_faces(faces1, faces2):
                        face_match_count += 1
                        
                    # Check perceptual hashes
                    hash_similarity = self._compute_hash_similarity(features1, features2)
                    if hash_similarity >= 0.90:  # Very strict hash similarity
                        face_match_count += 1
                        
                    # Check background features
                    if (features1['background'] is not None and 
                        features2['background'] is not None):
                        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                        matches = bf.match(features1['background'], 
                                        features2['background'])
                        bg_similarity = len(matches) / max(
                            features1['background'].shape[0],
                            features2['background'].shape[0]
                        )
                        if bg_similarity >= self.background_similarity_threshold:
                            face_match_count += 1

            # Calculate weighted match ratio
            match_ratio = face_match_count / (total_comparisons * 3) if total_comparisons > 0 else 0
            return match_ratio >= 0.70  # Increased threshold for stricter matching

        except Exception as e:
            logger.error(f"Error in group merging: {str(e)}")
            return False

    def _merge_duplicate_groups(self, duplicate_dict: Dict) -> List[Set[str]]:
        """Enhanced duplicate group merging with verification"""
        # Build similarity graph
        graph = defaultdict(dict)
        for img, duplicates in duplicate_dict.items():
            if not duplicates:
                continue
            for dup in duplicates:
                is_similar, match_type, score = self._are_images_similar(img, dup)
                if is_similar:
                    graph[img][dup] = score
                    graph[dup][img] = score
        
        # Sort all pairs by similarity score
        all_pairs = [(n1, n2, s) for n1 in graph for n2, s in graph[n1].items()]
        sorted_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)
        
        visited = set()
        merged_groups = []
        
        def dfs_weighted(node: str, current_group: Set[str], min_score: float = 0.5):
            """DFS with weighted edges for group formation"""
            visited.add(node)
            current_group.add(node)
            for neighbor, score in graph[node].items():
                if neighbor not in visited and score >= min_score:
                    dfs_weighted(neighbor, current_group, min_score)
        
        # Form initial groups
        for node1, node2, score in sorted_pairs:
            if node1 not in visited:
                current_group = set()
                dfs_weighted(node1, current_group)
                if len(current_group) > 1:
                    # Try to merge with existing groups
                    merged = False
                    for existing_group in merged_groups:
                        if self._should_merge_groups(current_group, existing_group):
                            # Double verification before merging
                            if self._verify_group_merge(current_group, existing_group):
                                existing_group.update(current_group)
                                merged = True
                                break
                    if not merged:
                        merged_groups.append(current_group)
        
        return merged_groups

    def _verify_group_merge(self, group1: Set[str], group2: Set[str]) -> bool:
        """Additional verification step before merging groups"""
        # Get representative images from each group
        repr1 = self._get_representative_image(group1)
        repr2 = self._get_representative_image(group2)
        
        if not repr1 or not repr2:
            return False
            
        # Perform strict verification
        features1 = self._get_image_features(repr1)
        features2 = self._get_image_features(repr2)
        
        # Primary verification: Face matching
        if (features1['face_embeddings'] and features2['face_embeddings'] and
            self._verify_same_faces(features1['face_embeddings'], 
                                  features2['face_embeddings'])):
            return True
            
        # Secondary verification: Comprehensive similarity check
        similarity_score = self._compute_comprehensive_similarity(features1, features2)
        if similarity_score >= 0.85:  # High threshold for merging validation
            return True

        # Final verification: Group composition check
        if self._verify_group_composition(group1, group2):
            return True
            
        return False

    def _compute_comprehensive_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate comprehensive similarity score between two feature sets"""
        scores = []
        
        # Hash similarity (30%)
        hash_sim = self._compute_hash_similarity(features1, features2)
        scores.append(hash_sim * 0.30)
        
        # Face feature similarity (40%)
        face_sim = 0.0
        if (features1['face_embeddings'] and features2['face_embeddings']):
            face_similarities = []
            for emb1 in features1['face_embeddings']:
                for emb2 in features2['face_embeddings']:
                    sim = self._compute_face_similarity(emb1, emb2)
                    face_similarities.append(sim)
            if face_similarities:
                face_sim = max(face_similarities)
        scores.append(face_sim * 0.40)
        
        # Background feature similarity (30%)
        bg_sim = 0.0
        if (features1['background'] is not None and features2['background'] is not None):
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(features1['background'], features2['background'])
            bg_sim = len(matches) / max(
                features1['background'].shape[0],
                features2['background'].shape[0]
            )
        scores.append(bg_sim * 0.30)
        
        return sum(scores)

    def _verify_group_composition(self, group1: Set[str], group2: Set[str]) -> bool:
        """Verify if two groups have similar composition of faces"""
        faces_group1 = self._get_group_face_count(group1)
        faces_group2 = self._get_group_face_count(group2)
        
        # If both groups have similar number of faces, they're more likely to be related
        face_count_diff = abs(faces_group1 - faces_group2)
        if face_count_diff <= 1:  # Allow small difference in face counts
            return self._verify_face_patterns(group1, group2)
        
        return False

    def _get_group_face_count(self, group: Set[str]) -> float:
        """Get average number of faces in a group"""
        face_counts = []
        for img_path in group:
            features = self._get_image_features(img_path)
            face_count = len(features['face_embeddings'])
            face_counts.append(face_count)
        
        return sum(face_counts) / len(face_counts) if face_counts else 0

    def _verify_face_patterns(self, group1: Set[str], group2: Set[str]) -> bool:
        """Verify if face patterns are consistent between groups"""
        success_count = 0
        total_comparisons = 0
        
        # Sample up to 3 images from each group
        sample_size = min(3, min(len(group1), len(group2)))
        group1_sample = list(group1)[:sample_size]
        group2_sample = list(group2)[:sample_size]
        
        for img1 in group1_sample:
            features1 = self._get_image_features(img1)
            faces1 = features1['face_embeddings']
            
            if not faces1:
                continue
                
            for img2 in group2_sample:
                features2 = self._get_image_features(img2)
                faces2 = features2['face_embeddings']
                
                if not faces2:
                    continue
                    
                total_comparisons += 1
                
                # Compare face patterns
                if self._match_face_patterns(faces1, faces2):
                    success_count += 1
        
        # Return true if at least 60% of comparisons were successful
        return (success_count / total_comparisons >= 0.6) if total_comparisons > 0 else False

    def _match_face_patterns(self, faces1: List[torch.Tensor], faces2: List[torch.Tensor]) -> bool:
        """Match face embeddings between two sets of faces"""
        if len(faces1) != len(faces2):
            return False
            
        # Track matched faces to prevent double-matching
        matched = set()
        match_count = 0
        
        for face1 in faces1:
            best_match = -1
            best_score = 0
            
            for idx, face2 in enumerate(faces2):
                if idx in matched:
                    continue
                    
                similarity = self._compute_face_similarity(face1, face2)
                if similarity > best_score:
                    best_score = similarity
                    best_match = idx
            
            if best_score >= self.face_similarity_threshold:
                matched.add(best_match)
                match_count += 1
        
        # Return true if most faces match (allowing for some variation)
        return match_count >= (len(faces1) * 0.8)

    def _calculate_group_quality_score(self, group: Set[str]) -> float:
        """Calculate overall quality score for a group of images"""
        quality_scores = []
        face_quality_scores = []
        
        for img_path in group:
            try:
                # Get base quality assessment
                quality_result = self.quality_assessor.assess_image_quality(img_path)
                if quality_result:
                    quality_scores.append(quality_result['overall_quality_score'])
                
                # Get face-specific quality
                features = self._get_image_features(img_path)
                if features['face_embeddings']:
                    face_embeddings = features['face_embeddings']
                    face_score = sum(emb.mean().item() for emb in face_embeddings) / len(face_embeddings)
                    face_quality_scores.append(face_score)
                    
            except Exception as e:
                logger.error(f"Error calculating quality score for {img_path}: {str(e)}")
                continue
        
        # Combine scores with weights
        base_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        face_quality = sum(face_quality_scores) / len(face_quality_scores) if face_quality_scores else 0
        
        return (base_quality * 0.6 + face_quality * 0.4)  # Weight overall quality more than face quality

