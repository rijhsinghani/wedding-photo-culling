"""
Optimized Duplicate Detector with Tiered Approach
Implements fail-fast algorithms for efficient processing
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torchvision
torchvision.disable_beta_transforms_warning()

import json
import time
from typing import Dict, List, Tuple, Optional, Any, Set
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import imagehash
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from src.core.quality_assessor import QualityAssessor
from ..config import logger, log_critical

class OptimizedDuplicateDetector:
    def __init__(self, threshold: int = 10):
        self.threshold = threshold
        self.quality_assessor = QualityAssessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lazy loading - only initialize when needed
        self._mtcnn = None
        self._face_embedder = None
        self._sift = None
        
        # Optimized settings
        self.max_workers = min(multiprocessing.cpu_count() - 1, 4)  # Leave CPU headroom
        self.batch_size = 8  # Process in smaller batches
        
        # Tiered thresholds
        self.hash_threshold = 0.85  # Quick filter
        self.face_threshold = 0.75  # Medium filter
        self.deep_threshold = 0.70  # Final verification
        
        # Cache for processed features
        self.feature_cache = {}
        self.comparison_cache = {}
        
    @property
    def mtcnn(self):
        if self._mtcnn is None:
            self._mtcnn = MTCNN(device=self.device, keep_all=True)
        return self._mtcnn
    
    @property
    def face_embedder(self):
        if self._face_embedder is None:
            self._face_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        return self._face_embedder
    
    @property
    def sift(self):
        if self._sift is None:
            self._sift = cv2.SIFT_create()
        return self._sift

    def _compute_hash_features(self, image_path: str) -> Dict:
        """Fast hash computation for initial filtering"""
        try:
            img = Image.open(image_path).convert('RGB')
            # Resize for faster processing
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            return {
                'phash': imagehash.phash(img),
                'dhash': imagehash.dhash(img),
                'size': img.size,
                'path': image_path
            }
        except Exception as e:
            logger.error(f"Error computing hash for {image_path}: {e}")
            return None

    def _quick_hash_similarity(self, hash1: Dict, hash2: Dict) -> float:
        """Fast hash comparison"""
        if not hash1 or not hash2:
            return 0.0
            
        # Size difference check - quick rejection
        size_ratio = min(hash1['size'][0] / hash2['size'][0], 
                        hash2['size'][0] / hash1['size'][0])
        if size_ratio < 0.5:  # Too different in size
            return 0.0
            
        # Hash similarity
        phash_sim = 1 - (hash1['phash'] - hash2['phash']) / 64.0
        dhash_sim = 1 - (hash1['dhash'] - hash2['dhash']) / 64.0
        
        return (phash_sim + dhash_sim) / 2

    def _compute_face_count(self, image_path: str) -> int:
        """Quick face counting using MTCNN"""
        try:
            img = Image.open(image_path).convert('RGB')
            # Resize for faster detection
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            faces, _ = self.mtcnn.detect(img)
            return len(faces) if faces is not None else 0
        except Exception as e:
            logger.error(f"Error counting faces in {image_path}: {e}")
            return -1

    def _compute_face_embeddings_cached(self, image_path: str) -> Optional[torch.Tensor]:
        """Compute face embeddings with caching"""
        if image_path in self.feature_cache:
            return self.feature_cache[image_path].get('embeddings')
            
        try:
            img = Image.open(image_path).convert('RGB')
            # Detect faces
            faces = self.mtcnn(img)
            
            if faces is None or len(faces) == 0:
                return None
                
            # Get embeddings for all faces
            embeddings = []
            for face in faces:
                if face is not None:
                    face = face.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.face_embedder(face)
                    embeddings.append(embedding.cpu())
                    
            if embeddings:
                # Average embeddings for multiple faces
                result = torch.stack(embeddings).mean(dim=0)
                self.feature_cache[image_path] = {'embeddings': result}
                return result
                
        except Exception as e:
            logger.error(f"Error computing embeddings for {image_path}: {e}")
        return None

    def _face_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute face similarity using cosine similarity"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Ensure tensors are 1D
        if emb1.dim() > 1:
            emb1 = emb1.squeeze()
        if emb2.dim() > 1:
            emb2 = emb2.squeeze()
            
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1)
        return cos_sim.item()

    def find_duplicates_optimized(self, image_dir: str, start_group_num: int = 0) -> Dict:
        """Optimized duplicate detection with tiered approach"""
        print("[cyan]Scanning directories for supported files...")
        
        # Collect all image files
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print("No images found")
            return {}
            
        print(f"[green]Found {len(image_files)} files")
        
        # Phase 1: Quick hash computation for all images
        print("\n[yellow]Phase 1: Computing image hashes (fast)...")
        hash_features = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._compute_hash_features, path): path 
                for path in image_files
            }
            
            with tqdm(total=len(image_files), desc="Computing hashes") as pbar:
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        features = future.result()
                        if features:
                            hash_features[path] = features
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                    pbar.update(1)
        
        # Phase 2: Find potential duplicates using hashes
        print("\n[yellow]Phase 2: Finding potential duplicates (hash-based)...")
        potential_groups = []
        processed = set()
        
        # Sort by file size for better grouping
        sorted_paths = sorted(hash_features.keys(), 
                            key=lambda p: hash_features[p]['size'][0] * hash_features[p]['size'][1], 
                            reverse=True)
        
        for i, path1 in enumerate(tqdm(sorted_paths, desc="Hash comparison")):
            if path1 in processed:
                continue
                
            current_group = {path1}
            hash1 = hash_features[path1]
            
            # Only compare with remaining images
            for path2 in sorted_paths[i+1:]:
                if path2 not in processed:
                    hash2 = hash_features[path2]
                    
                    # Quick hash similarity check
                    similarity = self._quick_hash_similarity(hash1, hash2)
                    if similarity >= self.hash_threshold:
                        current_group.add(path2)
            
            if len(current_group) > 1:
                potential_groups.append(current_group)
                processed.update(current_group)
        
        print(f"Found {len(potential_groups)} potential duplicate groups")
        
        # Phase 3: Verify duplicates with face analysis (only for potential matches)
        print("\n[yellow]Phase 3: Verifying duplicates with face analysis...")
        verified_groups = []
        
        for group_idx, group in enumerate(tqdm(potential_groups, desc="Verifying groups")):
            # Quick face count check
            face_counts = {}
            for path in group:
                count = self._compute_face_count(path)
                face_counts[path] = count
            
            # Split by face count
            count_groups = defaultdict(set)
            for path, count in face_counts.items():
                if count >= 0:  # Valid face count
                    count_groups[count].add(path)
            
            # Process each face count group
            for count, paths in count_groups.items():
                if len(paths) <= 1:
                    continue
                    
                # For images with same face count, do embedding comparison
                if count > 0:  # Has faces
                    verified_subgroup = self._verify_with_embeddings(paths)
                    if verified_subgroup:
                        verified_groups.extend(verified_subgroup)
                else:  # No faces - use hash similarity only
                    verified_groups.append(paths)
        
        # Phase 4: Create final groups with quality assessment
        print("\n[yellow]Phase 4: Assessing quality and creating final groups...")
        final_groups = {}
        
        for idx, group in enumerate(verified_groups, start=start_group_num + 1):
            quality_scores = {}
            
            # Assess quality for each image in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.quality_assessor.assess_image_quality, path): path 
                    for path in group
                }
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        if result:
                            quality_scores[path] = result['overall_quality_score']
                    except Exception as e:
                        logger.error(f"Error assessing quality for {path}: {e}")
                        quality_scores[path] = 0.0
            
            if quality_scores:
                best_image = max(quality_scores.items(), key=lambda x: x[1])[0]
                similar_images = [img for img in group if img != best_image]
                
                group_id = f"group_{idx}"
                final_groups[group_id] = {
                    'best_image': os.path.basename(best_image),
                    'similar_images': [os.path.basename(img) for img in similar_images],
                    'quality_scores': {os.path.basename(k): v for k, v in quality_scores.items()},
                    'group_size': len(group)
                }
        
        print(f"\n[green]Completed! Found {len(final_groups)} duplicate groups")
        return final_groups

    def _verify_with_embeddings(self, paths: Set[str]) -> List[Set[str]]:
        """Verify duplicates using face embeddings"""
        embeddings = {}
        
        # Compute embeddings for all paths
        for path in paths:
            emb = self._compute_face_embeddings_cached(path)
            if emb is not None:
                embeddings[path] = emb
        
        if len(embeddings) < 2:
            return [paths]  # Can't verify, return as is
        
        # Cluster based on face similarity
        verified_groups = []
        processed = set()
        
        for path1, emb1 in embeddings.items():
            if path1 in processed:
                continue
                
            current_group = {path1}
            
            for path2, emb2 in embeddings.items():
                if path2 != path1 and path2 not in processed:
                    similarity = self._face_similarity(emb1, emb2)
                    if similarity >= self.face_threshold:
                        current_group.add(path2)
            
            if len(current_group) > 1:
                verified_groups.append(current_group)
                processed.update(current_group)
        
        return verified_groups