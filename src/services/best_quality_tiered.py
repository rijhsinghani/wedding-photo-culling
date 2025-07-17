"""
Tiered Best Quality Selection Service
Implements intelligent 50% delivery target for wedding photography
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import logging

from src.core.gemini_image_analyzer import GeminiImageAnalyzer
from src.core.coverage_analyzer import CoverageAnalyzer
from src.core.processing_cache import ProcessingCache
from src.utils.util import setup_directories, save_json_report, get_all_image_files

logger = logging.getLogger(__name__)


class TieredQualitySelector:
    """Implements tiered selection system for client delivery."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.thresholds = config.get('thresholds', {})
        self.analyzer = GeminiImageAnalyzer()
        self.coverage_analyzer = CoverageAnalyzer(config)
        
        # Tiered thresholds
        self.tier_1_min = self.thresholds.get('tier_1_threshold', 80)
        self.tier_2_min = self.thresholds.get('tier_2_threshold', 60)
        self.tier_3_min = self.thresholds.get('tier_3_threshold', 45)
        self.solo_threshold = self.thresholds.get('solo_shot_threshold', 40)
        self.key_moment_threshold = self.thresholds.get('key_moment_threshold', 35)
        
        # Target delivery
        self.target_percentage = self.thresholds.get('target_delivery_percentage', 50)
        
    def select_for_delivery(self, 
                          all_photos: Dict[str, Dict],
                          existing_results: Dict) -> Dict[str, List[str]]:
        """
        Select photos for client delivery using tiered system.
        
        Args:
            all_photos: Dictionary of photo_path -> metadata
            existing_results: Results from previous processing steps
            
        Returns:
            Dictionary of tier -> list of selected photos
        """
        total_photos = len(all_photos)
        target_count = int(total_photos * self.target_percentage / 100)
        
        logger.info(f"Selecting photos for delivery: target {target_count} of {total_photos}")
        
        # Initialize tiers
        tiers = {
            'tier_1': [],  # Perfect shots
            'tier_2': [],  # Good shots
            'tier_3': [],  # Coverage shots
            'coverage': []  # Important for story
        }
        
        selected_photos = set()
        
        # Phase 1: Select all Tier 1 photos (score 80+)
        for photo, metadata in all_photos.items():
            score = metadata.get('quality_score', 0)
            if score >= self.tier_1_min and self._passes_basic_checks(photo, existing_results):
                tiers['tier_1'].append(photo)
                selected_photos.add(photo)
                
        logger.info(f"Tier 1 (80+): {len(tiers['tier_1'])} photos")
        
        # Phase 2: Add Tier 2 photos (score 60-79)
        if len(selected_photos) < target_count:
            for photo, metadata in all_photos.items():
                if photo not in selected_photos:
                    score = metadata.get('quality_score', 0)
                    if self.tier_2_min <= score < self.tier_1_min and self._passes_basic_checks(photo, existing_results):
                        tiers['tier_2'].append(photo)
                        selected_photos.add(photo)
                        
                        if len(selected_photos) >= target_count:
                            break
                            
        logger.info(f"Tier 2 (60-79): {len(tiers['tier_2'])} photos")
        
        # Phase 3: Analyze coverage and fill gaps
        coverage_analysis = self.coverage_analyzer.analyze_coverage(
            list(all_photos.keys()),
            list(selected_photos),
            all_photos
        )
        
        # Add recommended photos for coverage
        for recommendation in coverage_analysis['recommendations']:
            if len(selected_photos) >= target_count:
                break
                
            photo = recommendation['photo']
            if photo not in selected_photos:
                # Apply score boost for coverage importance
                boosted_score = all_photos[photo].get('quality_score', 0) + recommendation['score_boost']
                
                if boosted_score >= self.key_moment_threshold:
                    tiers['coverage'].append(photo)
                    selected_photos.add(photo)
                    logger.info(f"Added for coverage: {photo} (reason: {recommendation['reason']})")
                    
        # Phase 4: Fill remaining with Tier 3 photos (45-59)
        if len(selected_photos) < target_count:
            tier_3_candidates = []
            for photo, metadata in all_photos.items():
                if photo not in selected_photos:
                    score = metadata.get('quality_score', 0)
                    if self.tier_3_min <= score < self.tier_2_min:
                        tier_3_candidates.append((photo, score))
                        
            # Sort by score and add best ones
            tier_3_candidates.sort(key=lambda x: x[1], reverse=True)
            
            for photo, score in tier_3_candidates:
                if len(selected_photos) >= target_count:
                    break
                    
                if self._passes_basic_checks(photo, existing_results):
                    tiers['tier_3'].append(photo)
                    selected_photos.add(photo)
                    
        logger.info(f"Tier 3 (45-59): {len(tiers['tier_3'])} photos")
        logger.info(f"Coverage additions: {len(tiers['coverage'])} photos")
        
        # Phase 5: Ensure variety and solo shots
        self._ensure_solo_coverage(all_photos, selected_photos, tiers, existing_results)
        
        return tiers
        
    def _passes_basic_checks(self, photo: str, existing_results: Dict) -> bool:
        """Check if photo passes basic quality requirements."""
        # Skip if in excluded categories
        excluded_categories = ['blurry', 'closed_eyes', 'poor_quality']
        
        for category in excluded_categories:
            if category in existing_results:
                category_files = [os.path.basename(f['original_path']) 
                                for f in existing_results[category] 
                                if isinstance(f, dict) and 'original_path' in f]
                if os.path.basename(photo) in category_files:
                    return False
                    
        return True
        
    def _ensure_solo_coverage(self, 
                            all_photos: Dict,
                            selected_photos: Set,
                            tiers: Dict,
                            existing_results: Dict):
        """Ensure every important person has at least one good solo shot."""
        # Identify people with no solo shots in selection
        people_coverage = {}
        
        for photo in selected_photos:
            metadata = all_photos.get(photo, {})
            faces = metadata.get('faces', [])
            
            if len(faces) == 1:  # Solo shot
                person_id = faces[0]
                if person_id not in people_coverage:
                    people_coverage[person_id] = photo
                    
        # Find solo shots for missing people
        for photo, metadata in all_photos.items():
            if photo in selected_photos:
                continue
                
            faces = metadata.get('faces', [])
            if len(faces) == 1:  # Solo shot
                person_id = faces[0]
                score = metadata.get('quality_score', 0)
                
                if person_id not in people_coverage and score >= self.solo_threshold:
                    # Add this solo shot
                    tiers['coverage'].append(photo)
                    selected_photos.add(photo)
                    people_coverage[person_id] = photo
                    logger.info(f"Added solo shot for person {person_id}: {photo}")


def process_best_quality_tiered(input_dir: str, 
                              output_dir: str, 
                              raw_results: dict, 
                              config: dict,
                              cache: ProcessingCache = None,
                              exclude_files: set = None) -> dict:
    """
    Process images for best quality selection with tiered delivery system.
    """
    if cache is None:
        cache = ProcessingCache(output_dir)
    if exclude_files is None:
        exclude_files = set()
        
    directories = setup_directories(output_dir)
    
    selector = TieredQualitySelector(config)
    
    # Collect all photos with metadata
    all_photos = {}
    image_files = get_all_image_files(input_dir, config['supported_formats'])
    
    # Get quality scores from existing analysis
    in_focus_results = raw_results.get('focus_results', {}).get('in_focus', [])
    quality_scores = {}
    
    # Extract scores from various sources
    for result in in_focus_results:
        if isinstance(result, dict) and 'original_path' in result:
            path = result['original_path']
            score = result.get('score', 50)
            quality_scores[path] = score
    
    # Build metadata for all photos
    logger.info("Building photo metadata...")
    for photo in tqdm(image_files, desc="Analyzing photos"):
        if os.path.basename(photo) in exclude_files:
            continue
            
        metadata = {
            'quality_score': quality_scores.get(photo, 45),
            'faces': [],  # Would be populated from face detection
            'moments': [],  # Would be populated from scene analysis
            'timestamp': os.path.getmtime(photo)
        }
        
        all_photos[photo] = metadata
    
    # Select photos using tiered system
    logger.info("Selecting photos for delivery...")
    selected_tiers = selector.select_for_delivery(all_photos, raw_results)
    
    # Copy selected photos to best_quality directory
    results = {
        'best_quality': [],
        'stats': {
            'total_processed': len(all_photos),
            'total_selected': 0,
            'delivery_percentage': 0,
            'tier_distribution': {}
        }
    }
    
    # Copy photos to best_quality folder
    logger.info("Copying selected photos to best_quality...")
    all_selected = set()
    
    for tier, photos in selected_tiers.items():
        results['stats']['tier_distribution'][tier] = len(photos)
        all_selected.update(photos)
        
        for photo in photos:
            # Copy photo to best_quality
            dest_path = os.path.join(directories['best_quality'], os.path.basename(photo))
            shutil.copy2(photo, dest_path)
            
            results['best_quality'].append({
                'original_path': photo,
                'tier': tier,
                'score': all_photos[photo]['quality_score']
            })
    
    results['stats']['total_selected'] = len(all_selected)
    results['stats']['delivery_percentage'] = (len(all_selected) / len(all_photos) * 100) if all_photos else 0
    
    # Save detailed report at root level
    report_data = {
        'summary': results['stats'],
        'tier_details': {tier: len(photos) for tier, photos in selected_tiers.items()},
        'selected_photos': results['best_quality'],
        'coverage_analysis': selector.coverage_analyzer.analyze_coverage(
            list(all_photos.keys()),
            list(all_selected),
            all_photos
        )
    }
    
    save_json_report(directories['best_quality'], report_data, config, 'best_quality_report.json')
    
    # Print summary
    print(f"\n[green]Client Delivery Summary:")
    print(f"Total photos: {results['stats']['total_processed']}")
    print(f"Selected for delivery: {results['stats']['total_selected']} ({results['stats']['delivery_percentage']:.1f}%)")
    print(f"\nTier Distribution:")
    for tier, count in results['stats']['tier_distribution'].items():
        print(f"  {tier}: {count} photos")
    
    return results


