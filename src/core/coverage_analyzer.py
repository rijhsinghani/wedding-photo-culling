"""
Coverage Analyzer for Wedding Photography
Identifies gaps in event coverage and prioritizes important shots
"""

import os
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """Analyzes photo coverage to ensure complete event documentation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.thresholds = config.get('thresholds', {})
        
        # Event moment keywords
        self.key_moments = {
            'ceremony': ['aisle', 'vows', 'rings', 'kiss', 'recessional'],
            'reception': ['entrance', 'first_dance', 'speeches', 'cake', 'bouquet'],
            'portraits': ['bride', 'groom', 'couple', 'family', 'wedding_party'],
            'details': ['dress', 'rings', 'flowers', 'venue', 'decorations']
        }
        
        # Face detection for people tracking
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def analyze_coverage(self, 
                        all_photos: List[str], 
                        selected_photos: List[str],
                        photo_metadata: Dict) -> Dict:
        """
        Analyze coverage gaps and suggest additional photos.
        
        Args:
            all_photos: All available photos
            selected_photos: Currently selected photos
            photo_metadata: Metadata including scores, faces, etc.
            
        Returns:
            Coverage analysis with recommendations
        """
        analysis = {
            'people_coverage': self._analyze_people_coverage(all_photos, selected_photos, photo_metadata),
            'moment_coverage': self._analyze_moment_coverage(all_photos, selected_photos, photo_metadata),
            'time_coverage': self._analyze_time_coverage(all_photos, selected_photos),
            'variety_score': self._calculate_variety_score(selected_photos, photo_metadata),
            'recommendations': []
        }
        
        # Generate recommendations based on gaps
        analysis['recommendations'] = self._generate_recommendations(
            analysis, all_photos, selected_photos, photo_metadata
        )
        
        return analysis
        
    def _analyze_people_coverage(self, 
                                all_photos: List[str], 
                                selected_photos: List[str],
                                photo_metadata: Dict) -> Dict:
        """Analyze which people are represented in selected photos."""
        people_in_selected = defaultdict(int)
        people_in_all = defaultdict(int)
        
        # Count face occurrences
        for photo in all_photos:
            metadata = photo_metadata.get(photo, {})
            faces = metadata.get('faces', [])
            for face_id in faces:
                people_in_all[face_id] += 1
                if photo in selected_photos:
                    people_in_selected[face_id] += 1
                    
        # Identify missing people
        missing_people = []
        for person_id, count in people_in_all.items():
            if person_id not in people_in_selected and count >= 3:
                # Person appears in 3+ photos but none selected
                missing_people.append(person_id)
                
        return {
            'total_people': len(people_in_all),
            'covered_people': len(people_in_selected),
            'missing_people': missing_people,
            'coverage_percentage': len(people_in_selected) / max(len(people_in_all), 1) * 100
        }
        
    def _analyze_moment_coverage(self,
                               all_photos: List[str],
                               selected_photos: List[str],
                               photo_metadata: Dict) -> Dict:
        """Analyze coverage of key wedding moments."""
        moment_coverage = defaultdict(list)
        
        for photo in selected_photos:
            metadata = photo_metadata.get(photo, {})
            detected_moments = metadata.get('moments', [])
            
            for moment in detected_moments:
                moment_coverage[moment].append(photo)
                
        # Check for missing moments
        missing_moments = []
        for category, moments in self.key_moments.items():
            for moment in moments:
                if moment not in moment_coverage:
                    missing_moments.append(f"{category}:{moment}")
                    
        return {
            'covered_moments': list(moment_coverage.keys()),
            'missing_moments': missing_moments,
            'moment_distribution': {k: len(v) for k, v in moment_coverage.items()}
        }
        
    def _analyze_time_coverage(self,
                             all_photos: List[str],
                             selected_photos: List[str]) -> Dict:
        """Analyze temporal coverage of the event."""
        # Extract timestamps from filenames or EXIF
        all_times = self._extract_timestamps(all_photos)
        selected_times = self._extract_timestamps(selected_photos)
        
        if not all_times:
            return {'time_gaps': [], 'coverage_score': 100}
            
        # Find time gaps
        time_gaps = []
        selected_times_sorted = sorted(selected_times)
        
        for i in range(len(selected_times_sorted) - 1):
            gap = selected_times_sorted[i + 1] - selected_times_sorted[i]
            if gap > 600:  # 10 minute gap
                time_gaps.append({
                    'start': selected_times_sorted[i],
                    'end': selected_times_sorted[i + 1],
                    'duration': gap
                })
                
        return {
            'time_gaps': time_gaps,
            'coverage_score': self._calculate_time_coverage_score(all_times, selected_times)
        }
        
    def _calculate_variety_score(self,
                               selected_photos: List[str],
                               photo_metadata: Dict) -> float:
        """Calculate variety in selected photos (angles, compositions, etc)."""
        if not selected_photos:
            return 0.0
            
        variety_factors = {
            'compositions': set(),
            'orientations': set(),
            'shot_types': set(),
            'color_profiles': []
        }
        
        for photo in selected_photos:
            metadata = photo_metadata.get(photo, {})
            
            # Composition variety
            if 'composition' in metadata:
                variety_factors['compositions'].add(metadata['composition'])
                
            # Orientation (portrait/landscape)
            if 'dimensions' in metadata:
                w, h = metadata['dimensions']
                variety_factors['orientations'].add('portrait' if h > w else 'landscape')
                
            # Shot type (close-up, medium, wide)
            if 'shot_type' in metadata:
                variety_factors['shot_types'].add(metadata['shot_type'])
                
        # Calculate variety score
        score = 0.0
        score += len(variety_factors['compositions']) * 10
        score += len(variety_factors['orientations']) * 20
        score += len(variety_factors['shot_types']) * 15
        
        return min(score, 100)
        
    def _generate_recommendations(self,
                                analysis: Dict,
                                all_photos: List[str],
                                selected_photos: List[str],
                                photo_metadata: Dict) -> List[Dict]:
        """Generate specific recommendations for improving coverage."""
        recommendations = []
        selected_set = set(selected_photos)
        
        # Recommend photos with missing people
        if analysis['people_coverage']['missing_people']:
            for person_id in analysis['people_coverage']['missing_people'][:5]:
                # Find best photo with this person
                candidates = []
                for photo in all_photos:
                    if photo not in selected_set:
                        metadata = photo_metadata.get(photo, {})
                        if person_id in metadata.get('faces', []):
                            score = metadata.get('quality_score', 0)
                            candidates.append((photo, score))
                            
                if candidates:
                    best_photo = max(candidates, key=lambda x: x[1])
                    recommendations.append({
                        'photo': best_photo[0],
                        'reason': f'Includes missing person {person_id}',
                        'priority': 'high',
                        'score_boost': 15
                    })
                    
        # Recommend photos for missing moments
        for moment in analysis['moment_coverage']['missing_moments'][:5]:
            candidates = []
            for photo in all_photos:
                if photo not in selected_set:
                    metadata = photo_metadata.get(photo, {})
                    if moment in metadata.get('moments', []):
                        score = metadata.get('quality_score', 0)
                        candidates.append((photo, score))
                        
            if candidates:
                best_photo = max(candidates, key=lambda x: x[1])
                recommendations.append({
                    'photo': best_photo[0],
                    'reason': f'Covers missing moment: {moment}',
                    'priority': 'high',
                    'score_boost': 20
                })
                
        # Recommend photos for time gaps
        for gap in analysis['time_coverage']['time_gaps'][:3]:
            # Find photos in the gap period
            candidates = []
            for photo in all_photos:
                if photo not in selected_set:
                    timestamp = self._get_timestamp(photo)
                    if gap['start'] < timestamp < gap['end']:
                        metadata = photo_metadata.get(photo, {})
                        score = metadata.get('quality_score', 0)
                        candidates.append((photo, score))
                        
            if candidates:
                # Select best 2 photos from gap
                candidates.sort(key=lambda x: x[1], reverse=True)
                for photo, score in candidates[:2]:
                    recommendations.append({
                        'photo': photo,
                        'reason': f'Fills time gap of {gap["duration"]/60:.0f} minutes',
                        'priority': 'medium',
                        'score_boost': 10
                    })
                    
        return recommendations
        
    def _extract_timestamps(self, photos: List[str]) -> List[float]:
        """Extract timestamps from photo files."""
        timestamps = []
        for photo in photos:
            timestamp = self._get_timestamp(photo)
            if timestamp:
                timestamps.append(timestamp)
        return timestamps
        
    def _get_timestamp(self, photo_path: str) -> Optional[float]:
        """Get timestamp from photo (from EXIF or filename)."""
        try:
            # Try to get from filename first (if it contains timestamp)
            filename = os.path.basename(photo_path)
            # Implement timestamp extraction logic based on your filename format
            
            # Fallback to file modification time
            return os.path.getmtime(photo_path)
        except:
            return None
            
    def _calculate_time_coverage_score(self, 
                                     all_times: List[float],
                                     selected_times: List[float]) -> float:
        """Calculate how well selected photos cover the time range."""
        if not all_times or not selected_times:
            return 0.0
            
        # Calculate coverage as percentage of time periods represented
        time_range = max(all_times) - min(all_times)
        if time_range == 0:
            return 100.0
            
        # Divide into time buckets
        num_buckets = 20
        bucket_size = time_range / num_buckets
        
        all_buckets = set()
        selected_buckets = set()
        
        for t in all_times:
            bucket = int((t - min(all_times)) / bucket_size)
            all_buckets.add(bucket)
            
        for t in selected_times:
            bucket = int((t - min(all_times)) / bucket_size)
            selected_buckets.add(bucket)
            
        return len(selected_buckets) / len(all_buckets) * 100