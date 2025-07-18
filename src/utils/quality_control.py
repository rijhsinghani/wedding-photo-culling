#!/usr/bin/env python3
"""
Quality Control Check for Wedding Photo Culling Output

This script validates that the best_quality folder doesn't contain
multiple images from the same duplicate group.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityControlChecker:
    """Validates output quality and duplicate handling"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.best_quality_dir = self.output_dir / "best_quality"
        self.duplicates_dir = self.output_dir / "duplicates"
        self.issues_found = []
        
    def load_duplicate_groups(self) -> Dict[str, str]:
        """Load duplicate group mappings from duplicates folder
        
        Returns:
            Dict mapping image name to group ID
        """
        image_to_group = {}
        
        if not self.duplicates_dir.exists():
            logger.warning("Duplicates directory not found")
            return image_to_group
            
        # Scan duplicate group folders
        for group_folder in self.duplicates_dir.iterdir():
            if group_folder.is_dir() and group_folder.name.startswith("group_"):
                group_id = group_folder.name
                
                # Get all images in this group
                for image_file in group_folder.iterdir():
                    if image_file.is_file():
                        # Remove "BEST_" prefix if present
                        image_name = image_file.name
                        if image_name.startswith("BEST_"):
                            image_name = image_name[5:]
                        
                        image_to_group[image_name] = group_id
                        
        logger.info(f"Loaded {len(image_to_group)} images in {len(set(image_to_group.values()))} duplicate groups")
        return image_to_group
    
    def check_duplicate_violations(self) -> List[Dict]:
        """Check if best_quality contains multiple images from same duplicate group
        
        Returns:
            List of violation details
        """
        violations = []
        image_to_group = self.load_duplicate_groups()
        
        if not image_to_group:
            logger.info("No duplicate groups found, skipping duplicate check")
            return violations
            
        # Group best_quality images by their duplicate group
        groups_in_best = defaultdict(list)
        
        for image_file in self.best_quality_dir.iterdir():
            if image_file.is_file():
                image_name = image_file.name
                
                # Check if this image belongs to a duplicate group
                if image_name in image_to_group:
                    group_id = image_to_group[image_name]
                    groups_in_best[group_id].append(image_name)
        
        # Find violations (groups with more than one image in best_quality)
        for group_id, images in groups_in_best.items():
            if len(images) > 1:
                violation = {
                    'group_id': group_id,
                    'images': images,
                    'count': len(images),
                    'issue': f"Multiple images from {group_id} in best_quality folder"
                }
                violations.append(violation)
                logger.warning(f"VIOLATION: {violation['issue']} - Images: {', '.join(images)}")
        
        return violations
    
    def check_missing_categories(self) -> List[str]:
        """Check if any expected category folders are missing"""
        expected_folders = ['best_quality', 'duplicates', 'closed_eyes', 'blurry', 'in_focus']
        missing = []
        
        for folder in expected_folders:
            folder_path = self.output_dir / folder
            if not folder_path.exists():
                missing.append(folder)
                logger.warning(f"Missing expected folder: {folder}")
                
        return missing
    
    def check_orphaned_files(self) -> List[str]:
        """Check for files that appear in best_quality but not in any category"""
        orphaned = []
        
        # Get all files in best_quality
        best_files = set()
        for f in self.best_quality_dir.iterdir():
            if f.is_file():
                best_files.add(f.name)
        
        # Check if each file appears in at least one category
        categories = ['in_focus', 'blurry', 'closed_eyes']
        for file_name in best_files:
            found = False
            
            for category in categories:
                category_dir = self.output_dir / category
                if category_dir.exists():
                    if (category_dir / file_name).exists():
                        found = True
                        break
            
            # Also check duplicate groups
            if not found:
                for group_folder in self.duplicates_dir.iterdir():
                    if group_folder.is_dir():
                        for dup_file in group_folder.iterdir():
                            if dup_file.name == file_name or dup_file.name == f"BEST_{file_name}":
                                found = True
                                break
                    if found:
                        break
            
            if not found:
                orphaned.append(file_name)
                logger.warning(f"Orphaned file in best_quality: {file_name}")
                
        return orphaned
    
    def calculate_statistics(self) -> Dict:
        """Calculate output statistics"""
        stats = {}
        
        # Count files in each category
        for folder in ['best_quality', 'in_focus', 'blurry', 'closed_eyes']:
            folder_path = self.output_dir / folder
            if folder_path.exists():
                count = sum(1 for f in folder_path.iterdir() if f.is_file())
                stats[folder] = count
            else:
                stats[folder] = 0
        
        # Count duplicate groups
        if self.duplicates_dir.exists():
            group_count = sum(1 for d in self.duplicates_dir.iterdir() if d.is_dir())
            total_duplicates = 0
            for group_folder in self.duplicates_dir.iterdir():
                if group_folder.is_dir():
                    total_duplicates += sum(1 for f in group_folder.iterdir() if f.is_file())
            stats['duplicate_groups'] = group_count
            stats['total_duplicates'] = total_duplicates
        
        return stats
    
    def generate_report(self) -> Dict:
        """Generate comprehensive quality control report"""
        logger.info("Starting quality control check...")
        
        report = {
            'output_directory': str(self.output_dir),
            'violations': {
                'duplicate_violations': self.check_duplicate_violations(),
                'missing_folders': self.check_missing_categories(),
                'orphaned_files': self.check_orphaned_files()
            },
            'statistics': self.calculate_statistics(),
            'summary': {
                'total_violations': 0,
                'status': 'PASS'
            }
        }
        
        # Calculate total violations
        total_violations = (
            len(report['violations']['duplicate_violations']) +
            len(report['violations']['missing_folders']) +
            len(report['violations']['orphaned_files'])
        )
        
        report['summary']['total_violations'] = total_violations
        report['summary']['status'] = 'FAIL' if total_violations > 0 else 'PASS'
        
        return report
    
    def fix_duplicate_violations(self, dry_run: bool = True) -> List[str]:
        """Remove duplicate violations by keeping only the best image from each group
        
        Args:
            dry_run: If True, only report what would be done without making changes
            
        Returns:
            List of actions taken
        """
        actions = []
        violations = self.check_duplicate_violations()
        
        if not violations:
            logger.info("No duplicate violations to fix")
            return actions
        
        # Load quality scores if available
        best_quality_report = self.output_dir / "best_quality_report.json"
        quality_scores = {}
        
        if best_quality_report.exists():
            with open(best_quality_report, 'r') as f:
                report_data = json.load(f)
                # Extract scores from report
                for photo_data in report_data.get('selected_photos', []):
                    if isinstance(photo_data, dict):
                        filename = Path(photo_data.get('original_path', '')).name
                        score = photo_data.get('quality_score', 0)
                        quality_scores[filename] = score
        
        for violation in violations:
            group_id = violation['group_id']
            images = violation['images']
            
            # Find the best image based on quality scores
            best_image = None
            best_score = -1
            
            for image in images:
                score = quality_scores.get(image, 0)
                if score > best_score:
                    best_score = score
                    best_image = image
            
            # If no scores available, keep the first one
            if best_image is None:
                best_image = images[0]
            
            # Remove all except the best
            for image in images:
                if image != best_image:
                    image_path = self.best_quality_dir / image
                    action = f"Remove {image} from best_quality (keeping {best_image} for {group_id})"
                    actions.append(action)
                    
                    if not dry_run:
                        image_path.unlink()
                        logger.info(f"Removed: {image}")
                    else:
                        logger.info(f"Would remove: {image}")
        
        return actions


def main():
    """Run quality control check"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quality control check for photo culling output')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('--fix', action='store_true', help='Fix violations (remove duplicates)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')
    parser.add_argument('--json', help='Save report to JSON file')
    
    args = parser.parse_args()
    
    # Run quality control check
    checker = QualityControlChecker(args.output_dir)
    report = checker.generate_report()
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY CONTROL REPORT")
    print("="*60)
    
    print(f"\nOutput Directory: {report['output_directory']}")
    print(f"Status: {report['summary']['status']}")
    print(f"Total Violations: {report['summary']['total_violations']}")
    
    # Print statistics
    print("\nStatistics:")
    for key, value in report['statistics'].items():
        print(f"  {key}: {value}")
    
    # Print violations
    if report['violations']['duplicate_violations']:
        print(f"\nDuplicate Violations Found: {len(report['violations']['duplicate_violations'])}")
        for v in report['violations']['duplicate_violations']:
            print(f"  - {v['group_id']}: {v['count']} images ({', '.join(v['images'])})")
    
    if report['violations']['missing_folders']:
        print(f"\nMissing Folders: {', '.join(report['violations']['missing_folders'])}")
    
    if report['violations']['orphaned_files']:
        print(f"\nOrphaned Files: {len(report['violations']['orphaned_files'])}")
    
    # Fix violations if requested
    if args.fix or args.dry_run:
        print("\n" + "-"*60)
        print("FIXING VIOLATIONS" if args.fix else "DRY RUN - No changes will be made")
        print("-"*60)
        
        actions = checker.fix_duplicate_violations(dry_run=args.dry_run or not args.fix)
        for action in actions:
            print(f"  - {action}")
        
        if actions:
            print(f"\nTotal actions: {len(actions)}")
    
    # Save JSON report if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.json}")
    
    # Exit with error code if violations found
    return 0 if report['summary']['status'] == 'PASS' else 1


if __name__ == '__main__':
    exit(main())