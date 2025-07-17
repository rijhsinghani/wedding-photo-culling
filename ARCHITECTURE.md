# Wedding Photo Culling System Architecture

## Overview

The Wedding Photo Culling System is an intelligent photo selection tool designed to help photographers efficiently process large wedding photo collections. It automatically identifies and delivers the best ~50% of photos while organizing rejected photos by reason, making it clear what should be delivered to clients.

### Core Philosophy
- **50% Delivery Target**: Delivers approximately half of the photos to ensure clients receive a cohesive, high-quality gallery without overwhelming duplicates
- **Quality + Coverage**: Balances technical excellence with complete event documentation
- **Smart Detection**: Uses context-aware algorithms to avoid false positives
- **Simple Organization**: 4-folder output structure makes it immediately clear what's kept vs. discarded

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input Directory                          │
│                    (RAW files: ARW, CR2, NEF, etc.)             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAW Preprocessing Layer                       │
│                 (Parallel RAW → PNG conversion)                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Optimized Processing Pipeline                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Duplicate Detection (Sequential)                      │   │
│  │    - Perceptual hashing                                  │   │
│  │    - Face similarity matching                            │   │
│  │    - Best selection per group                            │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                     │
│  ┌─────────────────────────┴───────────────────────────────┐   │
│  │ 2. Parallel Analysis                                     │   │
│  │    ┌─────────────────┐     ┌─────────────────┐         │   │
│  │    │ Blur Detection  │     │ Focus Analysis  │         │   │
│  │    │ - Laplacian     │     │ - DINO model    │         │   │
│  │    │ - FFT analysis  │     │ - Gemini verify │         │   │
│  │    └─────────────────┘     └─────────────────┘         │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                     │
│  ┌─────────────────────────┴───────────────────────────────┐   │
│  │ 3. Context-Aware Eye Detection                          │   │
│  │    - Face size threshold (>5% of image)                 │   │
│  │    - ML model (MTCNN + InceptionResnet)                 │   │
│  │    - Skip venue/decor shots                             │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                     │
│  ┌─────────────────────────┴───────────────────────────────┐   │
│  │ 4. Tiered Quality Selection (50% target)                │   │
│  │    - Tier 1 (80-100): Portfolio quality                 │   │
│  │    - Tier 2 (60-79): Delivery quality                   │   │
│  │    - Tier 3 (45-59): Coverage quality                   │   │
│  │    - Coverage gap analysis                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Organization                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │  best_quality/  │  │ duplicates/  │  │ closed_eyes/   │    │
│  │  ✓ KEEP (50%)   │  │ ✗ DISCARD    │  │ ✗ DISCARD      │    │
│  └─────────────────┘  └──────────────┘  └────────────────┘    │
│                       ┌──────────────┐                          │
│                       │   blurry/    │                          │
│                       │  ✗ DISCARD   │                          │
│                       └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Detection Algorithms

#### Blur Detection (`src/core/blur_detector.py`)
- **Laplacian Variance**: Measures edge sharpness
- **FFT Analysis**: Detects motion blur patterns
- **Overexposure Check**: Identifies blown-out highlights
- **Threshold**: < 25 = blurry

#### Focus Analysis (`src/core/focus_detector.py`)
- **DINO Model**: Deep learning-based focus detection
- **Gemini Verification**: AI-powered quality assessment
- **Multi-level Classification**:
  - In-focus (score > 50)
  - Off-focus (score 20-50)
  - Severely blurred (score < 20)

#### Eye Detection (`src/core/eye_detector.py`)
- **Context-Aware Processing**:
  - Skip if no faces detected
  - Skip if largest face < 5% of image area
  - Skip venue/decor shots
- **Dual Verification**:
  - ML model (MTCNN + InceptionResnet)
  - Gemini API confirmation
- **Smart for Event Photography**: Only checks when relevant

#### Duplicate Detection (`src/core/duplicate_detector_optimized.py`)
- **Tiered Approach**:
  1. Quick hash comparison
  2. Face similarity matching
  3. Quality assessment
- **Best Selection**: Highest quality photo marked with "BEST_" prefix
- **Group Organization**: Similar photos kept together

### 2. Quality Assessment System

#### Tiered Selection (`src/services/best_quality_tiered.py`)
```
Tier 1 (80-100 points): Portfolio Quality
- Perfect technical execution
- Exceptional moments
- Could feature in photographer's portfolio

Tier 2 (60-79 points): Delivery Quality  
- Good technical quality
- Clear subjects and moments
- Client would want to keep

Tier 3 (45-59 points): Coverage Quality
- Acceptable quality
- Important for complete coverage
- Key people/moments despite minor flaws
```

#### Coverage Analyzer (`src/core/coverage_analyzer.py`)
- **People Coverage**: Ensures everyone is represented
- **Moment Coverage**: Checks for key event moments
- **Time Coverage**: Identifies temporal gaps
- **Variety Score**: Ensures diverse shots

### 3. Processing Pipeline

#### Parallel Processor (`src/utils/parallel_processor.py`)
```python
1. Sequential: Duplicate Detection
   └── Must complete first (provides face data)

2. Parallel: Blur + Focus Detection
   ├── Independent analyses
   └── Run concurrently for speed

3. Sequential: Eye Detection
   └── Uses face data from step 1

4. Sequential: Tiered Quality Selection
   └── Aggregates all previous results
```

## Folder Structure

### Input Structure
```
input_directory/
├── DSC001.ARW
├── DSC002.CR2
├── subfolder/
│   └── DSC003.NEF
└── ... (supports nested directories)
```

### Output Structure
```
output_directory/
├── best_quality/        # ✓ Photos for client delivery (~50%)
│   ├── DSC001.ARW
│   ├── DSC005.ARW
│   └── ...
├── duplicates/          # ✗ Similar photos grouped
│   ├── group_1/
│   │   ├── BEST_DSC002.ARW  # Best of group
│   │   ├── DSC003.ARW       # Similar shot
│   │   └── DSC004.ARW       # Similar shot
│   └── group_2/
├── closed_eyes/         # ✗ Portraits with eyes closed
├── blurry/             # ✗ All quality issues
├── best_quality_report.json
├── duplicates_report.json
├── blur_report.json
├── closed_eyes_report.json
└── focus_report.json
```

## Key Algorithms

### 1. Duplicate Group Best Selection
```python
For each duplicate group:
1. Assess quality of each photo:
   - Sharpness score
   - Focus quality
   - Eye status
   - Composition
2. Select highest scoring photo
3. Mark with "BEST_" prefix
4. Include in 50% selection pool
```

### 2. 50% Selection Algorithm
```python
1. Select all Tier 1 photos (80+ score)
2. If < 50%, add Tier 2 photos (60-79)
3. If still < 50%, analyze coverage gaps:
   - Missing people
   - Missing key moments
   - Time gaps
4. Add coverage photos with score boost
5. If still < 50%, add best Tier 3 photos (45-59)
6. Ensure variety and solo shot coverage
```

### 3. Context-Aware Eye Detection
```python
def should_check_eyes(image):
    faces = detect_faces(image)
    if not faces:
        return False  # Venue/decor shot
    
    largest_face_ratio = largest_face_area / image_area
    if largest_face_ratio < 0.05:
        return False  # Background people
    
    significant_faces = count_faces_above_threshold(faces, 0.03)
    if significant_faces == 0:
        return False  # No clear subjects
    
    return True  # Portrait/group photo
```

## Configuration

### Key Thresholds (`config.json`)
```json
{
  "thresholds": {
    "blur_threshold": 25,
    "focus_threshold": 50,
    "best_quality_min_score": 45,
    "target_delivery_percentage": 50,
    "tier_1_threshold": 80,
    "tier_2_threshold": 60,
    "tier_3_threshold": 45,
    "face_size_threshold": 0.05,
    "coverage_importance_boost": 1.2
  }
}
```

### Customization Points
- **Delivery Percentage**: Adjust `target_delivery_percentage`
- **Quality Tiers**: Modify tier thresholds
- **Technical Standards**: Tune blur/focus thresholds
- **Coverage Priority**: Adjust boost factors

## Performance Optimizations

### 1. Parallel Processing
- RAW conversion: Multi-process (CPU count - 1)
- Blur/Focus: Concurrent ThreadPoolExecutor
- Batch processing for memory efficiency

### 2. Tiered Duplicate Detection
- Quick hash filter (85% similarity)
- Face matching only for candidates
- Early rejection for efficiency

### 3. Memory Management
- Batch processing (25 images/batch)
- Garbage collection between batches
- Temporary file cleanup

### 4. Caching System
- Avoid reprocessing completed images
- Persistent cache between runs
- Category-based tracking

## Error Handling

### Graceful Degradation
- Missing Gemini API: Falls back to ML-only
- GPU unavailable: Automatic CPU fallback
- Corrupted images: Skip and log

### Progress Preservation
- Checkpoint saves every 10 images
- Resume capability for interrupted runs
- Detailed error logging

## Integration Points

### Gemini API
- Optional enhancement for quality assessment
- Provides context-aware evaluation
- Fallback to local models if unavailable

### File Format Support
- **RAW**: ARW, CR2, NEF, RAF, ORF
- **Standard**: JPG, PNG, TIFF, BMP
- Automatic RAW → PNG conversion

### Extensibility
- Pluggable detection algorithms
- Configurable quality criteria
- Custom coverage rules

## Usage Flow

1. **Photographer captures** 1000+ photos at wedding
2. **System processes** all photos through pipeline
3. **Delivers ~500 photos** in best_quality folder
4. **Rejects organized** by reason for reference
5. **Clear distinction**: Keep best_quality, discard others

This architecture ensures photographers can confidently deliver a cohesive, high-quality gallery that tells the complete wedding story without overwhelming clients with redundant shots.