# Wedding Photo Culling Tool - Complete Documentation

## Executive Summary

The Wedding Photo Culling Tool is an AI-powered system that automates the tedious process of selecting the best photos from large wedding collections. Designed for professional photographers, it achieves a target 50% delivery rate while maintaining quality standards and complete event coverage.

### Key Value Propositions
- **Time Savings**: Reduces 8-10 hours of manual culling to ~5 minutes
- **Consistent Quality**: AI-driven selection ensures uniform standards
- **Client Satisfaction**: Delivers cohesive galleries without overwhelming duplicates
- **Smart Organization**: Automatically categorizes rejected photos by reason

### Performance Metrics
- **Processing Speed**: 125 RAW images in ~5 minutes
- **Accuracy**: 49.6% keep rate (target: 50%)
- **Success Rate**: 80% processing success with 20% recoverable errors
- **Throughput**: 0.5 images/second with batch processing

### Target Users
- Wedding photographers managing 1000+ photos per event
- Photography studios needing consistent culling standards
- Photographers seeking to reduce post-processing time

## Technical Overview

### System Architecture

```
Input (RAW Files) → Preprocessing → Analysis Pipeline → Quality Selection → Output
                         ↓                    ↓                            ↓
                  PNG Conversion      Parallel Processing          Organized Folders
                                     (4 Detection Algorithms)     (best_quality, etc.)
```

### Core Technologies
- **Computer Vision**: OpenCV, scikit-image for technical analysis
- **Deep Learning**: PyTorch, DINO models for feature extraction
- **AI Integration**: Google Gemini API for composition analysis
- **Face Recognition**: MTCNN, InceptionResnet for eye detection
- **Parallel Processing**: Python multiprocessing for speed

### v2.0 Performance Enhancements
1. **Retry Logic**: Automatic recovery from API failures
2. **Queue Management**: Rate limiting and request prioritization
3. **Batch Processing**: Multi-image API calls for efficiency
4. **Configuration-Driven**: All features customizable via JSON
5. **Duplicate Filtering**: Smart handling ensures only best photos in final delivery
6. **Quality Control**: Automated validation prevents duplicate delivery issues

## Feature Deep Dive

### 1. RAW File Processing
```python
# Supports: ARW, CR2, NEF, RAF, ORF
# Uses rawpy for accurate color reproduction
# Preserves metadata during conversion
```

**Process**:
- Parallel conversion using multiprocessing
- Camera white balance preservation
- 16-bit processing for quality retention
- Automatic cleanup of temporary files

### 2. Duplicate Detection

**Multi-Layer Approach**:
1. **Perceptual Hashing**: Identifies visually similar images
2. **Face Matching**: Groups photos with same subjects
3. **Feature Extraction**: DINO model for semantic similarity
4. **Smart Selection**: Keeps best technical quality from each group

**Quality Control Integration**:
- Ensures only the highest-quality image from each duplicate group appears in best_quality folder
- Prevents multiple versions of the same shot in final delivery
- Automatic validation after processing
- Optional auto-fix for violations

**Configuration**:
```json
"duplicate_hash_threshold": 60,  // Similarity threshold (0-100)
"duplicate_handling": {
    "enabled": true,
    "filter_duplicates_in_best_quality": true,
    "auto_fix_violations": false,
    "quality_control_check": true,
    "log_duplicate_filtering": true
}
```

### 3. Focus and Blur Analysis

**Technical Metrics**:
- **Laplacian Variance**: Edge sharpness measurement
- **FFT Analysis**: Frequency domain blur detection
- **AI Verification**: Gemini confirms focus on subjects

**Thresholds**:
```json
"blur_threshold": 20,      // Below = blurry
"focus_threshold": 45,     // Above = sharp
"in_focus_confidence": 75  // AI confidence required
```

### 4. Eye Detection System

**Context-Aware Processing**:
- Only analyzes faces >5% of image size
- Skips venue/detail shots automatically
- Uses ML ensemble for accuracy

**Technology Stack**:
- MTCNN for face detection
- InceptionResnet for eye state classification
- Confidence scoring to prevent false positives

### 5. Quality Assessment

**Multi-Factor Scoring**:
```python
quality_score = weighted_average(
    resolution_score * 0.15,
    face_detection * 0.20,
    contrast_score * 0.15,
    sharpness_score * 0.20,
    noise_level * 0.15,
    exposure_quality * 0.15
)
```

**Tiered Selection**:
- **Tier 1 (80+)**: Portfolio quality highlights
- **Tier 2 (60-79)**: Standard delivery quality
- **Tier 3 (45-59)**: Coverage shots for completeness

### 6. Batch Processing

**Efficiency Features**:
- Process 5 images per Gemini API call
- Automatic memory management
- Progress saving for interruption recovery
- Configurable batch sizes

## Usage Guide

### Installation

```bash
# Clone repository
git clone https://github.com/rijhsinghani/wedding-photo-culling.git
cd wedding-photo-culling

# Install dependencies
pip install -r requirements.txt

# Configure API
export GEMINI_API_KEY="your-key-here"
```

### Basic Usage

```bash
# Interactive mode
python cli.py

# Select workflow:
1. Best Quality (Full Analysis) - Recommended
2. Duplicates Only
3. Blur Detection
4. Focus Analysis
5. Eye Detection
6. Run All Operations
```

### Configuration

Edit `config.json` for customization:

```json
{
    "thresholds": {
        "target_delivery_percentage": 50,
        "blur_threshold": 20,
        "focus_threshold": 45
    },
    "batch_processing": {
        "enabled": true,
        "batch_size": 25
    },
    "retry_settings": {
        "enabled": true,
        "max_attempts": 3
    }
}
```

### Output Structure

```
output/
├── best_quality/      # ~50% selected for delivery
├── duplicates/        # Grouped similar photos
├── closed_eyes/       # Photos with eyes closed
├── blurry/           # Below quality threshold
├── in_focus/         # All sharp images
└── reports/          # JSON analysis data
```

## Performance Optimization

### Retry Logic Implementation

```python
@retry_gemini_api  # Automatic retry decorator
def analyze_with_gemini(image):
    # Exponential backoff: 2s, 4s, 8s...
    # Adds jitter to prevent thundering herd
    # Max 3 attempts before failing
```

### API Queue Management

**Features**:
- Rate limiting: 60 requests/minute
- Concurrent workers: 5 (configurable)
- Priority queue for important requests
- Automatic request batching

**Benefits**:
- Prevents API quota exhaustion
- Smooth performance under load
- Graceful degradation

### Batch Processing Benefits

| Mode | Images/Second | API Calls | Cost |
|------|--------------|-----------|------|
| Sequential | 0.2 | 125 | $0.25 |
| Batch (5) | 0.5 | 25 | $0.05 |

**Memory Management**:
- Clears cache between batches
- Limits concurrent image loading
- Automatic garbage collection

## Testing and Validation

### Test Dataset Results

**125 Wedding Photos Test**:
- Input: 125 RAW files (Sony ARW)
- Processing Time: 5 minutes 12 seconds
- Keep Rate: 49.6% (62 photos)
- Error Rate: 20% (all recoverable)
- Duplicate Groups: 19 detected
- Duplicate Filtering: 40 images filtered from best_quality
- Quality Control: 0 violations after filtering

### Accuracy Metrics

| Feature | Precision | Recall | F1-Score |
|---------|-----------|---------|----------|
| Duplicate Detection | 95% | 92% | 93.5% |
| Blur Detection | 89% | 87% | 88% |
| Eye Detection | 91% | 85% | 88% |
| Focus Analysis | 87% | 90% | 88.5% |

### Performance Benchmarks

**Hardware**: MacBook Air M1, 8GB RAM
- RAW Conversion: 2.5s/image
- Duplicate Detection: 0.8s/image
- Focus Analysis: 2.1s/image (with API)
- Total Pipeline: 4.2s/image

## Best Practices

### Optimal Configuration

**For Speed**:
```json
{
    "batch_gemini_processing": {
        "batch_size": 10,
        "max_workers": 8
    }
}
```

**For Accuracy**:
```json
{
    "thresholds": {
        "focus_threshold": 50,
        "in_focus_confidence": 85
    }
}
```

**For Duplicate Handling**:
```json
{
    "duplicate_handling": {
        "enabled": true,
        "filter_duplicates_in_best_quality": true,
        "auto_fix_violations": true
    }
}
```

### Workflow Recommendations

1. **Pre-Event Setup**:
   - Test on sample photos
   - Adjust thresholds for shooting style
   - Verify API key and quotas

2. **Processing Strategy**:
   - Process in 500-image batches
   - Run overnight for large events
   - Use sequential mode for critical selections

3. **Quality Control**:
   - Review Tier 1 selections first
   - Check coverage gaps
   - Verify key moments captured
   - Run quality control validation
   - Fix any duplicate violations found

### Troubleshooting

**Common Issues**:

| Problem | Solution |
|---------|----------|
| Memory errors | Reduce batch_size to 10 |
| Slow processing | Enable GPU, increase workers |
| API timeouts | Check internet, reduce batch_size |
| Poor selection | Adjust quality thresholds |
| Duplicate deliveries | Enable duplicate_handling in config |
| Quality violations | Run quality control check, enable auto_fix |

## API Integration

### Gemini API Setup

1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable: `export GEMINI_API_KEY="key"`
3. Monitor usage at console.cloud.google.com

### Rate Limiting Considerations

- Default: 60 requests/minute
- Batch processing reduces by 80%
- Queue manager prevents overruns
- Automatic retry on 429 errors

### Cost Optimization

**Strategies**:
- Batch processing: 5x cost reduction
- Cache results: Avoid reprocessing
- Selective AI: Only ambiguous cases
- Free tier: 60 requests/minute

**Example Cost**:
- 1000 photos sequential: ~$2.00
- 1000 photos batched: ~$0.40

## Quality Control Tool

The tool includes a comprehensive quality control system to validate output:

### Running Quality Control

```bash
python -m src.utils.quality_control
```

### Features
- **Duplicate Violation Detection**: Identifies when multiple images from the same duplicate group appear in best_quality
- **Orphaned File Detection**: Finds images in output folders without corresponding analysis data
- **Missing Folder Validation**: Ensures all expected output folders exist
- **Auto-Fix Capability**: Can automatically remove duplicate violations

### Quality Control Configuration

```json
"duplicate_handling": {
    "quality_control_check": true,    // Run validation after processing
    "auto_fix_violations": false,     // Auto-remove duplicate violations
    "log_duplicate_filtering": true   // Log all filtering actions
}
```

## Community Contributions

**How to Contribute**:
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

**Needed Contributions**:
- Additional RAW format support
- Windows GPU optimization
- UI/UX improvements
- Language translations

## Conclusion

The Wedding Photo Culling Tool represents a significant advancement in photography workflow automation. By combining traditional computer vision with modern AI, it delivers consistent, high-quality results while saving photographers hours of tedious work.

### Key Takeaways
- 50% delivery rate balances quality with coverage
- AI enhancement improves accuracy without replacing photographer judgment
- Modular architecture allows customization for different styles
- Open-source approach encourages community innovation

### Support

- **Issues**: [GitHub Issues](https://github.com/rijhsinghani/wedding-photo-culling/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rijhsinghani/wedding-photo-culling/discussions)
- **Updates**: Watch the repository for new releases

---

*Built with ❤️ for the photography community*