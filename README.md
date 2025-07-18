# Wedding Photo Culling Assistant 📸

An intelligent, high-performance photo culling system designed to help photographers efficiently sort through large collections of wedding photos. The system uses advanced computer vision, machine learning, and AI techniques to identify the highest quality images while filtering out blurry, duplicate, or poorly composed photos.

## 🌟 Key Features

- **50% Client Delivery Target**: Intelligently selects ~50% of photos for cohesive client galleries
- **Tiered Quality System**: Organizes photos into delivery tiers (T1: 80+, T2: 60-79, T3: 45-59)
- **Coverage Analysis**: Ensures complete event documentation with gap detection
- **RAW File Support**: Automatically converts and processes RAW files (ARW, CR2, NEF, RAF, ORF)
- **Intelligent Duplicate Detection**: Identifies similar photos using perceptual hashing and face recognition
- **Multi-Algorithm Quality Assessment**: 
  - Blur detection using Laplacian and FFT analysis
  - Focus analysis with in-focus/off-focus categorization
  - Eye detection to identify photos with closed eyes
  - AI-powered composition analysis (with Gemini API)
- **Parallel Processing**: Optimized workflow runs independent analyses concurrently
- **Batch Processing**: Handles large collections efficiently with memory management
- **Progress Tracking**: Save and resume processing for interrupted sessions
- **Cross-Platform**: Works on macOS, Linux, and Windows

## 🚀 Performance

- Processes 125 high-resolution RAW images in ~5 minutes
- Parallel execution reduces processing time by 60%
- Memory-efficient batch processing prevents system overload
- Automatic caching avoids redundant processing

## 📋 Requirements

- Python 3.9 or higher
- 8GB+ RAM (16GB recommended for large collections)
- Optional: CUDA-capable GPU for faster processing
- Optional: Google Gemini API key for AI-powered analysis

## 🛠️ Installation

### Quick Setup (macOS/Linux)

```bash
# Clone the repository
git clone https://github.com/yourusername/cullingalgorithm.git
cd cullingalgorithm

# Run the setup script
chmod +x setup.sh
./setup.sh

# Or with virtual environment (recommended)
./setup.sh --venv
source venv/bin/activate
```

### Manual Setup

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python setup.py

# Create .env file from template
cp .env.example .env
```

### Configure Gemini API (Optional)

For enhanced AI-based quality assessment:

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env` file:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

## 📖 Usage

### Interactive Mode

```bash
python cli.py
```

Choose from the following options:

1. **Best Quality** - Tiered selection for 50% client delivery (full analysis)
2. **Duplicates** - Find and group similar photos
3. **Blurry** - Identify blurry photos (threshold < 25)
4. **Focus Analysis** - Find in-focus (>50) and off-focus (20-50) images
5. **Closed Eyes** - Detect photos with closed eyes
6. **Run All** - Process all operations in optimal order

### Example Workflow

```bash
# Process wedding photos
python cli.py

# Select option 1 (Best Quality)
# Enter input directory: /path/to/wedding/photos
# Enter output directory: /path/to/output

# Results will be organized in:
# output/
#   ├── delivery/         # Client-ready photos (~50% of total)
#   │   ├── highlights/   # Tier 1 portfolio quality (80+)
#   │   ├── ceremony/     # Organized by event segment
#   │   ├── reception/    
#   │   ├── portraits/    
#   │   └── details/      
#   ├── in_focus/         # Sharp, well-focused images
#   ├── duplicates/       # Grouped similar photos
#   ├── closed_eyes/      # Photos with eyes closed
#   ├── poor_quality/     # Below threshold images
#   └── reports/          # Detailed JSON analysis
```

## ⚙️ Configuration

Edit `config.json` to customize processing parameters:

```json
{
    "thresholds": {
        "blur_threshold": 25,
        "focus_threshold": 50,
        "eye_confidence": 50,
        "duplicate_hash_threshold": 50,
        "best_quality_min_score": 45,
        "target_delivery_percentage": 50,
        "tier_1_threshold": 80,
        "tier_2_threshold": 60,
        "tier_3_threshold": 45
    },
    "batch_processing": {
        "enabled": true,
        "batch_size": 25,
        "clear_cache_between_batches": true
    },
    "processing_settings": {
        "batch_size": 8,
        "max_workers": 4,
        "use_gpu": false
    },
    "retry_settings": {
        "enabled": true,
        "max_attempts": 3,
        "initial_delay": 2.0,
        "max_delay": 30.0
    },
    "api_queue_manager": {
        "enabled": true,
        "max_concurrent_requests": 5,
        "requests_per_minute": 60
    },
    "batch_gemini_processing": {
        "enabled": true,
        "batch_size": 5,
        "use_queue_manager": true
    }
}
```

## 🏗️ Architecture

### Optimized Processing Pipeline

```
1. RAW Conversion (parallel)
   ↓
2. Duplicate Detection (face recognition + hashing)
   ↓
3. Parallel Analysis:
   - Blur Detection ←→ Focus Analysis
   ↓
4. Eye Detection (uses face data from step 2)
   ↓
5. Tiered Quality Selection (50% delivery target):
   - Tier 1 (80-100): Portfolio quality
   - Tier 2 (60-79): Delivery quality  
   - Tier 3 (45-59): Coverage quality
   - Coverage analysis for complete event documentation
```

### Tiered Selection Philosophy

The system is designed to deliver approximately 50% of captured photos to ensure clients receive:
- **Complete Event Coverage**: All key moments and people represented
- **Quality Over Quantity**: Technical excellence balanced with storytelling
- **Cohesive Gallery**: Photos that work together to tell the wedding story
- **Solo Shot Guarantee**: Every important person has at least one good photo

### Core Components

- **`src/core/`**: Detection algorithms (blur, focus, eyes, duplicates)
- **`src/services/`**: High-level processing services
- **`src/utils/`**: Utilities (batch processing, parallel execution, error handling)
- **`models/`**: Pre-trained ML models for face and eye detection

## 📊 Output Structure

```
output/
├── best_quality/       # Highest quality photos (top 20%)
├── duplicates/         # Organized by similarity groups
│   ├── group_1/
│   ├── group_2/
│   └── report.json
├── blurry/            # Photos below blur threshold
├── closed_eyes/       # Photos with detected closed eyes
├── in_focus/          # Sharp, well-focused photos
├── off_focus/         # Slightly out-of-focus (salvageable)
├── poor_quality/      # Below quality thresholds
└── reports/           # Detailed analysis reports
    ├── quality_assessment.json
    ├── duplicate_analysis.json
    └── processing_summary.json
```

## 🔧 Advanced Features

### Enhanced Performance Features (v2.0)

**1. Intelligent Retry Logic**
- Automatic retry with exponential backoff for API failures
- Configurable retry attempts and delay settings
- Prevents workflow interruption from transient errors

**2. API Queue Management**
- Rate limiting to respect API quotas (60 requests/minute)
- Concurrent request management with configurable workers
- Request prioritization and queueing

**3. Batch Gemini Processing**
- Process multiple images in a single API call
- Reduces API overhead and improves throughput
- Intelligent caching of resized images

**4. Configuration-Driven Architecture**
- All enhancements configurable via config.json
- Enable/disable features without code changes
- Fine-tune performance for your hardware

### Batch Processing for Large Collections

The system automatically handles large collections in batches:
- Processes 25 images at a time by default
- Clears memory between batches
- Saves progress for resume capability

### Resume Interrupted Processing

If processing is interrupted:
```bash
# The system automatically resumes from the last checkpoint
python cli.py
```

### Custom Processing Workflows

Create custom workflows by combining services:
```python
from src.services import process_duplicates, process_blur
from src.utils import BatchProcessor

# Process only duplicates and blur
results = BatchProcessor().process_in_batches(
    images, 
    lambda batch: {
        'duplicates': process_duplicates(batch),
        'blur': process_blur(batch)
    }
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `batch_size` in config.json
   - Enable `clear_cache_between_batches`

2. **Slow Processing**
   - Check if GPU is detected (if available)
   - Adjust `max_workers` based on CPU cores
   - Use smaller `batch_size` for stability

3. **Model Download Failures**
   - Run `python setup.py` to re-download models
   - Check internet connection and firewall settings

4. **API Errors**
   - Verify Gemini API key in `.env`
   - Check API quota and rate limits

### Debug Mode

Enable detailed logging:
```bash
# Set in .env
LOG_LEVEL=DEBUG
```

Check logs in `logs/` directory for detailed error messages.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/cullingalgorithm.git
cd cullingalgorithm
./setup.sh --venv

# Run tests
python -m pytest tests/

# Code style
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision algorithms
- PyTorch team for deep learning framework
- Google for Gemini AI API
- Face recognition models from [face-recognition](https://github.com/ageitgey/face_recognition)
- The open-source community for various pre-trained models

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/cullingalgorithm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cullingalgorithm/discussions)
- **Email**: support@example.com

---

Built with ❤️ for photographers by photographers