# Wedding Photo Culling Assistant

An intelligent photo culling system designed to help photographers efficiently sort through large collections of wedding photos. The system uses advanced computer vision and AI techniques to identify the best quality images while filtering out blurry, duplicate, or poorly composed photos.

## Features

- **RAW File Support**: Automatically converts and processes RAW files (ARW, CR2, NEF, RAF, ORF)
- **Duplicate Detection**: Identifies and groups similar/duplicate photos using perceptual hashing
- **Blur Detection**: Uses Laplacian and FFT analysis to identify blurry images
- **Focus Analysis**: Detects in-focus and off-focus areas in images
- **Eye Detection**: Identifies photos with closed eyes using machine learning
- **Quality Assessment**: AI-powered analysis for overall photo quality
- **Batch Processing**: Efficient processing of large photo collections
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Installation

### Prerequisites

- Python 3.9 or higher
- 8GB+ RAM recommended
- Optional: CUDA-capable GPU for faster processing

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
# Install dependencies
pip install -r requirements.txt

# Download required models
python setup.py

# Create .env file and add your Gemini API key (optional)
cp .env.example .env
```

## Usage

### Interactive Mode

```bash
python cli.py
```

Choose from the following options:
1. **Best Quality** - Comprehensive analysis to identify the highest quality photos
2. **Duplicates** - Find and group similar photos
3. **Blurry** - Identify blurry photos
4. **Focus Analysis** - Find in-focus and off-focus images
5. **Closed Eyes** - Detect photos with closed eyes
6. **Run All** - Process all operations in optimal order

### Command Line (Coming Soon)

```bash
# Process best quality photos
python cli.py --mode best --input /path/to/photos --output /path/to/output

# Find duplicates only
python cli.py --mode duplicates --input /path/to/photos --output /path/to/output
```

## Configuration

Edit `config.json` to customize thresholds and processing settings:

```json
{
    "thresholds": {
        "blur_threshold": 25,
        "focus_threshold": 50,
        "eye_confidence": 50,
        "duplicate_hash_threshold": 50
    },
    "processing_settings": {
        "batch_size": 16,
        "max_workers": 4,
        "use_gpu": false
    }
}
```

## API Keys

For enhanced AI-based quality assessment, add your Gemini API key to `.env`:

```
GEMINI_API_KEY=your-api-key-here
```

Get your API key from: https://makersuite.google.com/app/apikey

## Output Structure

```
output/
├── best_quality/       # Highest quality photos
├── duplicates/         # Grouped duplicate photos
├── blurry/            # Blurry photos
├── closed_eyes/       # Photos with closed eyes
├── in_focus/          # Sharp, in-focus photos
├── off_focus/         # Slightly out-of-focus photos
└── reports/           # JSON reports with detailed analysis
```

## Performance Tips

1. **Use GPU acceleration** if available (automatically detected)
2. **Process in batches** for large collections
3. **Adjust worker count** based on your CPU cores
4. **Enable caching** to avoid reprocessing

## Troubleshooting

### Common Issues

1. **Missing models**: Run `python setup.py` to download all required models
2. **Memory errors**: Reduce batch_size in config.json
3. **Slow processing**: Enable GPU or reduce max_workers for stability
4. **API errors**: Check your Gemini API key and quota

### Logs

Check `logs/` directory for detailed error messages and processing logs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision algorithms
- PyTorch for deep learning models
- Google Gemini for AI-powered analysis
- The open-source community for various pre-trained models