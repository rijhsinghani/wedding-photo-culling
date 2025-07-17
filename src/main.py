import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add at the start of your code
import torchvision
torchvision.disable_beta_transforms_warning()
# Standard library imports
import os
from typing import Dict

import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
mp.set_start_method('spawn', force=True)
import logging
import sys
from rich.table import Table
from rich.console import Console

# Third-party imports

from skimage.metrics import structural_similarity as ssim

from dotenv import load_dotenv
load_dotenv()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Modified logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add log filter
class LogFilter(logging.Filter):
    def filter(self, record):
        # Only allow important logs
        if record.levelno >= logging.WARNING:
            return True
        
        # Filter out specific messages
        important_phrases = [
            'Error',
            'Failed',
            'Processing complete',
            'Starting analysis',
            'Completed analysis'
        ]
        return any(phrase in record.msg for phrase in important_phrases)

logger = logging.getLogger(__name__)
logger.addFilter(LogFilter())

# Suppress verbose third-party logs
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

def log_critical(msg):
    logger.info(msg)




def setup_logging(output_dir: str):
    """Configure logging with reduced verbosity."""
    log_file = os.path.join(output_dir, 'processing.log')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # Only show warnings and errors in console
            logging.StreamHandler(sys.stderr)
        ]
    )
    
    # Suppress verbose logs
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


def print_detailed_summary(results: Dict):
    """Print simplified processing summary."""
    console = Console()
    
    console.print("\n[bold cyan]=============== Photo Distribution Summary ===============[/]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("📸 Category", style="cyan")
    table.add_column("Count", style="yellow", justify="right")
    
    # Get the counts
    best_quality = len(results.get('processing_results', {}).get('quality_results', {}).get('best_quality', []))
    duplicates = results['stats'].get('duplicate_groups', 0)
    closed_eyes = len(results.get('processing_results', {}).get('eyes_results', {}).get('closed_eyes', []))
    blurry = len(results.get('processing_results', {}).get('blur_results', {}).get('blurry', []))
    focus_issues = len(results.get('processing_results', {}).get('focus_results', {}).get('out_focus', []))
    quality_issues = len(results.get('processing_results', {}).get('quality_results', {}).get('rejected', []))
    
    # Add rows to table
    table.add_row("Best Quality", str(best_quality))
    table.add_row("Duplicates", str(duplicates))
    table.add_row("Closed Eyes", str(closed_eyes))
    table.add_row("Poor Quality", str(quality_issues))
    table.add_row("Blurry", str(blurry))
    table.add_row("Focus Issues", str(focus_issues))
    
    # Calculate total
    total = best_quality + duplicates + closed_eyes + blurry + focus_issues + quality_issues
    
    console.print(table)
    console.print(f"\n[bold cyan]Total Photos Processed: {total}[/]")
    console.print("\n[bold cyan]================================================[/]\n")


# If running directly
if __name__ == "__main__":
    print("This module should be imported and used through cli.py")
