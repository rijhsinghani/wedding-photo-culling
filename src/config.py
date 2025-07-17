
import sys
import logging
import multiprocessing

# Modified logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    multiprocessing.set_start_method('fork')  # Fix for macOS
except:
    pass

try:
    multiprocessing.active_children()

    # multiprocessing.resource_tracker.unregister("/dev/shm", "semaphore")
except:
    pass


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
