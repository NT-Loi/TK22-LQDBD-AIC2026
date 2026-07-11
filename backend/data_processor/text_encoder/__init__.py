import logging
import sys

# Configure logging to output to both console and a file called 'app.log'
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from .base import TextEncoder
from .clip import CLIPTextEncoder
from .siglip import SigLIPTextEncoder

__all__ = ["TextEncoder", "CLIPTextEncoder", "SigLIPTextEncoder"]