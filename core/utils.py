import logging
from dataclasses import dataclass
from typing import List

@dataclass
class ProcessingConfig:
    """Configuration settings for processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 3
    temperature: float = 0.4
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_images: int = 10
    supported_mime_types: List[str] = None

    def __post_init__(self):
        if self.supported_mime_types is None:
            self.supported_mime_types = ["image/jpeg", "image/png"]

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)