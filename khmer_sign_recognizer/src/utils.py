"""
Windows Layer - Utility Functions
"""

import json
import logging
from pathlib import Path
from typing import Dict

def load_config(config_path: str = "config/settings.json") -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(config: Dict) -> logging.Logger:
    """Configure logging based on config"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Create logs directory if it doesn't exist
    log_file = Path(log_config.get('file', 'logs/app.log'))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    handlers = [logging.FileHandler(log_file)]
    if log_config.get('console', True):
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)