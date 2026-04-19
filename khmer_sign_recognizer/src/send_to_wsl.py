"""
Windows Layer - UDP Bridge to WSL
"""

import socket
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class WSLBridge:
    """
    Sends landmark data to WSL over UDP
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.net_config = config['network']
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.wsl_address = (
            self.net_config['wsl_ip'],
            self.net_config['wsl_port']
        )
        
        self.packets_sent = 0
        logger.info(f"WSLBridge initialized - Target: {self.wsl_address}")
    
    def send(self, landmark_data: Dict) -> bool:
        """Send landmark data to WSL"""
        try:
            # Convert to JSON
            json_data = json.dumps(landmark_data)
            
            # Send over UDP
            self.sock.sendto(json_data.encode(), self.wsl_address)
            
            self.packets_sent += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get bridge statistics"""
        return {
            'packets_sent': self.packets_sent,
            'target': f"{self.wsl_address[0]}:{self.wsl_address[1]}"
        }
    
    def close(self):
        """Close socket"""
        self.sock.close()
        logger.info("WSLBridge closed")