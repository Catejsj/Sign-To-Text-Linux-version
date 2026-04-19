"""
WSL Layer - UDP Bridge to Godot
Sends processed bone transform data to Godot for visualization
"""

import socket
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class GodotBridge:
    """
    Sends bone transform data to Godot over UDP
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.net_config = config['network']
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.godot_address = (
            self.net_config['godot_ip'],
            self.net_config['godot_port']
        )
        
        self.packets_sent = 0
        self.bytes_sent = 0
        
        logger.info(f"GodotBridge initialized - Target: {self.godot_address}")
    
    def send(self, bone_data: Dict) -> bool:
        """Send bone transform data to Godot"""
        try:
            # Convert to JSON
            json_data = json.dumps(bone_data)
            json_bytes = json_data.encode()
            
            # Send over UDP
            self.sock.sendto(json_bytes, self.godot_address)
            
            self.packets_sent += 1
            self.bytes_sent += len(json_bytes)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get bridge statistics"""
        avg_packet_size = (self.bytes_sent / self.packets_sent) if self.packets_sent > 0 else 0
        
        return {
            'packets_sent': self.packets_sent,
            'bytes_sent': self.bytes_sent,
            'avg_packet_size': avg_packet_size,
            'target': f"{self.godot_address[0]}:{self.godot_address[1]}"
        }
    
    def close(self):
        """Close socket"""
        self.sock.close()
        logger.info("GodotBridge closed")
