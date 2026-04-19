"""
WSL Main Script - Receive, Process, Forward
This is the brain of the system - receives from Windows, processes, sends to Godot
"""

import socket
import json
import time
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils_wsl import load_config, setup_logging, AIFramework
from src.mapper import SkeletonMapper
from src.bridge import GodotBridge

logger = logging.getLogger(__name__)

def main():
    # Load config
    config = load_config('config/settings.json')
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("KHMER SIGN RECOGNIZER - WSL PROCESSING LAYER")
    logger.info("=" * 60)
    
    # Initialize components
    mapper = SkeletonMapper(config)
    godot_bridge = GodotBridge(config)
    ai_framework = AIFramework(config)
    
    # Setup UDP receiver
    receive_port = config['network']['receive_port']
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", receive_port))
    
    logger.info(f"Listening on 0.0.0.0:{receive_port}")
    logger.info(f"Forwarding to Godot at {config['network']['godot_ip']}:{config['network']['godot_port']}")
    logger.info("Waiting for data from Windows...")
    logger.info("Press Ctrl+C to quit\n")
    
    # Stats
    frame_count = 0
    process_count = 0
    fps_start_time = time.time()
    last_stats_time = time.time()
    
    try:
        while True:
            # Receive data from Windows
            data, addr = sock.recvfrom(65535)
            
            try:
                # Parse JSON
                landmark_data = json.loads(data.decode())
                frame_count += 1
                
                # Process landmarks (normalization + bone mapping)
                processed = mapper.process(landmark_data)
                
                if processed:
                    process_count += 1
                    
                    # Send to Godot
                    godot_bridge.send(processed)
                    
                    # Add to AI buffer (if enabled)
                    if ai_framework.enabled:
                        ai_framework.add_frame(processed['bone_transforms'])
                        
                        # Try prediction
                        prediction = ai_framework.predict()
                        if prediction:
                            logger.info(f"🔮 AI Prediction: {prediction}")
                
                # Print stats every 2 seconds
                current_time = time.time()
                if current_time - last_stats_time >= 2.0:
                    elapsed = current_time - fps_start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    success_rate = (process_count / frame_count * 100) if frame_count > 0 else 0
                    
                    logger.info(f"📊 Frames: {frame_count} | Processed: {process_count} ({success_rate:.1f}%) | "
                              f"FPS: {fps:.1f} | To Godot: {godot_bridge.packets_sent}")
                    
                    if ai_framework.enabled:
                        buffer_status = ai_framework.get_buffer_status()
                        logger.info(f"🧠 AI Buffer: {buffer_status['buffer_size']}/{buffer_status['buffer_capacity']}")
                    
                    last_stats_time = current_time
                
            except json.JSONDecodeError:
                logger.warning(f"⚠️  Invalid JSON from {addr}")
            except Exception as e:
                logger.error(f"❌ Error processing frame: {e}")
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown signal received...")
    
    finally:
        # Cleanup
        sock.close()
        godot_bridge.close()
        
        # Print final stats
        mapper_stats = mapper.get_stats()
        bridge_stats = godot_bridge.get_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total frames received: {frame_count}")
        logger.info(f"Successfully processed: {mapper_stats['processed']}")
        logger.info(f"Processing failures: {mapper_stats['failed']}")
        logger.info(f"Success rate: {mapper_stats['success_rate']:.1f}%")
        logger.info(f"Packets sent to Godot: {bridge_stats['packets_sent']}")
        logger.info(f"Data sent: {bridge_stats['bytes_sent']} bytes ({bridge_stats['bytes_sent']/1024:.1f} KB)")
        logger.info(f"Average packet size: {bridge_stats['avg_packet_size']:.0f} bytes")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()
