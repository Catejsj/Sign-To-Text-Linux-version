"""
Windows Main Script - Capture and Send to WSL
"""

import sys
import time
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, setup_logging
from capture import LandmarkCapture
from send_to_wsl import WSLBridge

def main():
    # Load config
    config = load_config('config/settings.json')
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("KHMER SIGN RECOGNIZER - WINDOWS LAYER")
    logger.info("=" * 60)
    
    # Initialize components
    capture = LandmarkCapture(config)
    bridge = WSLBridge(config)
    
    # Start capture
    if not capture.start():
        logger.error("Failed to start capture!")
        return
    
    logger.info("System running.")
    logger.info("Controls:")
    logger.info("  Q - Quit")
    logger.info("  F - Toggle fullscreen")
    logger.info("  + - Increase window size")
    logger.info("  - - Decrease window size")
    
    try:
        fps_start_time = time.time()
        fps_frame_count = 0
        consecutive_failures = 0
        MAX_FAILURES = 30  # only show NOT CAPTURING after 30 missed frames
        
        while True:
            # Read frame
            ret, frame = capture.read_frame()
            if not ret:
                logger.warning("Failed to read frame, continuing...")
                time.sleep(0.1)
                continue
            
            # Process landmarks
            landmarks = capture.process_landmarks(frame)
            
            if landmarks:
                consecutive_failures = 0
                # Send to WSL
                bridge.send(landmarks)
                status_text = "CAPTURING"
                status_color = (0, 255, 0)
            else:
                consecutive_failures += 1
                if consecutive_failures < MAX_FAILURES:
                    status_text = "CAPTURING"
                    status_color = (0, 255, 0)
                else:
                    status_text = "NOT CAPTURING"
                    status_color = (0, 0, 255)
            
            # Display frame with status
            cv2.putText(frame, status_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps = fps_frame_count / (time.time() - fps_start_time)
                logger.info(f"FPS: {fps:.1f} | Sent: {bridge.packets_sent} packets")
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Display stats
            cv2.putText(frame, f"Packets: {bridge.packets_sent}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Resize frame for display if needed
            display_frame = frame
            if 'window_scale' in config['capture']:
                scale = config['capture']['window_scale']
                if scale != 1.0:
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    display_frame = cv2.resize(frame, (new_width, new_height))
            
            # Show frame
            cv2.namedWindow('Windows Capture Layer - Press Q to quit', cv2.WINDOW_NORMAL)
            cv2.imshow('Windows Capture Layer - Press Q to quit', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quit signal received")
                break
            elif key == ord('f'):
                # Toggle fullscreen
                cv2.setWindowProperty('Windows Capture Layer - Press Q to quit', 
                                     cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_FULLSCREEN)
            elif key == ord('n'):
                # Normal window
                cv2.setWindowProperty('Windows Capture Layer - Press Q to quit', 
                                     cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_NORMAL)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    
    finally:
        # Cleanup
        logger.info("Shutting down...")
        capture.stop()
        bridge.close()
        cv2.destroyAllWindows()
        
        # Print final stats
        cap_stats = capture.get_stats()
        bridge_stats = bridge.get_stats()
        
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info(f"Total frames: {cap_stats['total_frames']}")
        logger.info(f"Packets sent: {bridge_stats['packets_sent']}")
        logger.info(f"Target: {bridge_stats['target']}")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()