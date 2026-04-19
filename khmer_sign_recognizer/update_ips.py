import subprocess
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("IPUpdater")

def get_wsl_ip():
    """Get the IP address of the WSL2 virtual machine"""
    try:
        # Run hostname -I inside WSL
        result = subprocess.run(["wsl", "hostname", "-I"], capture_output=True, text=True, check=True)
        ips = result.stdout.strip().split()
        if ips:
            return ips[0]
        return None
    except Exception as e:
        logger.error(f"Could not get WSL IP: {e}")
        return None

def get_windows_ip_from_wsl():
    """Get the IP address of the Windows host from WSL's perspective"""
    try:
        # The Windows host is always the default gateway for WSL2
        result = subprocess.run(["wsl", "ip", "route"], capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if line.startswith("default via"):
                parts = line.split(" ")
                if len(parts) >= 3:
                    return parts[2]
        return None
    except Exception as e:
        logger.error(f"Could not get Windows IP from WSL route: {e}")
        return None

def main():
    logger.info("Auto-discovering WSL and Windows IP addresses...")
    
    wsl_ip = get_wsl_ip()
    win_ip = get_windows_ip_from_wsl()
    
    if not wsl_ip or not win_ip:
        logger.error("Failed to auto-discover IP addresses! Make sure WSL is running.")
        sys.exit(1)
        
    logger.info(f"✅ Found WSL IP: {wsl_ip}")
    logger.info(f"✅ Found Windows Host IP (from WSL): {win_ip}")
    
    # Update settings.json
    config_path = Path(__file__).parent / "config" / "settings.json"
    
    if not config_path.exists():
        logger.error(f"Could not find {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Update specifically the network ips
        old_wsl = config['network'].get('wsl_ip')
        old_godot = config['network'].get('godot_ip')
        
        config['network']['wsl_ip'] = wsl_ip
        config['network']['godot_ip'] = win_ip
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"✅ Successfully updated {config_path}")
        logger.info(f"   - wsl_ip: {old_wsl} -> {wsl_ip}")
        logger.info(f"   - godot_ip: {old_godot} -> {win_ip}")
        logger.info("\n🎉 Network routing fixed! You can now start the pipeline:")
        logger.info("1. Start Godot")
        logger.info("2. WSL Terminal: python src/main_wsl.py")
        logger.info("3. Windows Terminal: python run_windows.py")
        logger.info("(Run this update_ips.py script anytime you reboot your PC!)")
        
    except Exception as e:
        logger.error(f"Failed to update settings.json: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
