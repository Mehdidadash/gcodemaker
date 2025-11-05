import subprocess
import os
import logging

# Configure logging for detailed debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
SMB_SERVER_IP = "192.168.1.100"
SMB_USERNAME = "DTA-image"
SMB_PASSWORD = "6783"
SMB_SHARE = "ImageProcess"
FILE_NAME = "all_raw_results.csv"
OUTPUT_PATH = "/tmp/all_raw_results.csv"

def run_smbclient():
    try:
        # Construct smbclient command with verbose output
        cmd = [
            "smbclient",
            f"//{SMB_SERVER_IP}/{SMB_SHARE}",
            "-U", f"{SMB_USERNAME}%{SMB_PASSWORD}",
            "-c", f"get {FILE_NAME} {OUTPUT_PATH}",
            "-d", "1"  # Enable minimal debug output for smbclient
        ]
        logging.debug(f"Running command: {' '.join(cmd)}")
        print(f"Retrieving {FILE_NAME} from {SMB_SERVER_IP}/{SMB_SHARE}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"smbclient output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"smbclient failed: {e}")
        print(f"smbclient error: {e}")
        print(f"stderr: {e.stderr.strip()}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return False

def main():
    try:
        # Remove existing output file to avoid stale data
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
            logging.debug(f"Removed existing file: {OUTPUT_PATH}")

        # Run smbclient to retrieve the file
        if not run_smbclient():
            return

        # Verify and read the file
        if os.path.exists(OUTPUT_PATH):
            file_size = os.path.getsize(OUTPUT_PATH)
            print(f"Retrieved {FILE_NAME}, size: {file_size} bytes")
            with open(OUTPUT_PATH, "rb") as f:
                data = f.read(200)
                print("File preview:", data)
            # Optional: Read as text for CSV processing
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                text_preview = f.read(200)
                print("Text preview:", text_preview)
        else:
            print(f"Error: {OUTPUT_PATH} was not created")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()