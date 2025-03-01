import sys
import logging
import os
from datetime import datetime
import traceback
from pathlib import Path
import socket
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import from config file
from config import ensure_installed, Config

# Ensure required packages are installed.
ensure_installed('pysmb')
ensure_installed('paramiko')
ensure_installed('python-dotenv')
ensure_installed('chardet')

from smb.SMBConnection import SMBConnection
import paramiko

# Remove any existing handlers so that the logger can be reconfigured
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def setup_logging():
    # The log file will use the name of the data folder inside the LOG_DIR.
    log_filename = f"{os.path.basename(Config.DATA_DIR)}.log"
    log_file = os.path.join(Config.LOG_DIR, log_filename)
    # Reconfigure the logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

def connect_smb():
    try:
        logging.info(f"Connecting to SMB server {Config.SMB_HOST}...")
        conn = SMBConnection(
            Config.USERNAME,
            Config.PASSWORD,
            socket.gethostname(),
            Config.SMB_HOST,
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        if conn.connect(Config.SMB_HOST, 445) or conn.connect(Config.SMB_HOST, 139):
            logging.info("SMB Connection successful!")
            return conn
        raise ConnectionError("Failed to establish SMB connection.")
    except Exception as e:
        logging.error(f"SMB connection error: {e}")
        raise

def connect_sftp():
    try:
        logging.info(f"Connecting to SFTP server {Config.SFTP_HOST}...")
        transport = paramiko.Transport((Config.SFTP_HOST, Config.SFTP_PORT))
        transport.connect(username=Config.USERNAME, password=Config.PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        logging.info("SFTP Connection successful!")
        return sftp, transport
    except Exception as e:
        logging.error(f"SFTP connection error: {e}")
        raise

def load_downloaded_files(record_file):
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            return set(line.strip() for line in f)
    return set()

def save_downloaded_files(record_file, downloaded):
    with open(record_file, "w") as f:
        for item in downloaded:
            f.write(f"{item}\n")

def download_smb_files(conn):
    shares = {
        "Solarlogs": ["PV", "min"],
        "BellevueConso": ["Consumption", "Temperature", "Humidity"],
        "BellevueBooking": ["RoomAllocations"]
    }
    downloaded_files = load_downloaded_files(Config.SMB_RECORD_FILE)
    total_new_downloaded = 0

    for share, prefixes in shares.items():
        share_dir = os.path.join(Config.DATA_DIR, share)
        os.makedirs(share_dir, exist_ok=True)
        share_new_downloads = 0

        try:
            logging.info(f"Accessing share {share}...")
            files = conn.listPath(share, "/")
            for file in files:
                filename = file.filename
                if filename in [".", ".."] or filename in downloaded_files or not filename.endswith(".csv"):
                    continue
                if any(prefix in filename for prefix in prefixes):
                    local_path = os.path.join(share_dir, filename)
                    logging.info(f"Downloading {filename} to {local_path}")
                    with open(local_path, "wb") as local_file:
                        conn.retrieveFile(share, filename, local_file)
                    downloaded_files.add(filename)
                    share_new_downloads += 1
            logging.info(f"Summary for {share}: Downloaded {share_new_downloads} new files.")
            total_new_downloaded += share_new_downloads
        except Exception as e:
            logging.error(f"Error processing share {share}: {e}")
            logging.debug(traceback.format_exc())

    save_downloaded_files(Config.SMB_RECORD_FILE, downloaded_files)
    return total_new_downloaded

def download_sftp_files(sftp):
    downloaded_files = load_downloaded_files(Config.SFTP_RECORD_FILE)
    new_downloaded_count = 0
    meteo_dir = "Meteo"
    local_meteo_dir = os.path.join(Config.DATA_DIR, meteo_dir)
    os.makedirs(local_meteo_dir, exist_ok=True)
    
    try:
        logging.info(f"Accessing SFTP directory {meteo_dir}...")
        sftp.chdir(f"/{meteo_dir}")
        files = sftp.listdir()
        for filename in files:
            if filename.startswith("Pred_") and filename.endswith(".csv"):
                if filename not in downloaded_files:
                    local_path = os.path.join(local_meteo_dir, filename)
                    logging.info(f"Downloading {filename} to {local_path}")
                    sftp.get(filename, local_path)
                    downloaded_files.add(filename)
                    new_downloaded_count += 1
        logging.info(f"Summary for {meteo_dir}: Downloaded {new_downloaded_count} new files.")
    except Exception as e:
        logging.error(f"Error processing directory {meteo_dir}: {e}")
        logging.debug(traceback.format_exc())

    save_downloaded_files(Config.SFTP_RECORD_FILE, downloaded_files)
    return new_downloaded_count

def main():
    try:
        # First validate configuration 
        Config.validate()
        
        # Configure logging after validation to ensure directories exist
        setup_logging()
        
        logging.info(f"Starting data collection for {Config.current_date}")
        logging.info(f"Data will be saved in: {Config.DATA_DIR}")

        smb_conn = connect_smb()
        smb_new = download_smb_files(smb_conn)
        smb_conn.close()

        sftp_client, transport = connect_sftp()
        sftp_new = download_sftp_files(sftp_client)
        sftp_client.close()
        transport.close()

        total_new_files = smb_new + sftp_new
        logging.info("Data collection completed successfully!")
        logging.info(f"Total new downloaded files: {total_new_files}")
    except Exception as e:
        error_msg = f"Fatal error in data collection: {e}"
        error_traceback = traceback.format_exc()
        logging.error(error_msg)
        logging.debug(error_traceback)
        
        # Send email notification about the error using the Config class method
        Config.send_error_email(
            module_name="Data Collection",
            subject="Data Collection Failure",
            error_message=str(e),
            traceback_info=error_traceback
        )
        raise

if __name__ == "__main__":
    main()