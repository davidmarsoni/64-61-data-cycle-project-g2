import importlib.util
import subprocess
import sys
import logging
import os
from datetime import datetime

# Remove any existing handlers so that the logger can be reconfigured
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def ensure_installed(package_name):
    if importlib.util.find_spec(package_name) is None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install {package_name}: {e}")
            sys.exit(1)

# Ensure required packages are installed.
ensure_installed('pysmb')
ensure_installed('paramiko')
ensure_installed('python-dotenv')
ensure_installed('smtplib')  # For email functionality

from smb.SMBConnection import SMBConnection
import paramiko
import socket
import traceback
from pathlib import Path
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Holds the configuration details for SMB and SFTP connections,
    including directories for data storage and logging.
    """
    SMB_HOST = os.getenv('SMB_HOST')
    SFTP_HOST = os.getenv('SFTP_HOST')
    SFTP_PORT = int(os.getenv('SFTP_PORT'))
    USERNAME = os.getenv('DATA_USERNAME')
    PASSWORD = os.getenv('DATA_PASSWORD')

    print(SMB_HOST)
    print(SFTP_HOST)
    print(SFTP_PORT)
    print(USERNAME)
    print(PASSWORD)
    
    
    BASE_DIR = os.getenv('BASE_DIR')
    LOG_DIR = os.getenv(BASE_DIR, "logs")
    current_date = datetime.now().strftime('%Y-%m-%d')
    DATA_DIR = os.path.join(BASE_DIR, f"collected_data_{current_date}")
    
    # Central record files (located in the BASE_DIR, not date-based)
    SMB_RECORD_FILE = os.path.join(BASE_DIR, "downloaded_smb_files.txt")
    SFTP_RECORD_FILE = os.path.join(BASE_DIR, "downloaded_sftp_files.txt")
    
    # Email configuration
    GMAIL_SENDER = os.getenv('GOOGLE_EMAIL_SENDER')
    GMAIL_PASSWORD = os.getenv('APP_PASSWORD_GOOGLE_PASSWORD')
    GMAIL_RECIPIENT = os.getenv('GOOGLE_EMAIL_DESTINATOR')
    
    @classmethod
    def validate(cls):
        if not all([cls.USERNAME, cls.PASSWORD]):
            raise ValueError("Missing DATA_USERNAME or DATA_PASSWORD in .env file")
        if not all([cls.GMAIL_SENDER, cls.GMAIL_PASSWORD, cls.GMAIL_RECIPIENT]):
            raise ValueError("Missing email configuration in .env file")
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)

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

def send_error_email(subject, error_message, traceback_info=None):
    """
    Send an error notification email using Gmail SMTP.
    
    Args:
        subject: Email subject
        error_message: Main error message
        traceback_info: Optional traceback information
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = Config.GMAIL_SENDER
        msg['To'] = Config.GMAIL_RECIPIENT
        msg['Subject'] = f"Data Collection Error: {subject}"
        
        body = f"""
        <html>
          <body>
            <h2>Data Collection Error Report</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Error Message:</strong> {error_message}</p>
            <p><strong>Data Collection Directory:</strong> {Config.DATA_DIR}</p>
        """
        
        if traceback_info:
            body += f"""
            <h3>Error Details:</h3>
            <pre>{traceback_info}</pre>
            """
            
        body += """
          </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(Config.GMAIL_SENDER, Config.GMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        logging.info("Error notification email sent successfully")
    except Exception as e:
        logging.error(f"Failed to send error email: {e}")

def main():
    try:
        Config.validate()
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
        
        # Send email notification about the error
        send_error_email(
            subject="Data Collection Failure",
            error_message=str(e),
            traceback_info=error_traceback
        )
        raise

if __name__ == "__main__":
    main()