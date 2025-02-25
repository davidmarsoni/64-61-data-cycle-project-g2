import os
from datetime import datetime
import logging
from smb.SMBConnection import SMBConnection
import paramiko
import socket
import traceback
import stat
from pathlib import Path

# Configuration
class Config:
    # Get credentials from environment variables
    SMB_HOST = os.environ.get('SMB_HOST', '10.130.25.152')
    SFTP_HOST = os.environ.get('SFTP_HOST', '10.130.25.152')
    SFTP_PORT = int(os.environ.get('SFTP_PORT', '22'))
    USERNAME = os.environ.get('DATA_USERNAME')
    PASSWORD = os.environ.get('DATA_PASSWORD')
    
    # Windows paths
    BASE_DIR = r"C:\DataCollection"
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    current_date = datetime.now().strftime('%Y%m%d')
    DATA_DIR = os.path.join(BASE_DIR, f"collected_data_{current_date}")

    @classmethod
    def validate(cls):
        """Validate configuration and create necessary directories"""
        if not all([cls.USERNAME, cls.PASSWORD]):
            raise ValueError("Missing credentials in environment variables (DATA_USERNAME, DATA_PASSWORD)")
        
        # Create necessary directories
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)

# Configure logging
def setup_logging():
    log_file = os.path.join(Config.LOG_DIR, f"collector_{datetime.now().strftime('%Y%m%d')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

def connect_smb():
    """Establishes an SMB connection with error handling"""
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
        else:
            raise ConnectionError("Failed to establish SMB connection")
            
    except Exception as e:
        logging.error(f"SMB connection error: {str(e)}")
        raise

def connect_sftp():
    """Establishes an SFTP connection"""
    try:
        logging.info(f"Connecting to SFTP server {Config.SFTP_HOST}...")
        transport = paramiko.Transport((Config.SFTP_HOST, Config.SFTP_PORT))
        transport.connect(username=Config.USERNAME, password=Config.PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        logging.info("SFTP Connection successful!")
        return sftp, transport
    except Exception as e:
        logging.error(f"SFTP connection error: {str(e)}")
        raise

def download_smb_files(conn):
    """Downloads files from SMB shares"""
    shares = {
        "Solarlogs": ["PV", "min"],
        "BellevueConso": ["Consumption", "Temperature", "Humidity"],
        "BellevueBooking": ["RoomAllocations"]
    }
    
    for share, prefixes in shares.items():
        share_dir = os.path.join(Config.DATA_DIR, share)
        os.makedirs(share_dir, exist_ok=True)
        
        try:
            logging.info(f"Accessing share {share}...")
            files = conn.listPath(share, "/")
            
            for file in files:
                filename = file.filename
                if filename in [".", ".."]:
                    continue
                    
                if any(prefix in filename for prefix in prefixes):
                    local_path = os.path.join(share_dir, filename)
                    
                    logging.info(f"Downloading {filename} to {local_path}")
                    with open(local_path, "wb") as local_file:
                        conn.retrieveFile(share, filename, local_file)
                        
        except Exception as e:
            logging.error(f"Error processing share {share}: {str(e)}")
            logging.debug(traceback.format_exc())

def download_sftp_files(sftp):
    """Downloads weather forecast files from all Meteo directories"""
    try:
        logging.info("Listing SFTP directories...")
        all_items = sftp.listdir_attr()
        meteo_dirs = [
            item.filename for item in all_items 
            if item.filename.startswith('Meteo') and stat.S_ISDIR(item.st_mode)
        ]
        logging.info(f"Found Meteo directories: {meteo_dirs}")
        
        for meteo_dir in meteo_dirs:
            local_meteo_dir = os.path.join(Config.DATA_DIR, meteo_dir)
            os.makedirs(local_meteo_dir, exist_ok=True)
            
            try:
                sftp.chdir(f"/{meteo_dir}")
                files = sftp.listdir()
                
                for filename in files:
                    if filename.startswith("Pred_") and filename.endswith(".csv"):
                        local_path = os.path.join(local_meteo_dir, filename)
                        logging.info(f"Downloading {filename} to {local_path}")
                        sftp.get(filename, local_path)
                
                sftp.chdir("/")
                
            except Exception as e:
                logging.error(f"Error processing directory {meteo_dir}: {str(e)}")
                logging.debug(traceback.format_exc())
                continue
                
    except Exception as e:
        logging.error(f"SFTP error: {str(e)}")
        logging.debug(traceback.format_exc())

def main():
    """Main execution function"""
    try:
        # Initialize configuration and logging
        Config.validate()
        setup_logging()
        
        logging.info(f"Starting data collection for {Config.current_date}")
        logging.info(f"Data will be saved in: {Config.DATA_DIR}")
        
        # SMB downloads
        smb_conn = connect_smb()
        download_smb_files(smb_conn)
        smb_conn.close()
        
        # SFTP downloads
        sftp_client, transport = connect_sftp()
        download_sftp_files(sftp_client)
        sftp_client.close()
        transport.close()
        
        logging.info("Data collection completed successfully!")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()