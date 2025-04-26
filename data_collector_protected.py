# Standard library imports first - these shouldn't need installation checks
import logging
import sys
import os
import traceback
import socket
import uuid
from datetime import datetime

# Import the installation helper from config
from config import Config

# Now that we've ensured the packages are installed, import them
import paramiko
from smbprotocol.connection import Connection
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect
from smbprotocol.open import (
    Open,
    ImpersonationLevel,
    FilePipePrinterAccessMask,
    FileAttributes,
    ShareAccess,
    CreateDisposition,
    CreateOptions,
    FileInformationClass
)

# Track errors throughout the execution
error_log = []

def log_error(phase, error_msg, trace=None):
    """Log an error message with optional traceback.

    Args:
        phase (_type_): The phase of execution where the error occurred
        error_msg (_type_): The error message
        trace (_type_, optional): Traceback information. Defaults to None.
    """
    error_entry = {
        'phase': phase,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'message': str(error_msg),
        'traceback': trace
    }
    error_log.append(error_entry)
    if trace:
        logging.error(f"{phase}: {error_msg}")
        logging.debug(trace)
    else:
        logging.error(f"{phase}: {error_msg}")

def connect_smb():
    """Connect to the SMB server using smbprotocol library.

    Raises:
        ConnectionError: Logging error if connection fails.

    Returns:
        tuple: Connection, Session, and a dictionary of connected trees
    """
    try:
        logging.info(f"Connecting to SMB server {Config.SMB_HOST}...")
        # Create connection with unique client UUID
        connection = Connection(uuid.uuid4(), Config.SMB_HOST, 445)
        connection.connect()
        
        # Create session with credentials
        session = Session(connection, Config.USERNAME, Config.PASSWORD, require_encryption=True)
        session.connect()
        
        # Keep track of connected trees
        trees = {}
        
        logging.info("SMB Connection successful!")
        return connection, session, trees
    except Exception as e:
        error_msg = str(e)
        # Check for common authentication error indicators
        if any(keyword in error_msg.lower() for keyword in ['authentication', 'login', 'credential', 'password', 'access denied']):
            error_msg = f"{error_msg} - Please run secure/set_credentials.py to ensure credentials are properly configured."
        error_trace = traceback.format_exc()
        log_error("SMB Connection", error_msg, error_trace)
        raise

def connect_sftp():
    """Connect to the SFTP server and return the SFTP client and transport.

    Returns:
        _type_: SFTPClient object
    """
    try:
        logging.info(f"Connecting to SFTP server {Config.SFTP_HOST}...")
        transport = paramiko.Transport((Config.SFTP_HOST, Config.SFTP_PORT))
        transport.connect(username=Config.USERNAME, password=Config.PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        logging.info("SFTP Connection successful!")
        return sftp, transport
    except paramiko.AuthenticationException:
        error_msg = "Authentication failed for SFTP connection. Please run secure/set_credentials.py to ensure credentials are properly configured."
        error_trace = traceback.format_exc()
        log_error("SFTP Connection", error_msg, error_trace)
        raise
    except Exception as e:
        error_msg = str(e)
        # Check for common authentication error indicators
        if any(keyword in error_msg.lower() for keyword in ['authentication', 'login', 'credential', 'password', 'access denied']):
            error_msg = f"{error_msg} - Please run secure/set_credentials.py to ensure credentials are properly configured."
        error_trace = traceback.format_exc()
        log_error("SFTP Connection", error_msg, error_trace)
        raise

def load_downloaded_files(record_file):
    """Load previously downloaded files from a record file.

    Args:
        record_file (_type_): _path to the record file

    Returns:
        _type_: set of downloaded files
    """
    try:
        if os.path.exists(record_file):
            with open(record_file, "r") as f:
                return set(line.strip() for line in f)
        return set()
    except Exception as e:
        log_error("Load Downloaded Files", f"Error loading records from {record_file}: {e}", traceback.format_exc())
        return set()  # Return empty set to continue operation

def save_downloaded_files(record_file, downloaded):
    """Save the list of downloaded files to a record file.

    Args:
        record_file (_type_): _path to the record file
        downloaded (_type_): set of downloaded files
    """
    try:
        with open(record_file, "w") as f:
            for item in downloaded:
                f.write(f"{item}\n")
    except Exception as e:
        log_error("Save Downloaded Files", f"Error saving records to {record_file}: {e}", traceback.format_exc())

def download_smb_files(connection, session, trees):
    """Download files from SMB shares based on configuration using smbprotocol.

    Args:
        connection: smbprotocol Connection object
        session: smbprotocol Session object
        trees: dict to store TreeConnect objects

    Returns:
        tuple: total number of new downloaded files and errors by share
    """
    downloaded_files = load_downloaded_files(Config.SMB_RECORD_FILE)
    total_new_downloaded = 0
    errors_by_share = {}

    for share, prefixes in Config.SHARES_SMB.items():
        share_dir = os.path.join(Config.DATA_DIR, share)
        os.makedirs(share_dir, exist_ok=True)
        share_new_downloads = 0
        
        try:
            logging.info(f"Accessing share {share}...")
            
            # Create tree connection for this share
            share_path = rf"\\{Config.SMB_HOST}\{share}"
            tree = TreeConnect(session, share_path)
            tree.connect()
            trees[share] = tree
            
            # Open root directory for this share
            dir_open = Open(tree, "")
            dir_open.create(
                ImpersonationLevel.Impersonation,
                FilePipePrinterAccessMask.GENERIC_READ,
                FileAttributes.FILE_ATTRIBUTE_DIRECTORY,
                ShareAccess.FILE_SHARE_READ,
                CreateDisposition.FILE_OPEN,
                CreateOptions.FILE_DIRECTORY_FILE
            )
            
            # Query directory for files
            files = dir_open.query_directory("*", FileInformationClass.FILE_DIRECTORY_INFORMATION)
            
            for file_info in files:
                filename = file_info["file_name"].get_value().decode('utf-16-le')
                
                if filename in [".", ".."] or filename in downloaded_files or not filename.endswith(".csv"):
                    continue
                    
                if any(prefix in filename for prefix in prefixes):
                    local_path = os.path.join(share_dir, filename)
                    try:
                        logging.info(f"Downloading {filename} to {local_path}")
                        
                        # Create file open and read file content
                        file_open = Open(tree, filename)
                        file_open.create(
                            ImpersonationLevel.Impersonation,
                            FilePipePrinterAccessMask.GENERIC_READ,
                            FileAttributes.FILE_ATTRIBUTE_NORMAL,
                            ShareAccess.FILE_SHARE_READ,
                            CreateDisposition.FILE_OPEN,
                            CreateOptions.FILE_NON_DIRECTORY_FILE
                        )
                        
                        # Read entire file in one go (alternative to query_file_info which doesn't exist)
                        try:
                            # Start with a small size and increase if needed
                            position = 0
                            buffer_size = 8192
                            with open(local_path, "wb") as local_file:
                                while True:
                                    data = file_open.read(position, buffer_size)
                                    if not data:  # End of file reached
                                        break
                                    local_file.write(data)
                                    position += len(data)
                        except Exception as e:
                            # If specific error about EOF, this is normal
                            if "STATUS_END_OF_FILE" in str(e):
                                pass  # Successfully read all data
                            else:
                                raise  # Re-raise if it's a different error
                        
                        file_open.close()
                        downloaded_files.add(filename)
                        share_new_downloads += 1
                    except Exception as file_error:
                        error_msg = f"Error downloading file {filename}: {file_error}"
                        if share not in errors_by_share:
                            errors_by_share[share] = []
                        errors_by_share[share].append(error_msg)
                        log_error(f"SMB Download - {share}", error_msg, traceback.format_exc())
            
            dir_open.close()
            logging.info(f"Summary for {share}: Downloaded {share_new_downloads} new files.")
            total_new_downloaded += share_new_downloads
        except Exception as e:
            error_msg = f"Error processing share {share}: {e}"
            log_error("SMB Download", error_msg, traceback.format_exc())
            if share not in errors_by_share:
                errors_by_share[share] = []
            errors_by_share[share].append(error_msg)

    save_downloaded_files(Config.SMB_RECORD_FILE, downloaded_files)
    return total_new_downloaded, errors_by_share

def download_sftp_files(sftp):
    """Download files from SFTP shares based on configuration.

    Args:
        sftp (_type_): SFTPClient object

    Returns:
        _type_: total number of new downloaded files and errors by share
    """
    downloaded_files = load_downloaded_files(Config.SFTP_RECORD_FILE)
    total_new_downloaded = 0
    errors_by_share = {}

    for share, prefixes in Config.SHARES_SFTP.items():
        share_dir = os.path.join(Config.DATA_DIR, share)
        os.makedirs(share_dir, exist_ok=True)
        share_new_downloads = 0

        try:
            logging.info(f"Accessing SFTP directory {share}...")
            # Navigate to the share directory
            sftp.chdir(f"/{share}")
            files = sftp.listdir()
            
            for filename in files:
                if filename in downloaded_files or not filename.endswith(".csv"):
                    continue
                    
                # Check if any prefix from configuration matches the filename
                if any(prefix in filename for prefix in prefixes):
                    local_path = os.path.join(share_dir, filename)
                    try:
                        logging.info(f"Downloading {filename} to {local_path}")
                        sftp.get(filename, local_path)
                        downloaded_files.add(filename)
                        share_new_downloads += 1
                    except Exception as file_error:
                        error_msg = f"Error downloading file {filename}: {file_error}"
                        if share not in errors_by_share:
                            errors_by_share[share] = []
                        errors_by_share[share].append(error_msg)
                        log_error(f"SFTP Download - {share}", error_msg, traceback.format_exc())
                    
            logging.info(f"Summary for {share}: Downloaded {share_new_downloads} new files.")
            total_new_downloaded += share_new_downloads
        except Exception as e:
            error_msg = f"Error processing SFTP directory {share}: {e}"
            log_error("SFTP Download", error_msg, traceback.format_exc())
            if share not in errors_by_share:
                errors_by_share[share] = []
            errors_by_share[share].append(error_msg)

    save_downloaded_files(Config.SFTP_RECORD_FILE, downloaded_files)
    return total_new_downloaded, errors_by_share

def send_error_summary():
    """Send an email summarizing all errors encountered during execution
    """
    if not error_log:
        return
    
    # Prepare error message for email
    summary_msg = "The following errors occurred during data collection:\n\n"
    for error in error_log:
        summary_msg += f"- Phase: {error['phase']}\n"
        summary_msg += f"  Time: {error['time']}\n"
        summary_msg += f"  Message: {error['message']}\n\n"
    
    # Get the first error's traceback for detailed information
    first_traceback = error_log[0].get('traceback', None) if error_log else None
    
    try:
        Config.send_error_email(
            module_name="Data Collector",
            subject=f"Data Collection Errors ({len(error_log)} issues)",
            error_message=summary_msg,
            traceback_info=first_traceback
        )
        logging.info("Error summary email sent successfully.")
    except Exception as email_error:
        logging.error(f"Failed to send error summary email: {email_error}")

def main():
    """Main function to execute the data collection process.
    """
    smb_conn = None
    smb_session = None
    smb_trees = {}
    sftp_client = None
    transport = None
    critical_error = False
    
    try:
        # First validate configuration 
        Config.setup("data_collector")
        
        logging.info(f"Starting data collection for {Config.current_date}")
        logging.info(f"Data will be saved in: {Config.DATA_DIR}")

        # Track results to provide a comprehensive summary
        results = {
            "smb_files": 0,
            "sftp_files": 0,
            "smb_errors": {},
            "sftp_errors": {}
        }

        # Process SMB files
        try:
            smb_conn, smb_session, smb_trees = connect_smb()
            results["smb_files"], results["smb_errors"] = download_smb_files(smb_conn, smb_session, smb_trees)
        except Exception as e:
            log_error("SMB Processing", f"Failed to complete SMB file downloads: {e}", traceback.format_exc())
        
        # Process SFTP files
        try:
            sftp_client, transport = connect_sftp()
            results["sftp_files"], results["sftp_errors"] = download_sftp_files(sftp_client)
        except Exception as e:
            log_error("SFTP Processing", f"Failed to complete SFTP file downloads: {e}", traceback.format_exc())

        total_new_files = results["smb_files"] + results["sftp_files"]
        
        # Determine completion status
        if not error_log:
            logging.info("Data collection completed successfully!")
        else:
            logging.warning("Data collection completed with errors. Check logs for details.")
        
        logging.info(f"Total new downloaded files: {total_new_files}")
        
    except Exception as e:
        critical_error = True
        error_msg = f"Fatal error in data collection: {e}"
        log_error("Main Execution", error_msg, traceback.format_exc())
    finally:
        # Close connections
        if smb_conn:
            try:
                # Close tree connections first
                for tree in smb_trees.values():
                    try:
                        tree.disconnect()
                    except Exception:
                        pass
                
                # Close session
                if smb_session:
                    try:
                        smb_session.disconnect()
                    except Exception:
                        pass
                
                # Close connection
                smb_conn.disconnect(True)
                logging.debug("SMB connections closed")
            except Exception:
                pass
                
        if sftp_client:
            try:
                sftp_client.close()
                logging.debug("SFTP client closed")
            except Exception:
                pass
                
        if transport:
            try:
                transport.close()
                logging.debug("SFTP transport closed")
            except Exception:
                pass
        
        # Send error summary email if there were any errors
        if error_log:
            send_error_summary()
        
        if critical_error:
            sys.exit(1)  # More standard than raising an exception in finally
            
if __name__ == "__main__":
    main()