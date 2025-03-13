import importlib.util
import subprocess
import sys
import logging
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up basic console logging first so we can see any installation messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def ensure_installed(package_name):
    """
    Ensures that a Python package is installed, installing it if necessary.
    
    Args:
        package_name: Name of the package to check and install
    """
    if importlib.util.find_spec(package_name) is None:
        try:
            logging.info(f"Installing missing package: {package_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            logging.info(f"Package {package_name} installed successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install {package_name}: {e}")
            sys.exit(1)

# Ensure dotenv is installed before we try to import it (for fallback)
ensure_installed('python-dotenv')
from dotenv import load_dotenv

# Ensure keyring is installed for Windows Credential Manager
ensure_installed('keyring')
import keyring

# Load environment variables from .env file (for non-sensitive config)
load_dotenv()

class CredentialManager:
    """Manage credentials using Windows Credential Manager"""
    
    # Service name for storing credentials in Windows Credential Manager
    SERVICE_NAME = "DataCollectorApp"
    
    @staticmethod
    def get_credential(username, fallback_env_var=None):
        """Get password for the given username from Windows Credential Manager.
        
        Args:
            username (str): The username to retrieve credentials for
            fallback_env_var (str, optional): Environment variable name to use as fallback
            
        Returns:
            str: The password for the username or None if not found
        """
        password = keyring.get_password(CredentialManager.SERVICE_NAME, username)
        
        # If password not found in Credential Manager and fallback provided, try environment variable
        if password is None and fallback_env_var:
            password = os.getenv(fallback_env_var)
            if password:
                logging.info(f"Using {fallback_env_var} from .env file as fallback")
        
        return password
    
    @staticmethod
    def set_credential(username, password):
        """Set or update credentials in Windows Credential Manager.
        
        Args:
            username (str): The username to store
            password (str): The password to store
            
        Returns:
            bool: True if successful
        """
        try:
            keyring.set_password(CredentialManager.SERVICE_NAME, username, password)
            return True
        except Exception as e:
            logging.error(f"Failed to set credentials: {e}")
            return False

class Config:
    """
    Holds the configuration details or environment variables for the application.
    """
    BASE_DIR = os.getenv('BASE_DIR')
    LOG_DIR = os.path.join(BASE_DIR, "logs") if BASE_DIR else None
    current_date = datetime.now().strftime('%Y-%m-%d')
    DATA_DIR = os.path.join(BASE_DIR, f"collected_data_{current_date}") if BASE_DIR else None
    CLEAN_DATA_DIR = os.path.join(BASE_DIR, f"cleaned_data_{current_date}") if BASE_DIR else None
    
    # Credentials - we'll set these during setup to use Credential Manager
    SMB_HOST = os.getenv('SMB_HOST')
    SFTP_HOST = os.getenv('SFTP_HOST')
    SFTP_PORT = int(os.getenv('SFTP_PORT')) if os.getenv('SFTP_PORT') else None
    USERNAME = None  # Will be set during setup
    PASSWORD = None  # Will be set during setup
    
    # Record files
    SMB_RECORD_FILE = os.path.join(BASE_DIR, "downloaded_smb_files.txt") if BASE_DIR else None
    SFTP_RECORD_FILE = os.path.join(BASE_DIR, "downloaded_sftp_files.txt") if BASE_DIR else None
    
    # Email configuration - we'll set these during setup to use Credential Manager
    GMAIL_SENDER = os.getenv('GOOGLE_EMAIL_SENDER')
    GMAIL_PASSWORD = None  # Will be set during setup
    GMAIL_RECIPIENT = os.getenv('GOOGLE_EMAIL_DESTINATOR')
    
    # Dictionary of shares and all the subfolders
    SHARES_SMB = {
        "Solarlogs": ["PV", "min"],
        "BellevueConso": ["Consumption", "Temperature", "Humidity"],
        "BellevueBooking": ["RoomAllocations"]
    }
    
    SHARES_SFTP = {
        "Meteo": ["Pred"]
    }
    
    SUBFOLDERS = {
        "Solarlogs": ["PV", "min"],
        "BellevueConso": ["Consumption", "Temperature", "Humidity"],
        "BellevueBooking": ["RoomAllocations"],
        "Meteo": ["Pred"]
    }
    
    # Define a constant for the Solarlogs min prefix
    MIN_PREFIX_SOLARLOGS = "min"
    
    @classmethod
    def setup(cls, file_type):
        """
        Setup the configuration by validating and creating necessary directories
        """
        # Try to get credentials from Windows Credential Manager first
        cls.load_credentials()
        
        # Then proceed with normal setup
        cls.validate_env_config()
        cls.setup_logging(file_type)
        cls.create_directories(file_type)
    
    @classmethod
    def load_credentials(cls):
        """
        Load credentials from Windows Credential Manager with fallback to .env
        """
        # Get main connection credentials
        username = os.getenv('DATA_USERNAME')
        if username:
            cls.USERNAME = username
            # Try to get password from Credential Manager first, then fallback to .env
            cls.PASSWORD = CredentialManager.get_credential(username, 'DATA_PASSWORD')
            
            # If using from .env file, offer to save to Windows Credential Manager
            if cls.PASSWORD and os.getenv('DATA_PASSWORD'):
                logging.info("Using password from .env file. Consider storing it in Windows Credential Manager for better security.")
        
        # Get email credentials
        email = cls.GMAIL_SENDER
        if email:
            cls.GMAIL_PASSWORD = CredentialManager.get_credential(email, 'APP_PASSWORD_GOOGLE_PASSWORD')
            
            # If using from .env file, offer to save to Windows Credential Manager
            if cls.GMAIL_PASSWORD and os.getenv('APP_PASSWORD_GOOGLE_PASSWORD'):
                logging.info("Using email password from .env file. Consider storing it in Windows Credential Manager for better security.")
        
    @classmethod
    def setup_logging(cls, module_name):
        """
        Configure logging for a specific module
        
        Args:
            module_name: Name of the module (used in log filename)
        """
        if not cls.BASE_DIR:
            raise ValueError("BASE_DIR not set in environment variables")
            
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        log_filename = f"{module_name}_{cls.current_date}.log"
        log_file = os.path.join(cls.LOG_DIR, log_filename)
        
        # Reset handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def validate_env_config(cls):
        """
        Validate configuration for the data collector
        """
        
        if not cls.BASE_DIR:
            raise ValueError("Missing BASE_DIR in .env file")
        
        if not cls.SMB_HOST:
            raise ValueError("Missing SMB_HOST in .env file")
        
        if not cls.SFTP_HOST or not cls.SFTP_PORT:
            raise ValueError("Missing SFTP_HOST or SFTP_PORT in .env file")
        
        if not all([cls.USERNAME, cls.PASSWORD]):
            raise ValueError("Missing credentials - not found in Windows Credential Manager or .env file")
        
        # email configuration
        if not all([cls.GMAIL_SENDER, cls.GMAIL_PASSWORD, cls.GMAIL_RECIPIENT]):
            logging.warning("Missing email configuration. Email notifications will be disabled.")
        
    @classmethod
    def create_directories(cls, file_type):
        """
        Create necessary directories for the application
        
        Args:
            file_type: Type of file being processed (collector or cleaner)
        """
        # Create directories
        if cls.LOG_DIR:
            os.makedirs(cls.LOG_DIR, exist_ok=True)
            logging.info(f"Log directory created at {cls.LOG_DIR}")
        
        if cls.DATA_DIR and file_type == "data_collector":
            os.makedirs(cls.DATA_DIR, exist_ok=True)
            logging.info(f"Data directory created at {cls.DATA_DIR}")
        
        if cls.CLEAN_DATA_DIR and file_type == "data_cleaner":
            os.makedirs(cls.CLEAN_DATA_DIR, exist_ok=True)
            logging.info(f"Clean data directory created at {cls.CLEAN_DATA_DIR}")
    
    @classmethod
    def send_error_email(cls, module_name, subject, error_message, traceback_info=None):
        """
        Send an error notification email using Gmail SMTP.
        
        Args:
            module_name: Name of the module reporting the error
            subject: Email subject
            error_message: Main error message
            traceback_info: Optional traceback information
        """
        # Check if email configuration is available
        if not all([cls.GMAIL_SENDER, cls.GMAIL_PASSWORD, cls.GMAIL_RECIPIENT]):
            logging.error("Email configuration is missing. Cannot send error notification.")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = cls.GMAIL_SENDER
            msg['To'] = cls.GMAIL_RECIPIENT
            msg['Subject'] = f"{module_name} Error: {subject}"
            
            # Simple, focused email body with just the error information
            body = f"""
            <html>
            <body>
                <h2>{module_name} Error Report</h2>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Error Message:</strong> {error_message}</p>
            """
            
            # Add traceback if provided
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
            
            # Send the email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(cls.GMAIL_SENDER, cls.GMAIL_PASSWORD)
                server.send_message(msg)
            
            logging.info("Error notification email sent successfully")
            
        except Exception as e:
            logging.error(f"Failed to send error email: {e}")