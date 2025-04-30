import sys
import os
import shutil

# Add parent directory to path so we can import from config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CredentialManager, ensure_installed

# Make sure keyring and dotenv are installed
ensure_installed('keyring')
ensure_installed('python-dotenv')
from dotenv import load_dotenv, find_dotenv

def set_credentials_from_env():
    """
    Reads credentials from the .env file and stores them in Windows Credential Manager.
    Then creates a new .env file by removing sensitive information.
    """
    # 0. Load environment variables from .env
    load_dotenv()
    
    # === READ CREDENTIALS FROM .ENV FILE ===
    
    # Login credentials for SMB/SFTP
    username = os.getenv("DATA_USERNAME")
    password = os.getenv("DATA_PASSWORD")
    
    # Email credentials for Gmail
    email_username = os.getenv("GOOGLE_EMAIL_SENDER")
    # Check for both possible env variable names to ensure compatibility
    email_password = os.getenv("APP_PASSWORD_GOOGLE_PASSWORD") or os.getenv("GOOGLE_EMAIL_PASSWORD")
    
    # Verify that all required credentials are present
    if not all([username, password, email_username, email_password]):
        print("ERROR: Required credentials are not all defined in the .env file")
        print("Make sure DATA_USERNAME, DATA_PASSWORD, GOOGLE_EMAIL_SENDER and APP_PASSWORD_GOOGLE_PASSWORD (or GOOGLE_EMAIL_PASSWORD) are defined.")
        return 1
    
    # === DEFINE YOUR NON-SENSITIVE PARAMETERS FOR THE NEW .ENV FILE ===
    
    # These parameters will be written to your new .env file (without passwords)
    env_params = {
        "SMB_HOST": os.getenv("SMB_HOST", "10.130.25.152"),
        "SFTP_HOST": os.getenv("SFTP_HOST", "10.130.25.152"),
        "SFTP_PORT": os.getenv("SFTP_PORT", "22"),
        "BASE_DIR": os.getenv("BASE_DIR", "C:/DataCollection"),
        "GOOGLE_EMAIL_SENDER": email_username,
        "GOOGLE_EMAIL_DESTINATOR": os.getenv("GOOGLE_EMAIL_DESTINATOR", email_username),
        "DATA_USERNAME": username  # To link with credential manager
    }
    
    # === END OF CONFIGURATION SECTION ===
    
    # 1. Store credentials in Windows Credential Manager
    print("Configuring credentials in Windows Credential Manager...")
    
    # Save login credentials
    success1 = CredentialManager.set_credential(username, password)
    if success1:
        print(f"✅ Login credentials for '{username}' successfully saved!")
    else:
        print(f"ERROR: Failed to save credentials for '{username}'")
        return 1
    
    # Save email credentials
    success2 = CredentialManager.set_credential(email_username, email_password)
    if success2:
        print(f"✅ Email credentials for '{email_username}' successfully saved!")
    else:
        print(f"ERROR: Failed to save email credentials for '{email_username}'")
        return 1
    
    # 2. Create or update the .env file (removing all sensitive info)
    print("\nConfiguring .env file (without sensitive information)...")
    
    # Determine the path to the .env file
    env_path = find_dotenv()
    if not env_path:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    
    # Create a backup of the original .env (optional)
    if os.path.exists(env_path):
        backup_path = env_path + ".backup"
        shutil.copy2(env_path, backup_path)
        print(f"✅ A backup copy of your original .env has been created: {backup_path}")
    
    # Create a new .env file without sensitive information
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write("# Configuration for the data collector\n")
        f.write("# Sensitive information stored in Windows Credential Manager\n\n")
        
        # Write the new parameters
        for key, value in env_params.items():
            f.write(f"{key}={value}\n")
    
    print(f"✅ .env file successfully created at: {env_path}")
    print(f"✅ Passwords have been removed from the .env file")
    
    # Create necessary directories
    os.makedirs(env_params["BASE_DIR"], exist_ok=True)
    os.makedirs(os.path.join(env_params["BASE_DIR"], "logs"), exist_ok=True)
    
    print("\n✅ Complete configuration finished!")
    print("\n Credentials have been read from your .env file and transferred to Credential Manager.")
    
    return 0

if __name__ == "__main__":
    sys.exit(set_credentials_from_env())