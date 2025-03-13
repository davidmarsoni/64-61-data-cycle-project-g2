import sys
import os
import shutil
from config import CredentialManager, ensure_installed

# Assurez-vous que keyring et dotenv sont installés
ensure_installed('keyring')
ensure_installed('python-dotenv')
from dotenv import load_dotenv, find_dotenv

def set_credentials_from_env():
    """
    Lit les identifiants depuis le fichier .env et les enregistre dans Windows Credential Manager.
    Crée ensuite un nouveau fichier .env en supprimant les informations sensibles.
    """
    # 0. Charger les variables d'environnement depuis .env
    load_dotenv()
    
    # === LECTURE DES IDENTIFIANTS DEPUIS LE FICHIER .ENV ===
    
    # Identifiants de connexion pour SMB/SFTP
    username = os.getenv("DATA_USERNAME")
    password = os.getenv("DATA_PASSWORD")
    
    # Identifiants email pour Gmail
    email_username = os.getenv("GOOGLE_EMAIL_SENDER")
    email_password = os.getenv("GOOGLE_EMAIL_PASSWORD")
    
    # Vérifier que les identifiants nécessaires sont présents
    if not all([username, password, email_username, email_password]):
        print("❌ Les identifiants requis ne sont pas tous définis dans le fichier .env")
        print("Assurez-vous que DATA_USERNAME, DATA_PASSWORD, GOOGLE_EMAIL_SENDER et GOOGLE_EMAIL_PASSWORD sont définis.")
        return 1
    
    # === DÉFINISSEZ VOS PARAMÈTRES NON SENSIBLES POUR LE NOUVEAU FICHIER .ENV ===
    
    # Ces paramètres seront écrits dans votre nouveau fichier .env (sans les mots de passe)
    env_params = {
        "SMB_HOST": os.getenv("SMB_HOST", "10.130.25.152"),
        "SFTP_HOST": os.getenv("SFTP_HOST", "10.130.25.152"),
        "SFTP_PORT": os.getenv("SFTP_PORT", "22"),
        "BASE_DIR": os.getenv("BASE_DIR", "C:/DataCollection"),
        "GOOGLE_EMAIL_SENDER": email_username,
        "GOOGLE_EMAIL_DESTINATOR": os.getenv("GOOGLE_EMAIL_DESTINATOR", email_username),
        "DATA_USERNAME": username  # Pour lier avec le credential manager
    }
    
    # === FIN DE LA SECTION DE CONFIGURATION ===
    
    # 1. Stocker les identifiants dans Windows Credential Manager
    print("Configuration des identifiants dans Windows Credential Manager...")
    
    # Enregistrement des identifiants de connexion
    success1 = CredentialManager.set_credential(username, password)
    if success1:
        print(f"✅ Identifiants de connexion pour '{username}' enregistrés avec succès!")
    else:
        print(f"❌ Échec de l'enregistrement des identifiants pour '{username}'")
        return 1
    
    # Enregistrement des identifiants email
    success2 = CredentialManager.set_credential(email_username, email_password)
    if success2:
        print(f"✅ Identifiants email pour '{email_username}' enregistrés avec succès!")
    else:
        print(f"❌ Échec de l'enregistrement des identifiants email pour '{email_username}'")
        return 1
    
    # 2. Créer ou mettre à jour le fichier .env (en supprimant toute info sensible)
    print("\nConfiguration du fichier .env (sans informations sensibles)...")
    
    # Déterminer le chemin du fichier .env
    env_path = find_dotenv()
    if not env_path:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    
    # Créer une copie de sauvegarde du .env original (optionnel)
    if os.path.exists(env_path):
        backup_path = env_path + ".backup"
        shutil.copy2(env_path, backup_path)
        print(f"✅ Une copie de sauvegarde de votre .env original a été créée: {backup_path}")
    
    # On crée un nouveau fichier .env sans les informations sensibles
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write("# Configuration pour le collecteur de données\n")
        f.write("# Informations sensibles stockées dans Windows Credential Manager\n\n")
        
        # Écrire les nouveaux paramètres
        for key, value in env_params.items():
            f.write(f"{key}={value}\n")
    
    print(f"✅ Fichier .env créé avec succès à : {env_path}")
    print(f"✅ Les mots de passe ont été supprimés du fichier .env")
    
    # Créer les répertoires nécessaires
    os.makedirs(env_params["BASE_DIR"], exist_ok=True)
    os.makedirs(os.path.join(env_params["BASE_DIR"], "logs"), exist_ok=True)
    
    print("\n✅ Configuration complète terminée!")
    print("\nℹ️ Les identifiants ont été lus depuis votre fichier .env et transférés au Credential Manager.")
    
    return 0

if __name__ == "__main__":
    sys.exit(set_credentials_from_env())