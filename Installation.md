# Installation

## Prerequisites

Before running any of the scripts of the project, you need to have the following software installed on your system:

Nessesary software:

- Python 3.8 or higher recommended 3.13.3
- Git
- Microsoft SQL Server 2019
- Microsoft SQL Server Management Studio (SSMS)
- PowerBI Desktop

Recommended software:

- Visual Studio Code

### Python

To Install Python, you can download the installer from the official website: [Python Downloads](https://www.python.org/downloads/). Follow the instructions to install Python on your system. Make sure to check the box that says "Add Python to PATH" during installation.

> [!Note]
> Let all the default options be selected during installation. This will ensure that Python is installed correctly and all necessary components are included.

### Git

To install Git, you can download the installer from the official website: [Git Downloads](https://git-scm.com/downloads/win). Follow the instructions to install Git on your system. Make sure to check the box that says "Add Git to PATH" during installation. If you did not check this box, it's added by default.

### Microsoft SQL Server 2019

To install Microsoft SQL Server 2019, you can download the installer from the official website: [SQL Server Downloads](https://go.microsoft.com/fwlink/?linkid=866662).
Once the installer is downloaded, run it and follow the instructions to install SQL Server on your system. Make sure to select the "Developer" edition during installation, as it is free for development and testing purposes.

> [!Note]
> If you not able to see the developer edition dowload the Iso version and mount it to your computer. Then run the setup.exe file and select the developer edition.

### Microsoft SQL Server Management Studio (SSMS)

To install Microsoft SQL Server Management Studio (SSMS), you can download the installer from the official website: [SSMS Downloads](https://aka.ms/ssmsfullsetup). Follow the instructions to install SSMS on your system. (Everything is selected by default, so just click next until the installation is complete.)

### PowerBI Desktop

To install PowerBI Desktop, you can download the installer from the official website: [PowerBI Downloads](https://aka.ms/ssmsfullsetup). This will redirect you to the Microsoft Store where you can download the app. Follow the instructions to install PowerBI Desktop on your system. (Everything is selected by default, so just click next until the installation is complete.)

### Visual Studio Code

Visual Studio Code is a lightweight code editor that is recommended to use for this project for example to run the scripts but is not required. You can download it from the official website: [VS Code Downloads](https://code.visualstudio.com/Download). Follow the instructions to install Visual Studio Code on your system. (Everything is selected by default, so just click next until the installation is complete.)

> [!TIP]
> If you using Visual Studio Code, you can install the Some extensions to make your life easier:
> - Python extension [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
> - Bat Runner extension [Bat Runner](http://marketplace.visualstudio.com/items?itemName=NilsSoderman.batch-runner)
> - Material Icon Theme extension [Material Icon Theme](https://marketplace.visualstudio.com/items?itemName=PKief.material-icon-theme)

## Deployment

Once you have installed the prerequisites, you can deploy the project by following these steps:

1. Create a folder on your server under 'C:\' called 'code' and navigate to it:

   ```bash
   mkdir C:\code
   cd C:\code
   ```

2. Clone the repository to your local machine using Git:

    ```bash
    git clone https://github.com/davidmarsoni/64-61-data-cycle-project-g2.git
    ```

3. Navigate to the cloned repository:

   ```bash
   cd 64-61-data-cycle-project-g2
   ```

4. Fill the .env file by copying the .env.example file and renaming it to .env:

   ```env
   # Base directory full path preferred
   BASE_DIR="<your_base_directory>"

   # Connection credentials
   SMB_HOST="<your_smb_host>"
   SFTP_HOST="<your_sftp_host>"
   SFTP_PORT=your_sftp_port
   DATA_USERNAME="<your_username>"
   DATA_PASSWORD="<your_password>"

   # Email configuration for notifications
   GOOGLE_EMAIL_SENDER="<your_google_email_sender>"
   APP_PASSWORD_GOOGLE_PASSWORD="<your_app_password_for_google_email_sender>"
   GOOGLE_EMAIL_DESTINATOR="<your_google_email_destinator>"
   ```

   >[!NOTE]
   > The Database connection are the username and passowrd of the user you created in the SQL Server or a new account you have created when you installed the SQL Server.

   ### Email configuration

   to send email notifications, you need to create an app password for your Google account. Follow these steps:

   1. Open the following link: [Google App Passwords](https://myaccount.google.com/apppasswords).
   2. If you need more information about how to create an app password, you can follow this guide: [Google App Passwords Guide](https://support.google.com/mail/answer/185833?hl=en).

   >[!NOTE]
   > App passwords are only available for Google accounts with 2-Step Verification enabled.

5. Move into the init folder:

   ```bash
   cd init
   ```

6. Run the following command to setup and install the required packages:

    ```bash
    python set_dependencies.py
    ```
    ```bash
    python set_creditials.py
    ```
    ```bash
    python setup_database.py
    ```
    
    Then one done you can start the 'scheduler.bat' file to setup the scheduler:

    ```bash
    scheduler.bat
    ```

7. Once the setup is complete, you can ethier let it run each day with the scheduler or run the scripts manually. To run the scripts manually, you can use the following command:

    ```bash
    launch_script.bat
    ```
