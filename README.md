# 64-61-data-cycle-project-g2

This project is a HES-SO School Project for the course 64-61 Data Cycle Project.  The project as these 4 business objectives:

- Being able to visualize power production and consumption
- Being able to visualize SP failures and issues
- Being able to visualize building occupation
- Being able to forecast power production and consumption

## Prerequisites

Before running any of the scripts of the project, you need to have the following software installed on your system:

Necessary software:

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

## Setup the project

First copy the `.env.example` file to `.env` and fill in the required variables.

```bash
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

### `BASE_DIR`

The base directory is where the collected and cleaned data will be stored. A directory structure will be created as follows:

```bash
<your_base_directory>
└── collected_data_yyyy-mm-dd
    ├── BellevueBooking
    │  ├── RoomAllocations_yyyymmdd.csv
    │  └── ...
    ├── BellevueConso
    │  ├── dd.mm.yyyy-Consumption.csv
    │  ├── dd.mm.yyyy-Humidity.csv
    │  ├── dd.mm.yyyy-Temperature.csv
    │  └── ...
    ├── Meteo 
    │  ├── pred_yyyy-mm-dd.csv
    │  └── ...
    └── Solarlogs
       ├── dd.mm.yyyy-PV.csv
       ├── minyymmdd.csv
       └── ...
└── cleaned_data_yyyy-mm-dd
```

> We recommend using a directory on your local machine and need a root of a disk with enough space to store the data. The script will create a new directory for each day of data collection. the estimated size of all the data is around 3GB currently and grow by about 20 mb per day.

### `SMB_HOST`,`SFTP_HOST`,`SFTP_PORT`,`DATA_USERNAME`,`DATA_PASSWORD`

These are the credentials to access the data sources. The script will use these credentials to connect to the data sources and collect the data. The credentials are stored in a `.env` file and are not included in the repository. Make sure to keep this file secure and do not share it with anyone.

### `GOOGLE_EMAIL_SENDER`,`APP_PASSWORD_GOOGLE_PASSWORD`,`GOOGLE_EMAIL_DESTINATOR`

These are the credentials to send email notifications. The script will use these credentials to send email notifications when the data collection is complete. The credentials are stored in a `.env` file and are not included in the repository. Make sure to keep this file secure and do not share it with anyone.

To generate an app password for your Google account you can click on the following link:
[Google App Passwords](https://support.google.com/accounts/answer/185201?hl=en). You will need to enable 2-Step Verification on your Google account to use app passwords. Once you have enabled 2-Step Verification, you can generate an app password and use it in the `APP_PASSWORD_GOOGLE_PASSWORD` variable.

## First run of the project

Before running any step of the project, you need to set up the dependencies credentials and the database. To do this, you need to run the following script in order:

- `set_dependencies.py` to install all the nessesary python packages and dependencies.
- `set_credentials.py` to configure the credentials used by the application.
- `setup_database.py` to create the database and the tables.

> [!NOTE]
> Make sure to have the `.env` file in the root of the project before running this script. and also make sure to have the `python` installed on your machine.

> [!TIP]
> The `setup_database.py` 2 mode one safe that will only try to create the database and the tables and one that have the `--force` option that will later on durring the launch of the script let you choose to drop the database and recreate it. **This is not recommended to use this option if you are not sure of what you are doing.**

## Run of the project manually

To run the project locally on a server you need to respect the following steps:

1. Collect the data from the sources

To do this you need to run the script `data_collector.py` in the root of the project. This script will collect the data from the sources and store it in the `BASE_DIR` directory.

```bash
python data_collector.py
```

2. Clean the data

Then you need to run the script `data_cleaner.py` in the root of the project. This script will clean the data and store it in the `BASE_DIR` directory.

```bash
python data_cleaner.py
```

3. Prediction of the data

Then you need to run the script `data_predicton.py` in the root of the project. This script will predict the data and store it in the `BASE_DIR` directory.

```bash
python data_prediction.py
```

By lauching this script you will be able to get the result of the prediction of the data. This script will use the cleaned data to predict the data and store it in the `BASE_DIR` directory.

4. Run the ETL process

Once the data is cleaned, you can run the script `data_ETL.py` to transform and populate the data into the database. This script will use the cleaned data and store it in the database.

```bash
python data_ETL.py
```

5. Open the dashboard

Open the powerbi dashboard with the .pbix file in the `dashboard` folder. You can use the Power BI Desktop application to open the file and connect to the database. The dashboard is already configured to connect to the database.


## Run the project automatically

To make this project run automatically, you just need to first fill the `.env` file with the credentials and then run the 'set_credentials.py' script to set the credentials in the database.

Then you need to run the database setup script to create the database and the tables. This script will also populate the database with the cleaned data.

```bash
python init/setup_database.py --force
```

After that you can just run the 'scheduler.bat' inside the `init` folder. This script will create a schedule task that will run the laucher.bat script every day at 7:30. This script will run the data collector, data cleaner, and data ETL scripts in order to collect, clean, and populate the data into the database each day.

```bash
scheduler.bat
```