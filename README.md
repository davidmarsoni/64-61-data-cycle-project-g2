# 64-61-data-cycle-project-g2

This project is a HES-SO School Project for the course 64-61 Data Cycle Project.  The project as this 4 buisness objectives:

- Being able to visualize power production and consumption
- Being able to visualize SP failures and issues
- Being able to visualize building occupation
- Being able to forecast power production and consumption

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
```

> We recommend using a directory on your local machine and need a root of a disk with enough space to store the data. The script will create a new directory for each day of data collection. the estimated size of all the data is around 3GB currently and grow by about 20 mb per day. 

### `SMB_HOST`,`SFTP_HOST`,`SFTP_PORT`,`DATA_USERNAME`,`DATA_PASSWORD`

These are the credentials to access the data sources. The script will use these credentials to connect to the data sources and collect the data. The credentials are stored in a `.env` file and are not included in the repository. Make sure to keep this file secure and do not share it with anyone.

### `GOOGLE_EMAIL_SENDER`,`APP_PASSWORD_GOOGLE_PASSWORD`,`GOOGLE_EMAIL_DESTINATOR`

These are the credentials to send email notifications. The script will use these credentials to send email notifications when the data collection is complete. The credentials are stored in a `.env` file and are not included in the repository. Make sure to keep this file secure and do not share it with anyone.

To generate an app password for your Google account you can click on the following link:
[Google App Passwords](https://support.google.com/accounts/answer/185201?hl=en). You will need to enable 2-Step Verification on your Google account to use app passwords. Once you have enabled 2-Step Verification, you can generate an app password and use it in the `APP_PASSWORD_GOOGLE_PASSWORD` variable.

## First run of the project

Before running any step of the project, you need to configure the creidentials use by the application. You can do this by going into the `init` folder and starting the script `set_credentials.py`.

> Make sure to have the `.env` file in the root of the project before running this script. and also make sure to have the `python` installed on your machine.

## Run of the project

To run the project locally or scheduled on a server you need to respect the following steps:

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

3. Populate the database

To populate the database, you need to run the script `setup_database.py` in the init folder to create the database and the tables. This script will also populate the database with the cleaned data.

```bash
python init/setup_database.py
```

> Note: This script as 2 mode one safe that will only try to create the database and the tables and one that have the `--force` option that will later on durring the launch of the script let you choose to drop the database and recreate it. **This is not recommended to use this option if you are not sure of what you are doing.**

4. Run the ETL process

Once the data is cleaned, you can run the script `data_ETL.py` to transform and populate the data into the database. This script will use the cleaned data and store it in the database.

```bash
python data_ETL.py
```

5. Open the dashboard

Open the powerbi dashboard with the .pbix file in the `dashboard` folder. You can use the Power BI Desktop application to open the file and connect to the database. The dashboard is already configured to connect to the database.