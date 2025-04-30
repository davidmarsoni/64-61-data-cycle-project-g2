-- Optionally drop the database if it exists (for testing)
USE master;
ALTER DATABASE data_cycle_db SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
DROP DATABASE IF EXISTS data_cycle_db;
GO

-- Create the database
CREATE DATABASE data_cycle_db;
GO

-- Switch to the new database
USE data_cycle_db;
GO


-- ==============================
-- Dimension Tables Common to all Fact Tables
-- ==============================

CREATE TABLE DimDate (
    id_date INT PRIMARY KEY IDENTITY(1,1),
    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL
);

CREATE TABLE DimTime (
    id_time INT PRIMARY KEY IDENTITY(1,1),
    hour INT NOT NULL,
    minute INT NOT NULL
);

-- ==============================
-- Dimension Tables Specific to FactBooking
-- ==============================

CREATE TABLE DimRoom (
    id_room INT PRIMARY KEY IDENTITY(1,1),
    room_name VARCHAR(255) NOT NULL
);

CREATE TABLE DimUser (
    id_user INT PRIMARY KEY IDENTITY(1,1),
    username VARCHAR(255) NOT NULL
);

CREATE TABLE DimActivity (
    id_activity INT PRIMARY KEY IDENTITY(1,1),
    activity_name VARCHAR(255) NOT NULL
);

CREATE TABLE DimBookingType (
    id_booking_type INT PRIMARY KEY IDENTITY(1,1),
    code VARCHAR(50) NOT NULL,
    booking_type VARCHAR(100) NOT NULL
);

CREATE TABLE DimDivision (
    id_division INT PRIMARY KEY IDENTITY(1,1),
    division_name VARCHAR(255) NOT NULL
);

CREATE TABLE DimClassroom (
    id_classroom INT PRIMARY KEY IDENTITY(1,1),
    classroom_name VARCHAR(255) NOT NULL
);

-- ==============================
-- Dimension Tables Specific to FactSolarProduction
-- ==============================

CREATE TABLE DimInverter (
    id_inverter INT PRIMARY KEY IDENTITY(1,1),
    inverter_name VARCHAR(255) NOT NULL
);

CREATE TABLE DimStatus (
    id_status INT PRIMARY KEY IDENTITY(1,1),
    status_name VARCHAR(255) NOT NULL
);

-- ==============================
-- Dimension Tables Specific to FactMeteoSwissData
-- ==============================
CREATE TABLE DimSite (
    id_site INT PRIMARY KEY IDENTITY(1,1),
    site_name VARCHAR(255) NOT NULL
);

-- ==============================
-- Fact Table
-- ==============================

CREATE TABLE FactBookings (
    id_booking INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time_start INT NOT NULL,
    id_time_end INT NOT NULL,
    id_room INT NOT NULL,
    id_user INT NOT NULL,
    id_professor INT NULL,
    id_classroom INT NULL,
    id_booking_type INT NOT NULL,
    id_division INT NULL,
    id_activity INT NULL,
    is_active BIT NOT NULL DEFAULT 1,
    last_modified DATETIME NOT NULL DEFAULT GETDATE(),
    external_id VARCHAR(255) NULL,

    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time_start) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_time_end) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_room) REFERENCES DimRoom(id_room),
    FOREIGN KEY (id_user) REFERENCES DimUser(id_user),
    FOREIGN KEY (id_professor) REFERENCES DimUser(id_user),
    FOREIGN KEY (id_classroom) REFERENCES DimClassRoom(id_classroom),
    FOREIGN KEY (id_booking_type) REFERENCES DimBookingType(id_bookingType),
    FOREIGN KEY (id_division) REFERENCES DimDivision(id_division),
    FOREIGN KEY (id_activity) REFERENCES DimActivity(id_activity)
);

CREATE TABLE FactEnergyConsumption (
    id_energy_consumption INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    energy_consumed FLOAT NOT NULL, 
    humidity FLOAT NOT NULL,
    temperature FLOAT NOT NULL,
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time)
);

CREATE TABLE FactSolarProduction (
    id_solar_production INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    id_inverter INT NOT NULL,
    id_status INT NOT NULL,
    total_energy_produced FLOAT NOT NULL, -- Total energy produced By the whole power plant
    energy_produced FLOAT NOT NULL, -- Energy produced by 1 inverter
    error_count INT NOT NULL, -- Number of errors for the inverter
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_inverter) REFERENCES DimInverter(id_inverter),
    FOREIGN KEY (id_status) REFERENCES DimStatus(id_status)
);

CREATE TABLE FactPrediction (
    id_prediction_weather INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    predicted_consumption FLOAT NOT NULL,
    predicted_production FLOAT NOT NULL,
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time)
);

CREATE TABLE FactMeteoSwissData (
    id_meteo_swiss_data INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    id_site INT NOT NULL,
    numPrediction INT NOT NULL,
    temperature FLOAT NOT NULL,
    humidity FLOAT NOT NULL,
    rain FLOAT NOT NULL,
    radiation FLOAT NOT NULL,
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_site) REFERENCES DimSite(id_site)
);