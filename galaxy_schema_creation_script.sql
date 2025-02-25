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
    date DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,
);

CREATE TABLE DimTime (
    id_time INT PRIMARY KEY IDENTITY(1,1),
    time TIME NOT NULL,
    hour INT NOT NULL,
    minute INT NOT NULL
);

-- ==============================
-- Dimension Tables Specific to FactBooking
-- ==============================

CREATE TABLE DimRoom (
    id_room INT PRIMARY KEY IDENTITY(1,1),
    roomName VARCHAR(255) NOT NULL,
    roomFullName VARCHAR(255) NOT NULL
);

CREATE TABLE DimUser (
    id_user INT PRIMARY KEY IDENTITY(1,1),
    userName VARCHAR(255) NOT NULL
);

CREATE TABLE DimActivity (
    id_activity INT PRIMARY KEY IDENTITY(1,1),
    activityName VARCHAR(255) NOT NULL
);

CREATE TABLE DimBookingType (
    id_bookingType INT PRIMARY KEY IDENTITY(1,1),
    code VARCHAR(50) NOT NULL,
    bookingType VARCHAR(100) NOT NULL
);

CREATE TABLE DimDivision (
    id_division INT PRIMARY KEY IDENTITY(1,1),
    divisionName VARCHAR(255) NOT NULL
);

-- ==============================
-- Dimension Tables Specific to FactEnergyMeasurements
-- ==============================

CREATE TABLE DimWeatherSite (
    id_site INT PRIMARY KEY IDENTITY(1,1),
    siteName VARCHAR(255) NOT NULL
);

CREATE TABLE DimInverter (
    id_inverter INT PRIMARY KEY IDENTITY(1,1),
    inverterName VARCHAR(255) NOT NULL
);

CREATE TABLE DimStatus (
    id_status INT PRIMARY KEY IDENTITY(1,1),
    statusName VARCHAR(255) NOT NULL
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
    id_bookingType INT NOT NULL,
    id_division INT NULL,
    id_activity INT NULL,


    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time_start) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_time_end) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_room) REFERENCES DimRoom(id_room),
    FOREIGN KEY (id_user) REFERENCES DimUser(id_user),
    FOREIGN KEY (id_professor) REFERENCES DimUser(id_user),
    FOREIGN KEY (id_bookingType) REFERENCES DimBookingType(id_bookingType),
    FOREIGN KEY (id_division) REFERENCES DimDivision(id_division),
    FOREIGN KEY (id_activity) REFERENCES DimActivity(id_activity)
    
);

CREATE TABLE FactEnergyConsumption (
    id_measurement INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    energy_consumed FLOAT NOT NULL, 
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time)
);

CREATE TABLE FactWeather (
    id_measurement INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    id_site INT NOT NULL,
    temperature FLOAT NULL,
    humidity FLOAT NULL,
    solar_radiation FLOAT NULL, 
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_site) REFERENCES DimWeatherSite(id_site)
);

CREATE TABLE FactSolarProduction (
    id_measurement INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    id_inverter INT NOT NULL,
    power_generated FLOAT NOT NULL, 
    id_status INT NULL,  -- Only for solar panels
    error_count INT DEFAULT 0,
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_inverter) REFERENCES DimInverter(id_inverter),
    FOREIGN KEY (id_status) REFERENCES DimStatus(id_status),
);
