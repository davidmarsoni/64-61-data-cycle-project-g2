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
    roomName VARCHAR(255) NOT NULL
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
    id_bookingtype INT PRIMARY KEY IDENTITY(1,1),
    code VARCHAR(50) NOT NULL,
    bookingType VARCHAR(100) NOT NULL
);

CREATE TABLE DimDivision (
    id_division INT PRIMARY KEY IDENTITY(1,1),
    divisionName VARCHAR(255) NOT NULL
);

CREATE TABLE DimClassroom (
    id_classroom INT PRIMARY KEY IDENTITY(1,1),
    classroomName VARCHAR(255) NOT NULL
);

-- ==============================
-- Dimension Tables Specific to FactSolarProduction
-- ==============================

CREATE TABLE DimInverter (
    id_inverter INT PRIMARY KEY IDENTITY(1,1),
    inverterName VARCHAR(255) NOT NULL
);

CREATE TABLE DimStatus (
    id_status INT PRIMARY KEY IDENTITY(1,1),
    statusName VARCHAR(255) NOT NULL
);

-- ==============================
-- Dimension Tables Specific to FactMeteoSwissData
-- ==============================
CREATE TABLE DimSite (
    id_site INT PRIMARY KEY IDENTITY(1,1),
    siteName VARCHAR(255) NOT NULL
);

-- ==============================
-- Fact Tables
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
    id_bookingtype INT NOT NULL,
    id_division INT NULL,
    id_activity INT NULL,
    isActive BIT NOT NULL DEFAULT 1,
    lastModified DATETIME NOT NULL DEFAULT GETDATE(),
    externalId VARCHAR(255) NULL,

    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time_start) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_time_end) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_room) REFERENCES DimRoom(id_room),
    FOREIGN KEY (id_user) REFERENCES DimUser(id_user),
    FOREIGN KEY (id_professor) REFERENCES DimUser(id_user),
    FOREIGN KEY (id_classroom) REFERENCES DimClassroom(id_classroom),
    FOREIGN KEY (id_bookingtype) REFERENCES DimBookingType(id_bookingtype),
    FOREIGN KEY (id_division) REFERENCES DimDivision(id_division),
    FOREIGN KEY (id_activity) REFERENCES DimActivity(id_activity)
);

CREATE TABLE FactEnergyConsumption (
    id_energyconsumption INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    energyConsumed FLOAT NOT NULL, 
    humidity FLOAT NOT NULL,
    temperature FLOAT NOT NULL,
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time)
);

CREATE TABLE FactSolarProduction (
    id_solarproduction INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    id_inverter INT NOT NULL,
    id_status INT NOT NULL,
    totalEnergyProduced FLOAT NOT NULL, 
    energyProduced FLOAT NOT NULL, 
    errorCount INT NOT NULL,
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_inverter) REFERENCES DimInverter(id_inverter),
    FOREIGN KEY (id_status) REFERENCES DimStatus(id_status)
);

CREATE TABLE FactPrediction (
    id_prediction INT PRIMARY KEY IDENTITY(1,1),
    id_date INT NOT NULL,
    id_time INT NOT NULL,
    predictedConsumption FLOAT NOT NULL,
    predictedProduction FLOAT NOT NULL,
    FOREIGN KEY (id_date) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_time) REFERENCES DimTime(id_time)
);

CREATE TABLE FactMeteoSwissData (
    id_meteoswissdata INT PRIMARY KEY IDENTITY(1,1),
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