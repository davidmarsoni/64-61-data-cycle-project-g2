-- Optionally drop the database if it exists (for testing)
USE master;
ALTER DATABASE data_cycle_db SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
DROP DATABASE IF EXISTS data_cycle_db;

-- Create the database
CREATE DATABASE data_cycle_db;
GO

-- Switch to the data_cycle_db database
USE data_cycle_db;
GO

-- Collective dimensions
CREATE TABLE DimDate
(
    id_date INT PRIMARY KEY IDENTITY(1,1),
    day INT,
    month INT,
    year INT
);

CREATE TABLE DimTime
(
    id_time INT PRIMARY KEY IDENTITY(1,1),
    hour INT,
    minute INT
);

-- specific dimensions for the FactBooking table
CREATE TABLE DimReservation
(
    id_reservation INT PRIMARY KEY IDENTITY(1,1),
    originalReservationId INT
);

CREATE TABLE DimRoom
(
    id_room INT PRIMARY KEY IDENTITY(1,1),
    roomName VARCHAR(100),
    roomFullName VARCHAR(200)
);

CREATE TABLE [DimUser]
(
    id_user INT PRIMARY KEY IDENTITY(1,1),
    userName VARCHAR(100)
);

CREATE TABLE DimClass
(
    id_class INT PRIMARY KEY IDENTITY(1,1),
    className VARCHAR(100)
);

CREATE TABLE DimActivity
(
    id_activity INT PRIMARY KEY IDENTITY(1,1),
    activityName VARCHAR(100)
);

-- FactBooking table
CREATE TABLE FactBooking
(
    id_reservation INT,
    id_dateBooking INT,
    id_timeStart INT,
    id_timeEnd INT,
    id_room INT,
    id_user INT,
    id_class INT,
    id_code INT,
    id_type INT,
    id_activity INT,
    PRIMARY KEY (id_reservation, id_dateBooking, id_timeStart, id_timeEnd, id_room, id_user, id_class, id_code, id_type, id_activity),
    FOREIGN KEY (id_reservation) REFERENCES DimReservation(id_reservation),
    FOREIGN KEY (id_dateBooking) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_timeStart) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_timeEnd) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_room) REFERENCES DimRoom(id_room),
    FOREIGN KEY (id_user) REFERENCES DimUser(id_user),
    FOREIGN KEY (id_class) REFERENCES DimClass(id_class),
    FOREIGN KEY (id_activity) REFERENCES DimActivity(id_activity)
);

-- specific dimensions for the FactPower table
CREATE TABLE DimSolarPanel
(
    id_solarPanel INT PRIMARY KEY IDENTITY(1,1),
    invertedId INT
);

CREATE TABLE DimIncident
(
    id_incident INT PRIMARY KEY IDENTITY(1,1),
    incidentName VARCHAR(255)
);



CREATE TABLE FactPower
(
    id_datePower INT,
    id_timePower INT,
    id_solarPanel INT,
    id_incident INT,
    powerGenerated FLOAT,
    powerConsumedBuilding FLOAT,
    humidity FLOAT,
    temperature FLOAT,
    PRIMARY KEY (id_datePower, id_timePower, id_solarPanel, id_incident),
    FOREIGN KEY (id_datePower) REFERENCES DimDate(id_date),
    FOREIGN KEY (id_timePower) REFERENCES DimTime(id_time),
    FOREIGN KEY (id_solarPanel) REFERENCES DimSolarPanel(id_solarPanel),
    FOREIGN KEY (id_incident) REFERENCES DimIncident(id_incident)
);