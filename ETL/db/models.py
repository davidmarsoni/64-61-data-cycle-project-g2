from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean, DateTime
from ETL.db.base import Base
import datetime

class DimDate(Base):
    __tablename__ = 'DimDate'
    
    id_date = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    day = Column(Integer, nullable=False)

class DimTime(Base):
    __tablename__ = 'DimTime'
    
    id_time = Column(Integer, primary_key=True, autoincrement=True)
    hour = Column(Integer, nullable=False)
    minute = Column(Integer, nullable=False)

class DimSite(Base):
    __tablename__ = 'DimSite'
    
    id_site = Column(Integer, primary_key=True, autoincrement=True)
    site_name = Column(String(255), nullable=False)

# Missing dimension tables
class DimRoom(Base):
    __tablename__ = 'DimRoom'
    
    id_room = Column(Integer, primary_key=True, autoincrement=True)
    room_name = Column(String(255), nullable=False)

class DimUser(Base):
    __tablename__ = 'DimUser'
    
    id_user = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=False) 

class DimActivity(Base):
    __tablename__ = 'DimActivity'
    
    id_activity = Column(Integer, primary_key=True, autoincrement=True)
    activity_name = Column(String(255), nullable=False) 

class DimBookingType(Base):
    __tablename__ = 'DimBookingType'
    
    id_booking_type = Column(Integer, primary_key=True, autoincrement=True) 
    code = Column(String(50), nullable=False)
    booking_type = Column(String(100), nullable=False)

class DimDivision(Base):
    __tablename__ = 'DimDivision'
    
    id_division = Column(Integer, primary_key=True, autoincrement=True)
    division_name = Column(String(255), nullable=False)

class DimClassroom(Base):
    __tablename__ = 'DimClassroom'
    
    id_classroom = Column(Integer, primary_key=True, autoincrement=True)
    classroom_name = Column(String(255), nullable=False)

class DimInverter(Base):
    __tablename__ = 'DimInverter'
    
    id_inverter = Column(Integer, primary_key=True, autoincrement=True)
    inverter_name = Column(String(255), nullable=False)

class DimStatus(Base):
    __tablename__ = 'DimStatus'
    
    id_status = Column(Integer, primary_key=True, autoincrement=True)
    status_name = Column(String(255), nullable=False) # Renamed from statusName

class FactMeteoSwissData(Base):
    __tablename__ = 'FactMeteoSwissData'
    
    id_meteo_swiss_data = Column(Integer, primary_key=True, autoincrement=True)
    id_date = Column(Integer, ForeignKey('DimDate.id_date'), nullable=False)
    id_time = Column(Integer, ForeignKey('DimTime.id_time'), nullable=False)
    id_site = Column(Integer, ForeignKey('DimSite.id_site'), nullable=False)
    numPrediction = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    rain = Column(Float, nullable=False)
    radiation = Column(Float, nullable=False)

# Missing fact tables
class FactBookings(Base):
    __tablename__ = 'FactBookings'
    
    id_booking = Column(Integer, primary_key=True, autoincrement=True)
    id_date = Column(Integer, ForeignKey('DimDate.id_date'), nullable=False)
    id_time_start = Column(Integer, ForeignKey('DimTime.id_time'), nullable=False)
    id_time_end = Column(Integer, ForeignKey('DimTime.id_time'), nullable=False)
    id_room = Column(Integer, ForeignKey('DimRoom.id_room'), nullable=False)
    id_user = Column(Integer, ForeignKey('DimUser.id_user'), nullable=False)
    id_professor = Column(Integer, ForeignKey('DimUser.id_user'), nullable=True)
    id_classroom = Column(Integer, ForeignKey('DimClassroom.id_classroom'), nullable=True)
    id_booking_type = Column(Integer, ForeignKey('DimBookingType.id_booking_type'), nullable=False)
    id_division = Column(Integer, ForeignKey('DimDivision.id_division'), nullable=True)
    id_activity = Column(Integer, ForeignKey('DimActivity.id_activity'), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False) 
    last_modified = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now, nullable=False) 
    external_id = Column(String(255), nullable=True)

class FactEnergyConsumption(Base):
    __tablename__ = 'FactEnergyConsumption'
    
    id_energy_consumption = Column(Integer, primary_key=True, autoincrement=True)
    id_date = Column(Integer, ForeignKey('DimDate.id_date'), nullable=False)
    id_time = Column(Integer, ForeignKey('DimTime.id_time'), nullable=False)
    energy_consumed = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)

class FactSolarProduction(Base):
    __tablename__ = 'FactSolarProduction'
    
    id_solar_production = Column(Integer, primary_key=True, autoincrement=True)
    id_date = Column(Integer, ForeignKey('DimDate.id_date'), nullable=False)
    id_time = Column(Integer, ForeignKey('DimTime.id_time'), nullable=False)
    id_inverter = Column(Integer, ForeignKey('DimInverter.id_inverter'), nullable=False)
    id_status = Column(Integer, ForeignKey('DimStatus.id_status'), nullable=False)
    total_energy_produced = Column(Float, nullable=False)
    energy_produced = Column(Float, nullable=False)
    error_count = Column(Integer, nullable=False)

class FactPrediction(Base):
    __tablename__ = 'FactPrediction'
    
    id_prediction_weather = Column(Integer, primary_key=True, autoincrement=True)
    id_date = Column(Integer, ForeignKey('DimDate.id_date'), nullable=False)
    id_time = Column(Integer, ForeignKey('DimTime.id_time'), nullable=False)
    predicted_consumption = Column(Float, nullable=False)
    predicted_production = Column(Float, nullable=False)