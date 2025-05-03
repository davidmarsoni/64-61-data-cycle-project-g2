
import os
import pandas as pd
import logging
import tkinter as tk
from tkinter import filedialog
from sqlalchemy.orm import joinedload, aliased
from sqlalchemy import select
from ETL.db.base import get_session
from ETL.db.models import (
    FactSolarProduction, 
    DimDate, 
    DimTime, 
    DimInverter, 
    DimStatus,
    DimSite
)
from config import ensure_installed

# Ensure required packages are installed
ensure_installed('sqlalchemy')
ensure_installed('pandas')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def select_export_directory():
    """
    Open a dialog for the user to select a directory for exporting data
    
    Returns:
        str: Selected directory path or None if canceled
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Set a meaningful title for the dialog
    directory = filedialog.askdirectory(
        title="Select Directory to Export Solar Production Data"
    )
    
    root.destroy()
    return directory if directory else None

def export_solar_data(export_dir):
    """
    Export solar production data along with related dimension tables to CSV files
    using SQLAlchemy ORM instead of direct SQL queries
    
    Args:
        export_dir (str): Directory where CSV files will be saved
    """
    try:
        session = get_session()
        logger.info("Connected to database successfully")
        
        # Export dimension tables
        logger.info("Exporting dimension tables...")
        
        # Export DimDate
        dim_date_query = select(DimDate)
        dim_date_result = session.execute(dim_date_query).all()
        dim_date_data = [
            {
                DimDate.id_date.key: date[0].id_date, 
                DimDate.year.key: date[0].year, 
                DimDate.month.key: date[0].month, 
                DimDate.day.key: date[0].day
            } 
            for date in dim_date_result
        ]
        dim_date_df = pd.DataFrame(dim_date_data)
        dim_date_df.to_csv(os.path.join(export_dir, 'DimDate.csv'), index=False)
        logger.info(f"Exported DimDate: {len(dim_date_df)} records")
        
        # Export DimTime
        dim_time_query = select(DimTime)
        dim_time_result = session.execute(dim_time_query).all()
        dim_time_data = [
            {
                DimTime.id_time.key: time[0].id_time, 
                DimTime.hour.key: time[0].hour, 
                DimTime.minute.key: time[0].minute
            } 
            for time in dim_time_result
        ]
        dim_time_df = pd.DataFrame(dim_time_data)
        dim_time_df.to_csv(os.path.join(export_dir, 'DimTime.csv'), index=False)
        logger.info(f"Exported DimTime: {len(dim_time_df)} records")
        
        # Export DimInverter
        dim_inverter_query = select(DimInverter)
        dim_inverter_result = session.execute(dim_inverter_query).all()
        dim_inverter_data = [
            {
                DimInverter.id_inverter.key: inverter[0].id_inverter, 
                DimInverter.inverter_name.key: inverter[0].inverter_name
            } 
            for inverter in dim_inverter_result
        ]
        dim_inverter_df = pd.DataFrame(dim_inverter_data)
        dim_inverter_df.to_csv(os.path.join(export_dir, 'DimInverter.csv'), index=False)
        logger.info(f"Exported DimInverter: {len(dim_inverter_df)} records")
        
        # Export DimStatus
        dim_status_query = select(DimStatus)
        dim_status_result = session.execute(dim_status_query).all()
        dim_status_data = [
            {
                DimStatus.id_status.key: status[0].id_status, 
                DimStatus.status_name.key: status[0].status_name
            } 
            for status in dim_status_result
        ]
        dim_status_df = pd.DataFrame(dim_status_data)
        dim_status_df.to_csv(os.path.join(export_dir, 'DimStatus.csv'), index=False)
        logger.info(f"Exported DimStatus: {len(dim_status_df)} records")
        
        # Export DimSite
        dim_site_query = select(DimSite)
        dim_site_result = session.execute(dim_site_query).all()
        dim_site_data = [
            {
                DimSite.id_site.key: site[0].id_site, 
                DimSite.site_name.key: site[0].site_name
            } 
            for site in dim_site_result
        ]
        dim_site_df = pd.DataFrame(dim_site_data)
        dim_site_df.to_csv(os.path.join(export_dir, 'DimSite.csv'), index=False)
        logger.info(f"Exported DimSite: {len(dim_site_df)} records")
        
        # Export the fact table for solar production
        logger.info("Exporting FactSolarProduction table...")
        
        # Get solar production data
        fact_solar_query = select(FactSolarProduction)
        fact_solar_result = session.execute(fact_solar_query).all()
        fact_solar_data = [
            {
                FactSolarProduction.id_solar_production.key: prod[0].id_solar_production,
                FactSolarProduction.id_date.key: prod[0].id_date,
                FactSolarProduction.id_time.key: prod[0].id_time,
                FactSolarProduction.id_inverter.key: prod[0].id_inverter,
                FactSolarProduction.id_status.key: prod[0].id_status,
                FactSolarProduction.total_energy_produced.key: prod[0].total_energy_produced,
                FactSolarProduction.energy_produced.key: prod[0].energy_produced,
                FactSolarProduction.error_count.key: prod[0].error_count
            }
            for prod in fact_solar_result
        ]
        fact_solar_df = pd.DataFrame(fact_solar_data)
        fact_solar_df.to_csv(os.path.join(export_dir, 'FactSolarProduction.csv'), index=False)
        logger.info(f"Exported FactSolarProduction: {len(fact_solar_df)} records")
        
        # Create a denormalized view of the solar production data with joined dimension data
        logger.info("Creating denormalized view of solar production data...")
        
        # Use SQLAlchemy ORM for joins instead of raw SQL
        denormalized_query = (
            select(
                FactSolarProduction.id_solar_production,
                DimDate.year, DimDate.month, DimDate.day,
                DimTime.hour, DimTime.minute,
                DimInverter.inverter_name,
                DimStatus.status_name,
                FactSolarProduction.total_energy_produced,
                FactSolarProduction.energy_produced,
                FactSolarProduction.error_count
            )
            .join(DimDate, FactSolarProduction.id_date == DimDate.id_date)
            .join(DimTime, FactSolarProduction.id_time == DimTime.id_time)
            .join(DimInverter, FactSolarProduction.id_inverter == DimInverter.id_inverter)
            .join(DimStatus, FactSolarProduction.id_status == DimStatus.id_status)
            .order_by(DimDate.year, DimDate.month, DimDate.day, DimTime.hour, DimTime.minute)
        )
        
        denormalized_result = session.execute(denormalized_query).all()
        
        # Convert result to dictionary for pandas
        denormalized_data = [
            {
                FactSolarProduction.id_solar_production.key: row[0],
                DimDate.year.key: row[1], 
                DimDate.month.key: row[2], 
                DimDate.day.key: row[3],
                DimTime.hour.key: row[4], 
                DimTime.minute.key: row[5],
                DimInverter.inverter_name.key: row[6],
                DimStatus.status_name.key: row[7],
                FactSolarProduction.total_energy_produced.key: row[8],
                FactSolarProduction.energy_produced.key: row[9],
                FactSolarProduction.error_count.key: row[10]
            }
            for row in denormalized_result
        ]
        
        denormalized_df = pd.DataFrame(denormalized_data)
        denormalized_df.to_csv(os.path.join(export_dir, 'SolarProductionDenormalized.csv'), index=False)
        logger.info(f"Exported denormalized solar production data: {len(denormalized_df)} records")
        
        # Close the session
        session.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error exporting solar data: {str(e)}")
        return False

def main():
    """
    Main function that drives the export process
    """
    logger.info("Starting Solar Production Data Export Tool")
    
    # Ask user to select export directory
    export_dir = select_export_directory()
    
    if not export_dir:
        logger.warning("Export cancelled: No directory selected")
        return
    
    logger.info(f"Exporting data to: {export_dir}")
    
    # Export the data
    success = export_solar_data(export_dir)
    
    if success:
        logger.info("Solar production data export completed successfully!")
    else:
        logger.error("Failed to export solar production data")

if __name__ == "__main__":
    main()