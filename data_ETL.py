"""
Main entry point for all ETL processes related to historical data.
This script runs core data ETL processes in sequence.
"""

import logging
import os
import sys
from datetime import datetime
from ETL.Fact.ETL_process_Temperature_Humidity_Consumption import populate_dim_tables_and_facts as run_consumption_etl
from ETL.Fact.ETL_process_Solar import populate_dim_tables_and_facts as run_solar_etl
from ETL.Fact.ETL_process_RoomAllocation import populate_dim_tables_and_facts as run_room_allocation_etl
from ETL.Fact.ETL_process_Meteo import populate_dim_tables_and_facts as run_meteo_etl
from ETL.utils.logging_utils import setup_logging, send_error_summary

# Set up logging
log_file = setup_logging("ETL_Main")
logger = logging.getLogger("ETL_Main")

def run_etl():
    """Run all core ETL processes in sequence"""
    logger.info("Starting main ETL process")
    
    try:
        # Run Energy Consumption ETL
        logger.info("Starting Energy Consumption ETL process")
        run_consumption_etl()
        logger.info("Energy Consumption ETL process completed")
        
        # Run Solar Production ETL
        logger.info("Starting Solar Production ETL process")
        run_solar_etl()
        logger.info("Solar Production ETL process completed")
        
        # Run Room Allocation ETL
        logger.info("Starting Room Allocation ETL process")
        run_room_allocation_etl()
        logger.info("Room Allocation ETL process completed")
        
        # Run Meteo ETL
        logger.info("Starting Meteo ETL process")
        run_meteo_etl()
        logger.info("Meteo ETL process completed")
        
        logger.info("All ETL processes completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"ETL process failed: {str(e)}", exc_info=True)
        # Send error summary email if there were errors
        send_error_summary("ETL_Main")
        return False

if __name__ == "__main__":
    success = run_etl()
    sys.exit(0 if success else 1)