import requests
import json
import os
from datetime import datetime

def call_prediction_api():
    """
    Simple script to call both the consumption and production prediction endpoints
    and save the results to CSV files.
    """
    # Base URL for the API (adjust if running on a different host/port)
    base_url = "http://localhost:5000"
    
    print("Energy Prediction API Client")
    print("===========================")
    print(f"Accessing API at: {base_url}")
    
    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "api_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Reload api 
    print("\nReloading API...")
    try:
        reload_response = requests.get(f"{base_url}/reload", timeout=60)
        if reload_response.status_code == 200:
            print("  API reloaded successfully.")
        else:
            print(f"  Error reloading API. Status code: {reload_response.status_code}")
            print(f"  Response: {reload_response.text}")
    except Exception as e:
        print(f"  Exception reloading API: {str(e)}")
    
    # 1. Call consumption prediction endpoint
    print("\n1. Calling consumption prediction endpoint...")
    try:
        consumption_response = requests.get(f"{base_url}/predictConsumption", timeout=600)
        
        if consumption_response.status_code == 200:
            consumption_data = consumption_response.json()
            print("  Success! Consumption predictions received.")
            
            # Save consumption predictions to CSV
            consumption_csv = os.path.join(output_dir, f"consumption_{timestamp}.csv")
            with open(consumption_csv, "w") as f:
                f.write("date,consumptionPrediction\n")  # Header
                for pred in consumption_data.get('predictions', []):
                    f.write(f"{pred['date']},{pred['consumptionPrediction']}\n")
            
            print(f"  Consumption predictions saved to: {consumption_csv}")
            
            # Display predictions
            print("\n  Daily Consumption Predictions:")
            print("  -----------------------------")
            for pred in consumption_data.get('predictions', []):
                print(f"  {pred['date']}: {pred['consumptionPrediction']:.2f} kWh")
        else:
            print(f"  Error calling consumption endpoint. Status code: {consumption_response.status_code}")
            print(f"  Response: {consumption_response.text}")
    
    except Exception as e:
        print(f"  Exception calling consumption endpoint: {str(e)}")
    
    # 2. Call production prediction endpoint
    print("\n2. Calling production prediction endpoint...")
    try:
        production_response = requests.get(f"{base_url}/predictProduction", timeout=60)
        
        if production_response.status_code == 200:
            production_data = production_response.json()
            print("  Success! Production predictions received.")
            
            # Save production predictions to CSV
            production_csv = os.path.join(output_dir, f"production_{timestamp}.csv")
            with open(production_csv, "w") as f:
                f.write("date,hour,productionPrediction\n")  # Header
                for pred in production_data.get('predictions', []):
                    f.write(f"{pred['date']},{pred['hour']},{pred['productionPrediction']}\n")
            
            print(f"  Production predictions saved to: {production_csv}")
            
            # Aggregate daily totals for display
            daily_production = {}
            for pred in production_data.get('predictions', []):
                date = pred['date']
                if date not in daily_production:
                    daily_production[date] = 0
                daily_production[date] += float(pred['productionPrediction'])
            
            # Display daily totals
            print("\n  Daily Production Predictions (Total):")
            print("  -------------------------------------")
            for date, total in daily_production.items():
                print(f"  {date}: {total:.2f} kWh")
        else:
            print(f"  Error calling production endpoint. Status code: {production_response.status_code}")
            print(f"  Response: {production_response.text}")
    
    except Exception as e:
        print(f"  Exception calling production endpoint: {str(e)}")
    
    print("\nPrediction calls completed.")

if __name__ == "__main__":
    call_prediction_api()