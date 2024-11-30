import pandas as pd
from datetime import datetime


# Data import
awb_route_master = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/WS route and Dims.xlsx', sheet_name=0)
awb_dimensions = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/WS route and Dims.xlsx', sheet_name=1)
flight_master = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/Flight master with Aircraft.xlsx', sheet_name=0)
aircraft_master = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/AirCraft Master 1127.xlsx', sheet_name=0)

flight_master.rename(columns={'FlightID': 'FltNumber', 'Source': 'FltOrigin'}, inplace=True)

print(awb_route_master.dtypes)

def get_aircraft_details_with_date(flt_number, flt_origin, flt_date, flight_master, aircraft_master):
    """
    Get aircraft details for a given flight number, origin, and date.
    
    Args:
        flt_number (str): The flight number.
        flt_origin (str): The flight origin.
        flt_date (datetime.datetime): The flight date as a datetime object.
        flight_master (pd.DataFrame): DataFrame containing flight details.
        aircraft_master (pd.DataFrame): DataFrame containing aircraft details.
        
    Returns:
        list[dict]: List of dictionaries with aircraft details (ULDCategory, Length, Width, Height, Count, Weight).
        If no matching flight or aircraft is found, returns an empty list.
    """
    # Step 1: Extract the day of the week from the flt_date
    day_of_week = flt_date.weekday()  # Monday = 0, Sunday = 6

    # Step 2: Filter flight_master based on flt_number, flt_origin, and frequency
    filtered_flight = flight_master[
        (flight_master['FltNumber'] == flt_number) &
        (flight_master['FltOrigin'] == flt_origin)
    ]
    
    if not filtered_flight.empty:
        # Convert frequency to a list of integers
        filtered_flight['FrequencyList'] = filtered_flight['Frequency'].apply(lambda x: list(map(int, x.split(','))))

        # Filter flights operating on the given day
        filtered_flight = filtered_flight[filtered_flight['FrequencyList'].apply(lambda freq: freq[day_of_week] > 0)]
        
        if not filtered_flight.empty:
            # Get the AirCraftType
            aircraft_type = filtered_flight.iloc[0]['AirCraftType']
            
            # Step 3: Filter aircraft_master for rows matching the AirCraftType
            filtered_aircraft = aircraft_master[
                aircraft_master['AircraftType'] == aircraft_type
            ]
            
            # Step 4: Convert each row to a dictionary with specified keys
            aircraft_dicts = filtered_aircraft.to_dict(orient='records')
            result = [
                {key: row[key] for key in ['ULDCategory', 'Length', 'Width', 'Height', 'Count', 'Weight']}
                for row in aircraft_dicts
            ]
            
            return result
    
    # Return an empty list if no matching flight or aircraft is found
    return []

# Example usage
FltNumber = "WS425"
FltOrigin = "YYZ"
Date = "2024-10-29 00:00:00.000"
# Convert to datetime
FltDate = datetime.strptime(Date, "%Y-%m-%d %H:%M:%S.%f")
Palette_result = get_aircraft_details_with_date(FltNumber, FltOrigin, FltDate, flight_master, aircraft_master)
print(f"Palettes = {Palette_result}")


def get_awb_dimensions(flt_number, flt_origin, flt_date, awb_route_master, awb_dimensions):
    """
    Get AWB dimensions for a given flight number, origin, and date.

    Args:
        flt_number (str): The flight number.
        flt_origin (str): The flight origin.
        flt_date (datetime.datetime): The flight date as a datetime object.
        awb_route_master (pd.DataFrame): DataFrame containing AWB route details.
        awb_dimensions (pd.DataFrame): DataFrame containing AWB dimensions.

    Returns:
        list[dict]: List of dictionaries with AWB dimensions details (SerialNumber, MeasureUnit, Length,
                    Breadth, Height, PcsCount, PieceType, GrossWt).
        If no matching AWB is found, returns an empty list.
    """
    # Ensure FltDate in awb_route_master is a datetime object
    if not pd.api.types.is_datetime64_any_dtype(awb_route_master['FltDate']):
        awb_route_master['FltDate'] = pd.to_datetime(awb_route_master['FltDate'])

    # Step 1: Filter awb_route_master for the given FltNumber, FltOrigin, and FltDate
    filtered_awb_route = awb_route_master[
        (awb_route_master['FltNumber'] == flt_number) &
        (awb_route_master['FltOrigin'] == flt_origin) &
        (awb_route_master['FltDate'].dt.date == flt_date.date())  # Match date part only
    ]

    if not filtered_awb_route.empty:
        # Step 2: Get AWBPrefix and AWBNumber from the filtered rows
        awb_keys = filtered_awb_route[['AWBPrefix', 'AWBNumber']].drop_duplicates()

        # Step 3: Filter awb_dimensions for matching AWBPrefix and AWBNumber
        filtered_awb_dimensions = awb_dimensions.merge(
            awb_keys, on=['AWBPrefix', 'AWBNumber'], how='inner'
        )

        # Step 4: Convert each row in filtered_awb_dimensions to a dictionary
        awb_dimensions_dicts = filtered_awb_dimensions.to_dict(orient='records')
        result = [
            {key: row[key] for key in ['SerialNumber', 'MeasureUnit', 'Length', 'Breadth',
                                       'Height', 'PcsCount', 'PieceType', 'GrossWt']}
            for row in awb_dimensions_dicts
        ]

        return result

    # Return an empty list if no matching AWB is found
    return []

Product_result = get_awb_dimensions(FltNumber, FltOrigin, FltDate, awb_route_master, awb_dimensions)
print(f"Products = {Product_result}")
