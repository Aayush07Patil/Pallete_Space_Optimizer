import pandas as pd
import json
from datetime import datetime
from collections import defaultdict

def data_import():
    try:
        awb_route_master = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/WS route and Dims.xlsx', sheet_name=0)
        awb_dimensions = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/WS route and Dims.xlsx', sheet_name=1)
        flight_master = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/Flight master with Aircraft.xlsx', sheet_name=0)
        aircraft_master = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/AirCraft Master 1127 revised.xlsx', sheet_name=0)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None, None, None, None
    
    flight_master.rename(columns={'FlightID': 'FltNumber', 'Source': 'FltOrigin'}, inplace=True)
    return awb_route_master, awb_dimensions, flight_master, aircraft_master


def convert_dimensions_to_inches(awb_dimensions):
    # Conversion factor
    cm_to_inch = 0.393701

    # Convert dimensions for rows where MeasureUnit is 'Cms'
    awb_dimensions.loc[awb_dimensions['MeasureUnit'] == 'Cms', ['Length', 'Breadth', 'Height']] = (
        awb_dimensions.loc[awb_dimensions['MeasureUnit'] == 'Cms', ['Length', 'Breadth', 'Height']]
        * cm_to_inch
    ).round(2)

    # Update MeasureUnit to 'Inches' for these rows
    awb_dimensions.loc[awb_dimensions['MeasureUnit'] == 'Cms', 'MeasureUnit'] = 'Inches'

    return awb_dimensions

def add_ids_to_records(records):
    for idx, record in enumerate(records, start=1):
        # Add id key at the beginning
        record['id'] = idx
        # Reorder to ensure id is first
        records[idx - 1] = dict(sorted(record.items(), key=lambda item: item[0] != 'id'))
    return records

def get_aircraft_details_with_date(flt_number, flt_origin, flt_date, flight_master, aircraft_master):
    day_of_week = flt_date.weekday()  # Monday = 0, Sunday = 6

    # Filter flight_master
    filtered_flight = flight_master[
        (flight_master['FltNumber'] == flt_number) & 
        (flight_master['FltOrigin'] == flt_origin)
    ].copy()
    
    if not filtered_flight.empty:
        # Convert frequency to a list of integers
        filtered_flight['FrequencyList'] = filtered_flight['Frequency'].apply(
            lambda x: list(map(int, x.split(',')))
        )

        # Check if the flight operates on the given day
        filtered_flight = filtered_flight[
            filtered_flight['FrequencyList'].apply(lambda freq: freq[day_of_week] > 0)
        ]
        
        if not filtered_flight.empty:
            # Get the AirCraftType
            aircraft_type = filtered_flight.iloc[0]['AirCraftType']

            # Filter aircraft_master
            filtered_aircraft = aircraft_master[
                aircraft_master['AircraftType'] == aircraft_type
            ].copy()

            # Replace NaN in ULDCategory with blank strings
            filtered_aircraft['ULDCategory'] = filtered_aircraft['ULDCategory'].fillna("N")

            # Convert each row to a dictionary
            keys = ['ULDCategory', 'Length', 'Width', 'Height', 'Count', 'Weight']
            records = filtered_aircraft[keys].to_dict(orient='records')

            # Add unique IDs
            return add_ids_to_records(records)

    return []

def get_awb_dimensions(flt_number, flt_origin, flt_date, awb_route_master, awb_dimensions):
    # Ensure FltDate in awb_route_master is datetime
    awb_route_master['FltDate'] = pd.to_datetime(awb_route_master['FltDate'], errors='coerce')

    # Filter awb_route_master
    filtered_awb_route = awb_route_master[
        (awb_route_master['FltNumber'] == flt_number) &
        (awb_route_master['FltOrigin'] == flt_origin) &
        (awb_route_master['FltDate'].dt.date == flt_date.date())
    ]
    
    if not filtered_awb_route.empty:
        # Get AWBPrefix and AWBNumber
        awb_keys = filtered_awb_route[['AWBPrefix', 'AWBNumber', 'DestinationCode']].drop_duplicates()

        # Merge with awb_dimensions
        filtered_awb_dimensions = awb_dimensions.merge(
            awb_keys, on=['AWBPrefix', 'AWBNumber'], how='inner'
        )

        # Replace NaN in ULDCategory with blank strings
        filtered_awb_dimensions['ULDCategory'] = filtered_awb_dimensions['ULDCategory'].fillna("")

        # Create AWBNumber concatenated column
        filtered_awb_dimensions['awb_number'] = (
            filtered_awb_dimensions['AWBPrefix'].astype(str) + '-' +
            filtered_awb_dimensions['AWBNumber'].astype(str)
        )

        # Convert to list of dictionaries
        keys = ['SerialNumber', 'MeasureUnit', 'Length', 'Breadth', 
                'Height', 'PcsCount', 'PieceType', 'ULDCategory', 
                'GrossWt', 'DestinationCode', 'awb_number']
        records = filtered_awb_dimensions[keys].to_dict(orient='records')

        # Add unique IDs
        return add_ids_to_records(records)

    # Return empty list if no match found
    return []


def complete_containers(containers):
    
    expanded_containers = []
    current_id = 1 

    for container in containers:
        for _ in range(container['Count']):
            new_container = container.copy()
            new_container['id'] = current_id
            del new_container['Count']
            new_container['Volume'] = round(new_container['Length'] * new_container['Width'] * new_container['Height'],2)
            expanded_containers.append(new_container)
            current_id += 1  # 
    
    return expanded_containers

def complete_products_list(products):
    expanded_items = []
    for item in products:
        for _ in range(item['PcsCount']):
            new_item = item.copy()
            new_item.pop('PcsCount', None)
            new_item.pop('MeasureUnit', None)
            new_item['Volume'] = round(new_item['Length'] * new_item['Breadth'] * new_item['Height'],2)
            expanded_items.append(new_item)

    # Sort items: move items with PieceType "ULD" to the top
    sorted_items = sorted(
        expanded_items, 
        key=lambda x: (x['PieceType'] != 'ULD', x['DestinationCode'])
    )

    # Reset IDs
    for idx, item in enumerate(sorted_items, start=1):
        item['id'] = idx

    return sorted_items

def group_products_by_type_and_destination(products):
    # List to hold grouped lists of products
    grouped_products = []
    bulk_products = []
    # Extract all products with PieceType "ULD"
    uld_products = [product for product in products if product["PieceType"] == "ULD"]
    grouped_products.append(uld_products)  # Add ULD products as the first group
    
    # Group remaining products by DestinationCode
    remaining_products = [product for product in products if product["PieceType"] != "ULD"]
    destination_groups = defaultdict(list)
    
    for product in remaining_products:
        destination_groups[product["DestinationCode"]].append(product)
    
    # Add each destination group as a separate list
    for destination, group in destination_groups.items():
        bulk_products.append(sorted(group, key=lambda x: x["Volume"], reverse=True))
        rearranged_list = sorted(
            bulk_products,
            key=lambda sublist: sum(item['Volume'] for item in sublist) / len(sublist) if sublist else 0,
            reverse=True
        )
    grouped_products.extend(rearranged_list)
    
    return grouped_products

def volumes_list_for_each_destination(Products):
    total_volumes = [sum(item["Volume"] for item in sublist) for sublist in Products]
    result = {"ULDs": sum(item["Volume"] for item in Products[0])}
    for sublist in Products[1:]:
        destination_code = sublist[0]["DestinationCode"]
        result[destination_code] = sum(item["Volume"] for item in sublist)

    return result

def main(awb_dimensions, flight_master, aircraft_master, awb_route_master, FltNumber, FltOrigin, Date):

    FltDate = datetime.strptime(Date, "%Y-%m-%d %H:%M:%S.%f")

    # Convert cms to inches
    awb_dimensions = convert_dimensions_to_inches(awb_dimensions)
    
    # Get palettes
    Palette_result = get_aircraft_details_with_date(FltNumber, FltOrigin, FltDate, flight_master, aircraft_master)
    Palette_space = complete_containers(Palette_result)

    # Get products
    Product_result = get_awb_dimensions(FltNumber, FltOrigin, FltDate, awb_route_master, awb_dimensions)
    Product_list = complete_products_list(Product_result)
    Products = group_products_by_type_and_destination(Product_list)
    DC_total_volumes = volumes_list_for_each_destination(Products)
    
    return Palette_space, Products, DC_total_volumes

if __name__ == "__main__":
    awb_route_master, awb_dimensions, flight_master, aircraft_master = data_import()
    FltNumber = "WS009"
    FltOrigin = "CDG"
    Date = "2024-11-20 00:00:00.000"
    Palette_space, Products, DC_total_volumes = main(
        awb_dimensions, flight_master, aircraft_master, awb_route_master, FltNumber, FltOrigin, Date
    )
    print(Palette_space)
    print(DC_total_volumes)
    json_data = json.dumps(Products, indent=4)
    # Open the file in write mode
    with open("output.txt", "w") as file:
        file.writelines(json_data)

    print("Output successfully written to output.txt")
