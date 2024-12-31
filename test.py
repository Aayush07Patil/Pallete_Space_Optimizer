from itertools import permutations
import math
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import data_retrival
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

def get_orientations(product):
    return set(permutations([product['Length'], product['Breadth'], product['Height']]))

def fits(container, placed_products, x, y, z, l, w, h):
    epsilon = 1e-6
    # Check container bounds
    if x + l > container['Length'] + epsilon or y + w > container['Width'] + epsilon or z + h > container['Height'] + epsilon:
        return False

    # Check for overlap with existing products
    for p in placed_products:
        px, py, pz, pl, pw, ph = p['position']
        if not (x + l <= px or px + pl <= x + epsilon or
                y + w <= py or py + pw <= y + epsilon or
                z + h <= pz or pz + ph <= z + epsilon):
            return False

    return True

def preprocess_containers_and_products(products, containers, blocked_for_ULD):
    # Filter products of type 'ULD'
    uld_products = [p for p in products if p['PieceType'] == 'ULD']
    blocked_containers = []

    # Check if containers with matching ULDCategory are available
    for product in uld_products:
        matching_container = next((c for c in containers if c['ULDCategory'] == product['ULDCategory']), None)
        if matching_container:
            print(f"Product {product['id']} (ULDCategory: {product['ULDCategory']}) blocks container {matching_container['id']}.")
            blocked_containers.append(matching_container)
            blocked_for_ULD.append(matching_container)
            containers.remove(matching_container)
            products.remove(product)  # Exclude the product from packing

    # Remove blocked containers from the container list
    containers = [c for c in containers if c not in blocked_containers]
    return products, containers, blocked_containers, blocked_for_ULD

def pack_products_sequentially(containers, products, blocked_containers, DC_total_volumes):
    """
    Pack products into containers sequentially based on volume constraints and dimensions.

    Parameters:
        containers (list): List of available containers with their dimensions and volumes.
        products (list): List of products to be placed with their dimensions and volumes.
        blocked_containers (list): List of containers that are blocked.
        DC_total_volumes (dict): Mapping of destination codes to total allowable volumes.

    Returns:
        tuple: (placed_products, remaining_products, blocked_containers, available_containers)
    """
    placed_products = []
    remaining_products = products[:]
    used_containers = []
    missed_product_count = 0
    running_volume_sum = 0

    if products:
        destination_code = products[0]['DestinationCode']
        print(f"\nProcessing Destination Code: {destination_code}")
    else:
        destination_code = 'ULDs'

    for container in containers:
        print(f"Placing products in container {container['id']} ({container['ULDCategory']})")
        container_placed = []  # Products placed in the current container
        container_volume = container['Volume']
        occupied_volume = 0

        for product in remaining_products[:]:
            dc_volume = DC_total_volumes.get(product['DestinationCode'], 0)

            # Check volume constraints
            if not (dc_volume - running_volume_sum) > 0.8 * container_volume:
                print("Volume constraint not satisfied, stopping process.")
                blocked_containers.extend(used_containers)
                remaining_containers = [c for c in containers if c not in blocked_containers]
                running_volume_sum = 0
                return placed_products, remaining_products, blocked_containers, remaining_containers

            if missed_product_count >= 3:
                print("Too many missed products. Blocking containers.")
                blocked_containers.extend(used_containers)
                break

            placed = try_place_product(product, container, container_placed, occupied_volume, placed_products)

            if placed:
                running_volume_sum += product['Volume']
                remaining_products.remove(product)
                if container not in used_containers:
                    used_containers.append(container)
            else:
                print(f"Product {product['id']} could not be placed in container {container['id']}.")
                missed_product_count += 1

        if not remaining_products:
            print(f"All products have been placed for {destination_code}")
            running_volume_sum = 0
            blocked_containers.extend(used_containers)
            break

        # Reverse remaining products for next iteration
        if missed_product_count >= 3:
            print("Reversing product list for retry.")
            remaining_products = remaining_products[::-1]
            missed_product_count = 0

    remaining_containers = [c for c in containers if c not in blocked_containers]
    return placed_products, remaining_products, blocked_containers, remaining_containers


def try_place_product(product, container, container_placed, occupied_volume, placed_products):
    """
    Attempt to place a product in the container.

    Parameters:
        product (dict): The product to be placed.
        container (dict): The container to place the product in.
        container_placed (list): List of already placed products in the container.
        occupied_volume (float): Current occupied volume of the container.
        placed_products (list): List of all placed products.

    Returns:
        bool: True if the product was successfully placed, False otherwise.
    """
    for orientation in get_orientations(product):
        l, w, h = orientation
        for x in range(0, math.floor(container['Length'] - l)):
            for y in range(0, math.floor(container['Width'] - w)):
                for z in range(0, math.floor(container['Height'] - h)):
                    if fits(container, container_placed, x, y, z, l, w, h):
                        product_data = {
                            'id': product['id'],
                            'position': (x, y, z, l, w, h),
                            'container': container['id'],
                            'Volume': product['Volume'],
                            'DestinationCode': product['DestinationCode'],
                            'awb_number': product['awb_number']
                        }
                        container_placed.append(product_data)
                        placed_products.append(product_data)
                        occupied_volume += product['Volume']
                        remaining_volume_percentage = round(((container['Volume'] - occupied_volume) * 100 / container['Volume']), 2)
                        print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume: {remaining_volume_percentage}%")
                        return True
    return False



def visualize_separate_containers_with_plotly(containers, placed_products):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'lime', 'magenta']

    for container in containers:
        # Initialize subplot with two columns
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.3, 0.7],  # Adjust column widths: 30% for table, 70% for plot
            specs=[[{"type": "table"}, {"type": "scene"}]]  # Left: Table, Right: 3D plot
        )
        
        destination_codes = set()
        awb_data = defaultdict(lambda: {'DestinationCode': None, 'Count': 0})  # Tracks AWBs with DestinationCode and count

        # Add products to container
        for p in placed_products:
            if p['container'] == container['id']:
                x, y, z, l, w, h = p['position']
                destination_codes.add(p['DestinationCode'])
                
                # Update awb_data
                awb_data[p['awb_number']]['DestinationCode'] = p['DestinationCode']
                awb_data[p['awb_number']]['Count'] += 1
                
                fig.add_trace(go.Mesh3d(
                    x=[x, x + l, x + l, x, x, x + l, x + l, x],
                    y=[y, y, y + w, y + w, y, y, y + w, y + w],
                    z=[z, z, z, z, z + h, z + h, z + h, z + h],
                    alphahull=0,
                    color=colors[p['id'] % len(colors)],
                    opacity=1.0,
                    name=f"{p['awb_number']})"
                ), row=1, col=2)

        destination_codes_list = list(destination_codes)
        destination_codes_text = ', '.join(destination_codes_list)

        # Convert awb_data to a DataFrame
        awb_table_data = pd.DataFrame([
            {'AWB Number': awb, 'DestinationCode': data['DestinationCode'], 'Pieces': data['Count']}
            for awb, data in awb_data.items()
        ])
        awb_table_data.sort_values(by='Pieces', inplace=True, ascending=False)  # Optional: Sort by AWB Number

        # Add table to the left column
        table_trace = go.Table(
            header=dict(values=['AWB Number', 'Destination Code', 'Pieces'], fill_color='lightblue', align='left'),
            cells=dict(
                values=[
                    awb_table_data['AWB Number'],
                    awb_table_data['DestinationCode'],
                    awb_table_data['Pieces']
                ],
                fill_color='white',
                align='left'
            )
        )
        fig.add_trace(table_trace, row=1, col=1)

        # Draw container on the 3D plot
        fig.add_trace(go.Mesh3d(
            x=[0, container['Length'], container['Length'], 0, 0, container['Length'], container['Length'], 0],
            y=[0, 0, container['Width'], container['Width'], 0, 0, container['Width'], container['Width']],
            z=[0, 0, 0, 0, container['Height'], container['Height'], container['Height'], container['Height']],
            color='lightgrey',
            opacity=0.1,
            name=f"{container['ULDCategory']} - {container['id']}"
        ), row=1, col=2)

        # Calculate aspect ratio based on container dimensions
        aspect_x = container['Length']
        aspect_y = container['Width']
        aspect_z = container['Height']
        max_dim = max(aspect_x, aspect_y, aspect_z)
        aspect_ratio = {
            'x': aspect_x / max_dim,
            'y': aspect_y / max_dim,
            'z': aspect_z / max_dim
        }

        # Update layout with custom aspect ratio and title
        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=10, title='Length'),
                yaxis=dict(nticks=10, title='Width'),
                zaxis=dict(nticks=10, title='Height'),
                aspectratio=aspect_ratio  # Set proportional aspect ratio
            ),
            title_text=f"Container {container['ULDCategory']} - {container['id']} and Placed Products<br>Destinations: {destination_codes_text}",
            title_x=0.5  # Center the title
        )

        fig.show()

def process(products, containers, blocked_containers, DC_total_volumes):
    containers_tp = containers[:]
    blocked_for_ULD = []
    placed = []
    unplaced = []
    placements = {container['id']: [] for container in containers_tp}  # Tracks placements per container

    # First pass: Process each product
    for product in products:
        # Preprocess containers and products to block ULD-related containers
        products, containers_tp, blocked_containers, blocked_for_ULD = preprocess_containers_and_products(product, containers_tp, blocked_for_ULD)

        # Place products sequentially
        placed_products, remaining_products, blocked_containers, containers_tp = pack_products_sequentially(
            containers_tp, products, blocked_containers, DC_total_volumes
        )

        # Record placements and update lists
        for p in placed_products:
            container_id = p['container']
            placements[container_id].append(p['position'])  # Store placement data
        placed.extend(placed_products)
        unplaced.extend(remaining_products)
        
    print("\nAttempting to place unplaced products in remaining spaces...")
    second_pass_placed = []
    placed = sorted(placed, key=lambda x:x['Volume'])
    used_container = []
    containers_tp = containers[:]
    containers_tp = [item for item in containers_tp if item not in blocked_for_ULD]
    for container in containers_tp:
        placed_products_in_container = [item for item in placed if item["container"] == container["id"]]
        total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
        if total_volume_of_placed_products >0.8*container['Volume']:
            containers_tp.remove(container)
    
    if containers_tp:
        unplaced = sorted(unplaced,key=lambda x:x["Volume"])
        missed_products_count = 0 
        for container in containers_tp:
            placed_products_in_container = [item for item in placed if item["container"] == container["id"]]
            total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
            container_volume = container['Volume']
            occupied_volume = total_volume_of_placed_products
            for product in unplaced:
                placed_ = False
                if missed_products_count < 3:
                    for orientation in get_orientations(product):
                        l, w, h = orientation
                        
                        for x in range(0,math.floor(container['Length'] - l)):
                            for y in range(0,math.floor(container['Width'] - w )):
                                for z in range(0,math.floor(container['Height'] - h)):
                                    if fits(container, placed_products_in_container, x, y, z, l, w, h):
                                        product_data = {
                                            'id': product['id'],
                                            'position': (x, y, z, l, w, h),
                                            'container': container['id'],
                                            'Volume': product['Volume'],
                                            'DestinationCode': product['DestinationCode'],
                                            'awb_number': product['awb_number']     
                                        }
                                        occupied_volume += product['Volume']
                                        remaining_volume_percentage =  round(((container['Volume'] - occupied_volume) * 100 / container['Volume']), 2)
                                        print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume in container = {remaining_volume_percentage}")
                                        placed.append(product_data)
                                        placed_products_in_container.append(product_data)
                                        unplaced.remove(product)
                                        placed_ = True
                                        if container not in used_container:
                                            used_container.append(container)
                                        break
                                if placed_:
                                    break
                            if placed_:
                                break
                        if placed_:
                            break
                    if not placed_:
                        print(f"Product {product['id']} could not be placed in container {container['id']}.")
                        missed_products_count += 1
                        if container not in used_container:
                            used_container.append(container)
            
            else:
                
                print("\nSwitching list around\n")
                unplaced = unplaced[::-1]
                missed_products_count = 0 
                
                for product in unplaced:
                    placed_ = False
                    if missed_products_count < 3:
                        for orientation in get_orientations(product):
                            l, w, h = orientation
                            
                            for x in range(0,math.floor(container['Length'] - l)):
                                for y in range(0,math.floor(container['Width'] - w )):
                                    for z in range(0,math.floor(container['Height'] - h)):
                                        if fits(container, placed_products_in_container, x, y, z, l, w, h):
                                            product_data = {
                                                'id': product['id'],
                                                'position': (x, y, z, l, w, h),
                                                'container': container['id'],
                                                'Volume': product['Volume'],
                                                'awb_number': product['awb_number']
                                            }
                                            occupied_volume += product['Volume']
                                            remaining_volume_percentage = (container_volume - occupied_volume)/container_volume
                                            print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume in container = {remaining_volume_percentage}")
                                            placed.append(product_data)
                                            placed_products_in_container.append(product_data)
                                            unplaced.remove(product)
                                            placed_ =True
                                            if container not in used_container:
                                                used_container.append(container)
                                            break
                                    if placed_:
                                        break
                                if placed_:
                                    break
                            if placed_:
                                break
                        if not placed_:
                            print(f"Product {product['id']} could not be placed in container {container['id']}.")
                            missed_products_count += 1
                            if container not in used_container:
                                used_container.append(container)
                            unplaced = unplaced[::-1]
                    
                    else:
                        break
            
            if not unplaced:
                print("All products have been placed.")
                break
                
    return placed, unplaced, blocked_for_ULD


def main():
    awb_route_master, awb_dimensions, flight_master, aircraft_master = data_import()
    FltNumber = "WS009"
    FltOrigin = "CDG"
    Date = "2024-11-20 00:00:00.000"
    Palette_space, Product_list, DC_total_volumes = data_retrival.main(awb_dimensions, flight_master, aircraft_master, awb_route_master, FltNumber, FltOrigin, Date)

    containers = Palette_space
    products = Product_list
    
    blocked_containers = []
    start_time = time.time()
    placed_products, unplaced_products, blocked_for_ULD = process(products, containers, blocked_containers, DC_total_volumes)
    placed_products = sorted(placed_products, key=lambda x: x['container'])
    unplaced_products = sorted(unplaced_products, key=lambda x: x['id'])
    print("placed_products")
    #print(placed_products)
    print(len(placed_products))
        
    print("unplaced_products")
    #print(unplaced_products)
    print(len(unplaced_products))
    
    end_time= time.time()
    time_elapsed = end_time - start_time
    print(f"Time taken for execution {time_elapsed}")
    containers = [item for item in containers if item not in blocked_for_ULD]
    visualize_separate_containers_with_plotly(containers, placed_products)

if __name__ == "__main__":
    main()