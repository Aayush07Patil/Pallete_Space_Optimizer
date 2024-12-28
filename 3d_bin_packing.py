from itertools import permutations
from collections import defaultdict
import math
import time
import plotly.graph_objects as go
import pandas as pd
import data_retrival

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

def has_support(x, y, z, l, w, placed_products):
    if z == 0:  # On the container floor
        return True

    for p in placed_products:
        px, py, pz, pl, pw, ph = p['position']
        # Check if the product lies directly beneath the current product
        if px <= x < px + pl and py <= y < py + pw and pz + ph == z:
            # Calculate the overlapping area
            overlap_length = min(x + l, px + pl) - max(x, px)
            overlap_width = min(y + w, py + pw) - max(y, py)
            
            # If at least 50% of the product's bottom is covered by the placed product
            if overlap_length * overlap_width >= (l * w) / 2:
                return True

    return False

def preprocess_containers_and_products(products, containers):
    # Filter products of type 'ULD'
    uld_products = [p for p in products if p['PieceType'] == 'ULD']
    blocked_containers = []

    # Check if containers with matching ULDCategory are available
    for product in uld_products:
        matching_container = next((c for c in containers if c['ULDCategory'] == product['ULDCategory']), None)
        if matching_container:
            print(f"Product {product['id']} (ULDCategory: {product['ULDCategory']}) blocks container {matching_container['id']}.")
            blocked_containers.append(matching_container)
            containers.remove(matching_container)
            products.remove(product)  # Exclude the product from packing

    # Remove blocked containers from the container list
    containers = [c for c in containers if c not in blocked_containers]
    blocked_ULD_containers = blocked_containers

    return products, containers, blocked_containers, blocked_ULD_containers

def pack_products_sequentially(containers, products, blocked_container):
    placed_products = []  # To store placed products with positions
    remaining_products = products[:]  # Copy of the products list to track unplaced items
    used_container = []
    missed_products_count = 0
    retry_count = 0
    
    if retry_count <= 3:
        
        for container in containers:
            print(f"Placing products in container {container['id']} ({container['ULDCategory']})")
            container_placed = []  # Track products placed in this specific container
            container_volume = container['Volume']
            occupied_volume = 0
            
            
            for product in remaining_products[:]:  # Iterate over a copy to avoid issues
        
                placed = False
                """if occupied_volume >= 0.75 * container_volume:
                    print("Container is 80 percent full. Skipping further placements.")
                    if container not in used_container:
                        used_container.append(container)
                    break"""
                if missed_products_count <= 3:
                    
                    for orientation in get_orientations(product):
                        l, w, h = orientation
                        for x in range(0,math.floor(container['Length'] - l)):
                            for y in range(0,math.floor(container['Width'] - w )):
                                for z in range(0,math.floor(container['Height'] - h)):
                                    if fits(container, container_placed, x, y, z, l, w, h):
                                        product_data = {
                                            'id': product['id'],
                                            'SerialNumber': product['SerialNumber'],
                                            'position': (x, y, z, l, w, h),
                                            'container': container['id']
                                        }
                                        occupied_volume = occupied_volume + product['Volume']
                                        remaining_volume_percentage = (container_volume - occupied_volume)/container_volume
                                        print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume in container = {remaining_volume_percentage}")
                                        
                                        placed_products.append(product_data)
                                        container_placed.append(product_data)
                                        remaining_products.remove(product)
                                        placed = True
                                        if container not in used_container:
                                            used_container.append(container)
                                        break
                                if placed:
                                    break
                            if placed:
                                break
                        if placed:
                            break

                    if not placed:
                        print(f"Product {product['id']} could not be placed in container {container['id']}.")
                        missed_products_count += 1
                        if container not in used_container:
                            used_container.append(container)
                    
            else:
                
                print("\nSwitching list around\n")
                remaining_products = remaining_products[::-1]
                missed_products_count = 0
        
                    
                for product in remaining_products[:]:  # Iterate over a copy to avoid issues
            
                    placed = False
                    """if occupied_volume >= 0.75 * container_volume:
                        print("Container is 80 percent full. Skipping further placements.")
                        if container not in used_container:
                            used_container.append(container)
                        break"""
                    if missed_products_count <=3:
                        for orientation in get_orientations(product):
                            l, w, h = orientation
                            for x in range(0,math.floor(container['Length'] - l)):
                                for y in range(0,math.floor(container['Width'] - w)):
                                    for z in range(0,math.floor(container['Height'] - h)):
                                        if fits(container, container_placed, x, y, z, l, w, h):
                                            product_data = {
                                                'id': product['id'],
                                                'SerialNumber': product['SerialNumber'],
                                                'position': (x, y, z, l, w, h),
                                                'container': container['id']
                                            }
                                            occupied_volume = occupied_volume + product['Volume']
                                            remaining_volume_percentage = (container_volume - occupied_volume)/container_volume
                                            print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume in container = {remaining_volume_percentage}")
                                            
                                            placed_products.append(product_data)
                                            container_placed.append(product_data)
                                            remaining_products.remove(product)
                                            placed = True
                                            if container not in used_container:
                                                used_container.append(container)
                                            break
                                    if placed:
                                        break
                                if placed:
                                    break
                            if placed:
                                break

                        if not placed:
                            print(f"Product {product['id']} could not be placed in container {container['id']}.")
                            missed_products_count += 1
                            if container not in used_container:
                                used_container.append(container)
                    else:
                        blocked_container.extend(used_container)
                        remaining_products = remaining_products[::-1]
                        missed_products_count = 0
                        break
                            
            if not remaining_products:
                print("All products have been placed.")
                blocked_container.extend(used_container)
                break
            
        retry_count += 1
        
    else:
        print("Retries done")
              
    containers = [c for c in containers if c not in blocked_container]
    return placed_products, remaining_products, blocked_container, containers


def visualize_separate_containers_with_plotly(containers, placed_products):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'lime', 'magenta']

    for container in containers:
        fig = go.Figure()

        # Draw container
        fig.add_trace(go.Mesh3d(
            x=[0, container['Length'], container['Length'], 0, 0, container['Length'], container['Length'], 0],
            y=[0, 0, container['Width'], container['Width'], 0, 0, container['Width'], container['Width']],
            z=[0, 0, 0, 0, container['Height'], container['Height'], container['Height'], container['Height']],
            color='lightgrey',
            opacity=0.1,
            name=f"{container['ULDCategory']} - {container['id']}"
        ))

        # Add products to container
        for p in placed_products:
            if p['container'] == container['id']:
                x, y, z, l, w, h = p['position']
                fig.add_trace(go.Mesh3d(
                    x=[x, x + l, x + l, x, x, x + l, x + l, x],
                    y=[y, y, y + w, y + w, y, y, y + w, y + w],
                    z=[z, z, z, z, z + h, z + h, z + h, z + h],
                    alphahull=0,
                    color=colors[p['id'] % len(colors)],
                    opacity=1.0,
                    name=f"{p['id']})"
                ))

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

        # Update layout with custom aspect ratio
        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=10, title='Length'),
                yaxis=dict(nticks=10, title='Width'),
                zaxis=dict(nticks=10, title='Height'),
                aspectratio=aspect_ratio  # Set proportional aspect ratio
            ),
            title=f"Container {container['ULDCategory']} - {container['id']} and Placed Products"
        )

        fig.show()

def process(products, containers, blocked_containers):
    placed= []
    unplaced = []
    
    for product in products:
        
        # Preprocess containers and products to block ULD-related containers products
        products, containers_, blocked_containers, blocked_ULD_containers = preprocess_containers_and_products(product, containers)

        print("\nBlocked Containers:")
        for c in blocked_containers:
            print(f"Container {c['id']} (ULDCategory: {c['ULDCategory']}) is blocked.")

        placed_products, remaining_products, blocked_containers, containers_ = pack_products_sequentially(containers_, products, blocked_containers)
        
        print("\nPlaced Products:")
        for p in placed_products:
            print(f"Product {p['id']} placed in container {p['container']} at {p['position']}")
        placed.extend(placed_products)
        
        print("\nUnplaced Products:")
        for p in remaining_products:
            print(f"Product {p['id']} could not be placed.")
        unplaced.extend(remaining_products)
        
        grouped_data = defaultdict(list)
        for item in unplaced:
            grouped_data[item["DestinationCode"]].append(item)

        # Convert to a list of lists
        result = list(grouped_data.values())
        
        containers_ = [item for item in containers if item not in blocked_ULD_containers]            
    return placed, unplaced


def main():
    awb_route_master, awb_dimensions, flight_master, aircraft_master = data_import()
    FltNumber = "WS009"
    FltOrigin = "CDG"
    Date = "2024-11-20 00:00:00.000"
    Palette_space, Product_list = data_retrival.main(awb_dimensions, flight_master, aircraft_master, awb_route_master, FltNumber, FltOrigin, Date)

    containers = Palette_space
    products = Product_list
    """
    products =  [
        [
            {'id': 1, 'SerialNumber': 801907, 'Length': 125.0, 'Breadth': 96.0, 'Height': 64.0, 'PieceType': 'ULD', 'ULDCategory': 'LD7', 'GrossWt': 1130.0, 'DestinationCode': 'CVG', 'Volume': 768000.0}, 
            {'id': 2, 'SerialNumber': 801908, 'Length': 125.0, 'Breadth': 96.0, 'Height': 64.0, 'PieceType': 'ULD', 'ULDCategory': 'LD7', 'GrossWt': 930.0, 'DestinationCode': 'CVG', 'Volume': 768000.0}, 
            {'id': 3, 'SerialNumber': 802388, 'Length': 125.0, 'Breadth': 96.0, 'Height': 64.0, 'PieceType': 'ULD', 'ULDCategory': 'LD7', 'GrossWt': 1123.1, 'DestinationCode': 'CVG', 'Volume': 768000.0}, 
            {'id': 4, 'SerialNumber': 802402, 'Length': 125.0, 'Breadth': 96.0, 'Height': 64.0, 'PieceType': 'ULD', 'ULDCategory': 'LD7', 'GrossWt': 1123.1, 'DestinationCode': 'CVG', 'Volume': 768000.0}
        ], 
        [
            {'id': 5, 'SerialNumber': 802916, 'Length': 106.3, 'Breadth': 59.06, 'Height': 31.5, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1123.1, 'DestinationCode': 'CVG', 'Volume': 197759.46}, 
            {'id': 6, 'SerialNumber': 802928, 'Length': 106.3, 'Breadth': 59.06, 'Height': 31.5, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1123.1, 'DestinationCode': 'CVG', 'Volume': 197759.46}, 
            {'id': 7, 'SerialNumber': 802924, 'Length': 106.3, 'Breadth': 59.06, 'Height': 31.5, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1123.1, 'DestinationCode': 'CVG', 'Volume': 197759.46}, 
            {'id': 8, 'SerialNumber': 804042, 'Length': 106.3, 'Breadth': 59.06, 'Height': 31.5, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1123.1, 'DestinationCode': 'CVG', 'Volume': 197759.46}, 
            {'id': 9, 'SerialNumber': 805435, 'Length': 74.41, 'Breadth': 52.36, 'Height': 48.03, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 685.0, 'DestinationCode': 'CVG', 'Volume': 187130.05}
        ], 
        [
            {'id': 10, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 11, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 12, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 13, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 14, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 15, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 16, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 17, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 18, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 19, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 20, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 21, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 22, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 23, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 24, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}, 
            {'id': 25, 'SerialNumber': 803887, 'Length': 34.25, 'Breadth': 22.44, 'Height': 16.14, 'PieceType': 'Bulk', 'ULDCategory': '', 'GrossWt': 1000.0, 'DestinationCode': 'LAX', 'Volume': 12404.72}
        ]
    ]
    """
    
    blocked_containers = []
    start_time = time.time()
    placed_products, unplaced_products = process(products, containers, blocked_containers)
    print("Placed Products")
    print(placed_products)
    print("Unplaced Products")
    print(unplaced_products)
    end_time= time.time()
    time_elapsed = end_time - start_time
    print(f"Time taken for execution {time_elapsed}")

    visualize_separate_containers_with_plotly(containers, placed_products)
    #visualize_separate_containers_with_matplotlib(containers, placed_products)

if __name__ == "__main__":
    main()