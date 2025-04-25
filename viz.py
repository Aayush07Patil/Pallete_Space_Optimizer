from itertools import permutations
import math
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import test_data_retrival
from collections import defaultdict
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from collections import defaultdict
import numpy as np
from flask import request

app = Dash(__name__)
server = app.server  # Needed for Flask endpoint

containers_data = [{'id': 1, 'ULDCategory': 'LD7', 'Length': 87.8, 'Width': 166.9, 'Height': 63.8, 'Widthx': 124.8, 'Heightx': 44.1, 'TB': 'B', 'SD': 'D', 'Type': 'Palette', 'Weight': 4500, 'Volume': 934913.72},
    {'id': 2, 'ULDCategory': 'LD7', 'Length': 87.8, 'Width': 166.9, 'Height': 63.8, 'Widthx': 124.8, 'Heightx': 44.1, 'TB': 'B', 'SD': 'D', 'Type': 'Palette', 'Weight': 4500, 'Volume': 934913.72},
    {'id': 3, 'ULDCategory': 'LD7', 'Length': 87.8, 'Width': 166.9, 'Height': 63.8, 'Widthx': 124.8, 'Heightx': 44.1, 'TB': 'B', 'SD': 'D', 'Type': 'Palette', 'Weight': 4500, 'Volume': 934913.72},
    {'id': 4, 'ULDCategory': 'LD7', 'Length': 87.8, 'Width': 166.9, 'Height': 63.8, 'Widthx': 124.8, 'Heightx': 44.1, 'TB': 'B', 'SD': 'D', 'Type': 'Palette', 'Weight': 4500, 'Volume': 934913.72},
    {'id': 5, 'ULDCategory': 'LD7', 'Length': 87.8, 'Width': 166.9, 'Height': 63.8, 'Widthx': 124.8, 'Heightx': 44.1, 'TB': 'B', 'SD': 'D', 'Type': 'Palette', 'Weight': 4500, 'Volume': 934913.72},
    {'id': 6, 'ULDCategory': 'LD7', 'Length': 87.8, 'Width': 166.9, 'Height': 63.8, 'Widthx': 124.8, 'Heightx': 44.1, 'TB': 'B', 'SD': 'D', 'Type': 'Palette', 'Weight': 4500, 'Volume': 934913.72},
    {'id': 7, 'ULDCategory': 'LD7', 'Length': 87.8, 'Width': 166.9, 'Height': 63.8, 'Widthx': 124.8, 'Heightx': 44.1, 'TB': 'B', 'SD': 'D', 'Type': 'Palette', 'Weight': 4500, 'Volume': 934913.72},
    {'id': 8, 'ULDCategory': 'LD7', 'Length': 87.8, 'Width': 166.9, 'Height': 63.8, 'Widthx': 124.8, 'Heightx': 44.1, 'TB': 'B', 'SD': 'D', 'Type': 'Palette', 'Weight': 4500, 'Volume': 934913.72},
    {'id': 9, 'ULDCategory': 'LD3', 'Length': 60.2, 'Width': 78.7, 'Height': 63.8, 'Widthx': 54.3, 'Heightx': 44.1, 'TB': 'B', 'SD': 'S', 'Type': 'Container', 'Weight': 1200, 'Volume': 302267.81}]
products_data = [
    [
        {
            "id": 1,
            "SerialNumber": 801907,
            "Length": 125.00006750000001,
            "Breadth": 96.00005184000001,
            "Height": 64.00003456,
            "PieceType": "ULD",
            "ULDCategory": "LD7",
            "GrossWt": 1130.0,
            "DestinationCode": "CVG",
            "awb_number": "615-92278141",
            "Volume": 768001.24
        },
        {
            "id": 2,
            "SerialNumber": 801908,
            "Length": 125.00006750000001,
            "Breadth": 96.00005184000001,
            "Height": 64.00003456,
            "PieceType": "ULD",
            "ULDCategory": "LD7",
            "GrossWt": 930.0,
            "DestinationCode": "CVG",
            "awb_number": "615-92278141",
            "Volume": 768001.24
        },
        {
            "id": 3,
            "SerialNumber": 802388,
            "Length": 125.00006750000001,
            "Breadth": 96.00005184000001,
            "Height": 64.00003456,
            "PieceType": "ULD",
            "ULDCategory": "LD7",
            "GrossWt": 1123.1,
            "DestinationCode": "CVG",
            "awb_number": "838-00058833",
            "Volume": 768001.24
        },
        {
            "id": 4,
            "SerialNumber": 802402,
            "Length": 125.00006750000001,
            "Breadth": 96.00005184000001,
            "Height": 64.00003456,
            "PieceType": "ULD",
            "ULDCategory": "LD7",
            "GrossWt": 1123.1,
            "DestinationCode": "CVG",
            "awb_number": "838-00058844",
            "Volume": 768001.24
        }
    ],
    [
        {
            "id": 5,
            "SerialNumber": 802916,
            "Length": 106.29927,
            "Breadth": 59.055150000000005,
            "Height": 31.496080000000003,
            "PieceType": "Bulk",
            "ULDCategory": "",
            "GrossWt": 1123.1,
            "DestinationCode": "CVG",
            "awb_number": "838-00058855",
            "Volume": 197717.25
        },
        {
            "id": 6,
            "SerialNumber": 802928,
            "Length": 106.29927,
            "Breadth": 59.055150000000005,
            "Height": 31.496080000000003,
            "PieceType": "Bulk",
            "ULDCategory": "",
            "GrossWt": 1123.1,
            "DestinationCode": "CVG",
            "awb_number": "838-00058866",
            "Volume": 197717.25
        },
        {
            "id": 7,
            "SerialNumber": 802924,
            "Length": 106.29927,
            "Breadth": 59.055150000000005,
            "Height": 31.496080000000003,
            "PieceType": "Bulk",
            "ULDCategory": "",
            "GrossWt": 1123.1,
            "DestinationCode": "CVG",
            "awb_number": "838-00058870",
            "Volume": 197717.25
        },
        {
            "id": 8,
            "SerialNumber": 804042,
            "Length": 106.29927,
            "Breadth": 59.055150000000005,
            "Height": 31.496080000000003,
            "PieceType": "Bulk",
            "ULDCategory": "",
            "GrossWt": 1123.1,
            "DestinationCode": "CVG",
            "awb_number": "838-00058914",
            "Volume": 197717.25
        },
        {
            "id": 9,
            "SerialNumber": 805435,
            "Length": 74.40948900000001,
            "Breadth": 52.362233,
            "Height": 48.031522,
            "PieceType": "Bulk",
            "ULDCategory": "",
            "GrossWt": 685.0,
            "DestinationCode": "CVG",
            "awb_number": "838-00046701",
            "Volume": 187142.67
        }
    ],
    [
        {
            "id": 214,
            "SerialNumber": 803899,
            "Length": 32.677183,
            "Breadth": 31.496080000000003,
            "Height": 35.43309,
            "PieceType": "Bulk",
            "ULDCategory": "",
            "GrossWt": 84.5,
            "DestinationCode": "YEG",
            "awb_number": "838-09916373",
            "Volume": 36467.85
        },
        {
            "id": 213,
            "SerialNumber": 805276,
            "Length": 37.401595,
            "Breadth": 25.590565,
            "Height": 16.535442,
            "PieceType": "Bulk",
            "ULDCategory": "",
            "GrossWt": 170.0,
            "DestinationCode": "YEG",
            "awb_number": "607-28019692",
            "Volume": 15826.53
        }
    ]
]
selected_container_id = 0
DC_total_volumes = {'ULDs': 3072004.96, 'CVG': 978011.67, 'YEG': 52294.38, 'YYC': 188572.84, 'LAX': 2290699.4100000127, 'YYZ': 735789.1500000012}

def get_orientations(product):
    return set(permutations([product['Length'], product['Breadth'], product['Height']]))

def fits(container, placed_products, x, y, z, l, w, h):
    epsilon = 1e-6
    if container['SD'] == 'S':
        if container['TB'] == "B":
            limit = container['Height'] - container['Heightx']
            if z < limit:
                # Check container bounds
                if x + l > container['Length'] + epsilon or y + w > container['Widthx'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
            else:
                if x + l > container['Length'] + epsilon or y + w > container['Width'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
        elif container['TB'] == "T":
            if z < container['Heightx']:
                # Check container bounds
                if x + l > container['Length'] + epsilon or y + w > container['Width'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
            else:
                if x + l > container['Length'] + epsilon or y + w > container['Widthx'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
    
    elif container['SD'] == "D":
        if container['TB'] == "B":
            limit_height = container['Height'] - container['Heightx']
            width_small = (container['Width'] - container['Widthx'])/2
            if z < limit_height:
                # Check container bounds
                if x + l > container['Length'] + epsilon or y < width_small - epsilon or y + w > container['Width'] - width_small + epsilon or z + h > container['Height'] + epsilon:
                    return False
            else:
                if x + l > container['Length'] + epsilon or y + w > container['Width'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
        elif container['TB'] == "T":
            width_small = (container['Width'] - container['Widthx'])/2
            if z < container['Heightx']:
                # Check container bounds
                if x + l > container['Length'] + epsilon or y + w > container['Width'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
            else:
                if x + l > container['Length'] + epsilon or y < width_small - epsilon or y + w > container['Width'] - width_small + epsilon or z + h > container['Height'] + epsilon:
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

        # Container dimensions
        L, W, H = container['Length'], container['Width'], container['Height']
        HX, WX = container['Heightx'], container['Widthx']  # Example offsets
        W_offset = (W - WX) / 2

        if container['SD'] == 'S':
            if container['TB'] == 'T':
                # Define the vertices based on given parameters
                vertices = np.array([
                    [0, 0, 0],          # 0
                    [0, W, 0],          # 1
                    [0, 0, H],          # 2
                    [0, WX, H],         # 3
                    [0, W, HX],         # 4
                    [L, 0, 0],          # 5
                    [L, W, 0],          # 6
                    [L, 0, H],          # 7
                    [L, WX, H],         # 8
                    [L, W, HX]          # 9
                ])
            elif container['TB'] == 'B':
                # Define the vertices based on given parameters
                vertices = np.array([
                    [0, 0, 0],          # 0
                    [0, WX, 0],          # 1
                    [0, 0, H],          # 2
                    [0, W, H],         # 3
                    [0, W, H- HX],         # 4
                    [L, 0, 0],          # 5
                    [L, WX, 0],          # 6
                    [L, 0, H],          # 7
                    [L, W, H],         # 8
                    [L, W, H-HX]          # 9
                ])
            # Define edges (pairs of vertex indices)
            edges = [
                [0, 1], [1, 4], [4, 3], [0, 2], [2, 3],  # Left side edges
                [5, 6], [6, 9], [9, 8], [5, 7], [7, 8],  # Right side edges
                [3, 8], [4, 9], [1, 6],  # Connecting edges
                [2, 7], [0, 5]  # Connecting edges
            ]

        elif container['SD'] == 'D':
            if container['TB'] == 'T':
                # Define vertices
                vertices = np.array([
                    [0, 0, 0],             # 0
                    [0, W, 0],             # 1
                    [0, 0, HX],            # 2
                    [0, W_offset, H],      # 3
                    [0, W_offset + WX, H], # 4
                    [0, W, HX],            # 5
                    [L, 0, 0],             # 6
                    [L, W, 0],             # 7
                    [L, 0, HX],            # 8
                    [L, W_offset, H],      # 9
                    [L, W_offset + WX, H], # 10
                    [L, W, HX]             # 11
                ])
            elif container['TB'] == 'B':
                # Define vertices
                vertices = np.array([
                    [0, W_offset, 0],             # 0
                    [0, W_offset + WX, 0],             # 1
                    [0, 0, H-HX],            # 2
                    [0, 0, H],      # 3
                    [0, W, H], # 4
                    [0, W, H-HX],            # 5
                    [L, W_offset, 0],             # 0
                    [L, W_offset + WX, 0],             # 1
                    [L, 0, H-HX],            # 2
                    [L, 0, H],      # 3
                    [L, W, H], # 4
                    [L, W, H-HX]             # 11
                ])

            edges = [
                [0, 1], [1, 5], [5, 2], [2, 0], # Left base
                [6, 7], [7, 11], [11, 8], [8, 6], # Right base
                [2, 3], [3, 4], [4, 5], # Left top
                [8, 9], [9, 10], [10, 11], # Right top
                [3, 9], [4, 10], # Connecting edges
                [0, 6], [1, 7], [2, 8], [5, 11] # Vertical edges
            ]
        
        # Extract edge coordinates
        edge_x, edge_y, edge_z = [], [], []
        for start, end in edges:
            edge_x += [vertices[start][0], vertices[end][0], None]
            edge_y += [vertices[start][1], vertices[end][1], None]
            edge_z += [vertices[start][2], vertices[end][2], None]

        if container['Type'] == 'Container':
            # Add wireframe container
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='grey', width=4),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)
        elif container['Type'] == 'Palette':
            # Add wireframe container
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='grey', width=4, dash='dot'),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)


        # Calculate aspect ratio
        max_dim = max(L, W, H)
        aspect_ratio = {'x': L / max_dim, 'y': W / max_dim, 'z': H / max_dim}

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


def main(products, containers):
    #FltNumber = "WS009"
    #FltOrigin = "CDG"
    #Date = "2024-11-20 00:00:00.000"
    #awb_route_master, awb_dimensions, flight_master, aircraft_master = test_data_retrival.data_import()
    #Palette_space, Product_list, DC_total_volumes = test_data_retrival.main(awb_dimensions, flight_master, aircraft_master, awb_route_master, FltNumber, FltOrigin, Date)

    #containers = Palette_space
    #products = Product_list
    
    blocked_containers = []
    #start_time = time.time()
    placed_products, unplaced_products, blocked_for_ULD = process(products, containers, blocked_containers, DC_total_volumes)
    placed_products = sorted(placed_products, key=lambda x: x['container'])
    unplaced_products = sorted(unplaced_products, key=lambda x: x['id'])
    print("placed_products")
    #print(placed_products)
    print(len(placed_products))
        
    print("unplaced_products")
    #print(unplaced_products)
    print(len(unplaced_products))
    
    #end_time= time.time()
    #time_elapsed = end_time - start_time
    #print(f"Time taken for execution {time_elapsed}")
    containers = [item for item in containers if item not in blocked_for_ULD]
    #visualize_separate_containers_with_plotly(containers, placed_products)
    return containers, placed_products

def create_3d_plot(container, placed_products):
    fig = go.Figure()

    # Filter placed products for the specific container
    container_products = [p for p in placed_products if p['container'] == container['id']]

    # Create mesh for each placed product
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'lime', 'magenta']
    for p in container_products:
        x, y, z, l, w, h = p['position']
        fig.add_trace(go.Mesh3d(
            x=[x, x + l, x + l, x, x, x + l, x + l, x],
            y=[y, y, y + w, y + w, y, y, y + w, y + w],
            z=[z, z, z, z, z + h, z + h, z + h, z + h],
            alphahull=0,
            color=colors[p['id'] % len(colors)],
            opacity=1.0,
            name=f"{p['awb_number']})"
        ))

    # Container dimensions
    L, W, H = container['Length'], container['Width'], container['Height']
    vertices = np.array([
        [0, 0, 0], [0, W, 0], [0, 0, H], [0, W, H],
        [L, 0, 0], [L, W, 0], [L, 0, H], [L, W, H]
    ])
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Create edges for container
    edge_x, edge_y, edge_z = [], [], []
    for start, end in edges:
        edge_x += [vertices[start][0], vertices[end][0], None]
        edge_y += [vertices[start][1], vertices[end][1], None]
        edge_z += [vertices[start][2], vertices[end][2], None]

    # Add wireframe container
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='grey', width=4),
        name=f"Container {container['ULDCategory']} - {container['id']}"
    ))

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, title='Length'),
            yaxis=dict(nticks=10, title='Width'),
            zaxis=dict(nticks=10, title='Height'),
            aspectmode="cube"
        ),
        title=f"Container {container['ULDCategory']} - {container['id']}",
    )
    
    return fig

# Layout
app.layout = html.Div([
    html.H1("Cargo Dashboard (.NET Integration)"),
    dcc.Graph(id='container-graph'),
    html.Div(id='product-table'),
    dcc.Interval(id='interval-refresh', interval=5*1000, n_intervals=0)
])

# Flask route to receive .NET data
@server.route("/update_data", methods=["POST"])
def update_data():
    global containers_data, products_data, selected_container_id
    data = request.get_json()
    containers_data = data.get('containers', [])
    products_data = data.get('products', [])
    selected_container_id = data.get('selected_container_id', 0)
    return "Data received", 200

containers, placed_products = main(products_data, containers_data)

# Callback to render graph and table
@app.callback(
    [Output('container-graph', 'figure'),
     Output('product-table', 'children')],
    [Input('interval-refresh', 'n_intervals')]
)

def render_graph(n):
    if not containers_data or not products_data:
        return go.Figure(), "No data received."

    # Find the selected container
    container = next((c for c in containers_data if c['id'] == selected_container_id), None)
    if not container:
        return go.Figure(), f"Container {selected_container_id} not found."

    container_id = container['id']
    placed_products = [p for p in products_data if p['container'] == container_id]

    # Create figure
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "table"}, {"type": "scene"}]], column_widths=[0.3, 0.7])

    # Destination Codes and AWB Data (for table)
    destination_codes = set()
    awb_data = defaultdict(lambda: {'DestinationCode': None, 'Count': 0})

    for p in placed_products:
        destination_codes.add(p['DestinationCode'])
        awb_data[p['awb_number']]['DestinationCode'] = p['DestinationCode']
        awb_data[p['awb_number']]['Count'] += 1
        
        # Safely evaluate the position string to a tuple
        try:
            position = ast.literal_eval(p['position'])  # Convert "(x,y,z,l,w,h)" to (x,y,z,l,w,h)
            if len(position) == 6:  # Ensure we have 6 elements: (x, y, z, l, w, h)
                x, y, z, l, w, h = position
                # Add product boxes to the 3D plot
                fig.add_trace(go.Mesh3d(
                    x=[x, x + l, x + l, x, x, x + l, x + l, x],
                    y=[y, y, y + w, y + w, y, y, y + w, y + w],
                    z=[z, z, z, z, z + h, z + h, z + h, z + h],
                    alphahull=0,
                    color='blue',  # You can change color or dynamically assign colors
                    opacity=1.0,
                    name=f"{p['awb_number']})"
                ), row=1, col=2)
        except Exception as e:
            print(f"Error parsing position for product {p['awb_number']}: {e}")
            continue

    # Create the table
    awb_table_data = pd.DataFrame([{
        'AWB Number': awb,
        'DestinationCode': data['DestinationCode'],
        'Pieces': data['Count']
    } for awb, data in awb_data.items()])
    awb_table_data.sort_values(by='Pieces', inplace=True, ascending=False)

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

    # Draw the container wireframe (edges)
    L, W, H = container['Length'], container['Width'], container['Height']
    # Corrected edges list
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face edges (4 edges)
        [4, 5], [5, 7], [7, 6], [6, 4],  # Top face edges (4 edges)
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges between top and bottom faces
    ]

    # Container vertices (for wireframe edges)
    vertices = np.array([
        [0, 0, 0], [0, W, 0], [0, 0, H], [0, W, H],  # Four corners of bottom face
        [L, 0, 0], [L, W, 0], [L, 0, H], [L, W, H]  # Four corners of top face
    ])

    edge_x, edge_y, edge_z = [], [], []
    for start, end in edges:
        edge_x += [vertices[start][0], vertices[end][0], None]
        edge_y += [vertices[start][1], vertices[end][1], None]
        edge_z += [vertices[start][2], vertices[end][2], None]

    # Add container wireframe to the 3D plot
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='grey', width=4),
        name=f"Container {container['id']}"
    ), row=1, col=2)

    # Calculate aspect ratio for the container (to ensure proper 3D visualization)
    max_dim = max(L, W, H)
    aspect_ratio = {'x': L / max_dim, 'y': W / max_dim, 'z': H / max_dim}

    # Update layout with custom aspect ratio
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, title='Length'),
            yaxis=dict(nticks=10, title='Width'),
            zaxis=dict(nticks=10, title='Height'),
            aspectratio=aspect_ratio
        ),
        title_text=f"Container {container['id']} and Placed Products",
        title_x=0.5
    )

    return fig, f"Visualized container {container_id}"

if __name__ == "__main__":
    app.run(debug=True)

