from itertools import permutations
import math
import plotly.graph_objects as go
import pandas as pd
import data_retrival

def data_import():
    awb_route_master = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/WS route and Dims.xlsx', sheet_name=0)
    awb_dimensions = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/WS route and Dims.xlsx', sheet_name=1)
    flight_master = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/Flight master with Aircraft.xlsx', sheet_name=0)
    aircraft_master = pd.read_excel('D:/GIT/Pallete_Space_Optimizer/data/AirCraft Master 1127 revised.xlsx', sheet_name=0)

    # Rename columns for consistency
    flight_master.rename(columns={'FlightID': 'FltNumber', 'Source': 'FltOrigin'}, inplace=True)

    return awb_route_master, awb_dimensions, flight_master, aircraft_master

def get_orientations(product):
    return set(permutations([product['Length'], product['Breadth'], product['Height']]))

def fits(container, placed_products, x, y, z, l, w, h):
    # Check container bounds
    if x + l > container['Length'] or y + w > container['Width'] or z + h > container['Height']:
        return False

    # Check for overlap with existing products
    for p in placed_products:
        px, py, pz, pl, pw, ph = p['position']
        if not (x + l <= px or px + pl <= x or
                y + w <= py or py + pw <= y or
                z + h <= pz or pz + ph <= z):
            return False

    return True

def pack_products_sequentially(containers, products):
    placed_products = []  # To store placed products with positions
    remaining_products = products[:]  # Copy of the products list to track unplaced items

    for container in containers:
        print(f"Placing products in container {container['id']} ({container['ULDCategory']})")
        container_placed = []  # Track products placed in this specific container

        for product in remaining_products[:]:  # Iterate over a copy to avoid issues
            placed = False
            for orientation in get_orientations(product):
                l, w, h = orientation
                for x in range(math.floor(container['Length'] - l + 1)):
                    for y in range(math.floor(container['Width'] - w + 1)):
                        for z in range(math.floor(container['Height'] - h + 1)):
                            if fits(container, container_placed, x, y, z, l, w, h):
                                product_data = {
                                    'id': product['id'],
                                    'SerialNumber': product['SerialNumber'],
                                    'position': (x, y, z, l, w, h),
                                    'container': container['id']
                                }
                                placed_products.append(product_data)
                                container_placed.append(product_data)
                                remaining_products.remove(product)
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break
                if placed:
                    break

            if not placed:
                print(f"Product {product['id']} could not be placed in container {container['id']}.")

        if not remaining_products:
            print("All products have been placed.")
            break

    return placed_products, remaining_products

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
                    name=f"{p['SerialNumber']} (Container {container['id']})"
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

def main():
    awb_route_master, awb_dimensions, flight_master, aircraft_master = data_import()
    FltNumber = "WS009"
    FltOrigin = "CDG"
    Date = "2024-11-20 00:00:00.000"
    Palette_space, Product_list = data_retrival.main(awb_dimensions, flight_master, aircraft_master, awb_route_master, FltNumber, FltOrigin, Date)
    
    containers = Palette_space
    products = Product_list
    
    placed_products, remaining_products = pack_products_sequentially(containers, products)

    print("\nPlaced Products:")
    for p in placed_products:
        print(f"Product {p['id']} placed in container {p['container']} at {p['position']}")

    print("\nUnplaced Products:")
    for p in remaining_products:
        print(f"Product {p['id']} could not be placed.")

    visualize_separate_containers_with_plotly(containers, placed_products)

if __name__ == "__main__":
    main()