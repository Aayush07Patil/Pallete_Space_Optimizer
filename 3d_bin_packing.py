from itertools import permutations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go


# Define container dimensions
container = {'length': 10, 'width': 10, 'height': 10}

# Define products with dimensions (Length, Width, Height)
products = [
    {'id': 1, 'length': 4, 'width': 4, 'height': 4},
    {'id': 2, 'length': 6, 'width': 3, 'height': 2},
    {'id': 3, 'length': 5, 'width': 5, 'height': 3},
    {'id': 4, 'length': 2, 'width': 3, 'height': 5}
]

def get_orientations(product):
    return set(permutations([product['length'], product['width'], product['height']]))

def fits(container, placed_products, x, y, z, l, w, h):
    # Check container bounds
    if x + l > container['length'] or y + w > container['width'] or z + h > container['height']:
        return False
    
    for p in placed_products:
        px, py, pz, pl, pw, ph = p['position']
        if not (x + l <= px or px + pl <= x or
                y + w <= py or py + pw <= y or
                z + h <= pz or pz + ph <= z):
            return False
    
    return True

def pack_products(container, products):
    placed_products = []  # To store placed products with positions
    for product in products:
        placed = False
        for orientation in get_orientations(product):
            l, w, h = orientation
            for x in range(container['length'] - l + 1):
                for y in range(container['width'] - w + 1):
                    for z in range(container['height'] - h + 1):
                        if fits(container, placed_products, x, y, z, l, w, h):
                            # Place the product
                            placed_products.append({
                                'id': product['id'],
                                'position': (x, y, z, l, w, h)
                            })
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            print(f"Product {product['id']} could not be placed.")
    return placed_products

# Run the packing algorithm
placed_products = pack_products(container, products)

# Output the results
print("Placed Products:")
for p in placed_products:
    print(f"Product {p['id']} placed at {p['position']}")

def visualize_with_plotly(container, placed_products):
    fig = go.Figure()

    # Draw the container outline
    fig.add_trace(go.Mesh3d(
        x=[0, container['length'], container['length'], 0, 0, container['length'], container['length'], 0],
        y=[0, 0, container['width'], container['width'], 0, 0, container['width'], container['width']],
        z=[0, 0, 0, 0, container['height'], container['height'], container['height'], container['height']],
        color='lightgrey',
        opacity=0.1,
        name="Container"
    ))

    # Colors for products
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']

    # Add each product as a cuboid
    for i, p in enumerate(placed_products):
        x, y, z, l, w, h = p['position']

        # Vertices of the cuboid
        vertices = [
            [x, y, z], [x + l, y, z], [x + l, y + w, z], [x, y + w, z],  # Bottom face
            [x, y, z + h], [x + l, y, z + h], [x + l, y + w, z + h], [x, y + w, z + h]  # Top face
        ]

        edges = [
            # Each edge of the cuboid
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]

        # Create mesh for the cuboid
        x_vertices = [v[0] for v in vertices]
        y_vertices = [v[1] for v in vertices]
        z_vertices = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x_vertices,
            y=y_vertices,
            z=z_vertices,
            alphahull=0,
            color=colors[i % len(colors)],
            opacity=1.0,
            name=f"Product {p['id']}"
        ))

    # Set axis labels and limits
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[0, container['length']], title='Length'),
            yaxis=dict(nticks=10, range=[0, container['width']], title='Width'),
            zaxis=dict(nticks=10, range=[0, container['height']], title='Height'),
        ),
        title="Container and Placed Products",
    )

    fig.show()

# Visualize with Plotly
visualize_with_plotly(container, placed_products)
