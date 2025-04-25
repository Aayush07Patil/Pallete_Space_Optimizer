import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from collections import defaultdict
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Sample Data (replace this with your actual data processing functions)
containers = [
    {'id': 1, 'Length': 10, 'Width': 5, 'Height': 5, 'Type': 'Container', 'ULDCategory': 'A', 'SD': 'S', 'TB': 'T', 'Heightx': 2, 'Widthx': 3},
    {'id': 2, 'Length': 8, 'Width': 4, 'Height': 4, 'Type': 'Container', 'ULDCategory': 'B', 'SD': 'D', 'TB': 'B', 'Heightx': 2, 'Widthx': 3}
]

# Sample placed products (replace with actual product data)
placed_products = [
    {'id': 1, 'awb_number': 'AWB123', 'container': 1, 'position': (0, 0, 0, 2, 2, 2), 'DestinationCode': 'NYC', 'Volume': 8},
    {'id': 2, 'awb_number': 'AWB124', 'container': 1, 'position': (3, 0, 0, 2, 2, 2), 'DestinationCode': 'LA', 'Volume': 8},
    {'id': 3, 'awb_number': 'AWB125', 'container': 2, 'position': (0, 0, 0, 2, 2, 2), 'DestinationCode': 'SF', 'Volume': 8}
]

# Function to create 3D plot for a specific container
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

# Dash App Layout
app.layout = html.Div([
    html.H1("Container Product Visualization"),
    
    # Input field for container ID
    html.Div([
        html.Label("Enter Container ID:"),
        dcc.Input(id="container-id-input", type="number", value=1, min=1, step=1),
        html.Button("Show Container", id="show-container-button", n_clicks=0),
    ], style={'margin-bottom': '20px'}),
    
    # Graph for displaying the container plot
    dcc.Graph(id="container-plot")
])

# Dash Callback to update plot based on container ID input
@app.callback(
    Output("container-plot", "figure"),
    Input("container-id-input", "value"),
    Input("show-container-button", "n_clicks")
)
def update_plot(container_id, n_clicks):
    if n_clicks > 0:
        # Find the container by ID
        container = next((c for c in containers if c['id'] == container_id), None)
        if container:
            # Create plot for the specific container
            return create_3d_plot(container, placed_products)
        else:
            return {
                'data': [],
                'layout': go.Layout(title="Container not found!")
            }
    return {}

if __name__ == "__main__":
    app.run_server(debug=True)
