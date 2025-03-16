import plotly.graph_objects as go
import numpy as np

# Define the vertices of the irregular cuboid # 30, 20, 25,
vertices = np.array([
    [0, 0, 0],  # 0
    [0, 20, 0],  # 1
    [0, 0, 25],  # 2
    [0, 15, 25],  # 3
    [0, 20, 22.5],  # 4
    [30, 0, 0],  # 5
    [30, 20, 0],  # 6
    [30, 0, 25],  # 7
    [30, 15, 25],  # 8
    [30, 20, 22.5]  # 9
])

# Define triangular faces using vertex indices
triangles = [
    [0, 1, 4], [0, 4, 3], [0, 3, 2],  # Left side
    [5, 6, 9], [5, 9, 8], [5, 8, 7],  # Right side
    [8, 9, 4], [8, 4, 3],  # Front
    [9, 6, 1], [9, 1, 4],  # Back
    [7, 8, 3], [7, 3, 2],  # Top
    [5, 6, 1], [5, 1, 0],  # Bottom
    [7, 5, 0], [7, 0, 2]   # Another side
]

# Unpack triangle indices into i, j, k
i, j, k = zip(*triangles)

# Define edges (pairs of vertex indices)
edges = [
    [0, 1], [1, 4], [4, 3], [3, 0], [0, 2], [2, 3],  # Left side edges
    [5, 6], [6, 9], [9, 8], [8, 5], [5, 7], [7, 8],  # Right side edges
    [3, 8], [4, 9], [1, 6],  # Connecting edges
    [2, 7], [0, 5]  # Connecting edges
]

# Create edge lines
edge_x, edge_y, edge_z = [], [], []
for edge in edges:
    for point in edge:
        edge_x.append(vertices[point][0])
        edge_y.append(vertices[point][1])
        edge_z.append(vertices[point][2])
    edge_x.append(None)  # Break the line between segments
    edge_y.append(None)
    edge_z.append(None)

# Create figure with Mesh3d for faces and Scatter3d for edges
fig = go.Figure()

# Add 3D mesh
fig.add_trace(go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=i, j=j, k=k,
    color='blue',
    opacity=1
))

# Add black edges
fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='black', width=4),
    name='Edges'
))

# Set plot layout
fig.update_layout(
    title="Irregular Cuboid with Black Edges",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

# Show plot
fig.show()


