import plotly.graph_objects as go
import numpy as np

def plot_irregular_cuboid(L, W, H, WX, HX, TB, SD):

    if SD == 'S':
        if TB == 'T':
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
        elif TB == 'B':
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
            opacity=1,
            flatshading=True 
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

    if SD == 'D':
        if TB == 'T':
            W_offset = (W - WX) / 2 
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

        elif TB == 'B':
            W_offset = (W - WX) / 2 
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

        # Define triangular faces
        triangles = [
            [0, 1, 5], [0, 5, 2],  # Left side
            [6, 7, 11], [6, 11, 8], # Right side
            [2, 3, 4], [2, 4, 5],   # Left front slanted
            [8, 9, 10], [8, 10, 11], # Right front slanted
            [3, 9, 10], [3, 10, 4], # Top middle
            [0, 6, 8], [0, 8, 2],  # Bottom left
            [1, 7, 11], [1, 11, 5], # Bottom right
            [3, 9, 6], [3, 6, 0],  # Left front vertical
            [4, 10, 7], [4, 7, 1]  # Right front vertical
        ]

        # Extract (i, j, k) indices
        i, j, k = zip(*triangles)

        # Define edges
        edges = [
            [0, 1], [1, 5], [5, 2], [2, 0], # Left base
            [6, 7], [7, 11], [11, 8], [8, 6], # Right base
            [2, 3], [3, 4], [4, 5], # Left top
            [8, 9], [9, 10], [10, 11], # Right top
            [3, 9], [4, 10], # Connecting edges
            [0, 6], [1, 7], [2, 8], [5, 11] # Vertical edges
        ]

        # Create edge lines
        edge_x, edge_y, edge_z = [], [], []
        for edge in edges:
            for point in edge:
                edge_x.append(vertices[point][0])
                edge_y.append(vertices[point][1])
                edge_z.append(vertices[point][2])
            edge_x.append(None)  # Break line for visualization
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
            opacity=1,
            flatshading=True 
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
            title="Irregular Cuboid with Fallen Top Edges",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            )
        )

        # Show plot
        fig.show()
# Example usage
plot_irregular_cuboid(L=125, W=96, H=63, WX=75, HX=45, TB='B',SD ='D')
