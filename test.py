from ortools.linear_solver import pywraplp
import plotly.graph_objects as go
import random

# Initialize solver
solver = pywraplp.Solver.CreateSolver('SCIP')

# Define container dimensions
container_length = 10
container_width = 10
container_height = 10

# Define package dimensions
packages = [
    {'id': 1, 'length': 2, 'width': 2, 'height': 2},
    {'id': 2, 'length': 3, 'width': 3, 'height': 3},
]

# Step 1: Validate total package volume vs container volume
total_volume = sum([p['length'] * p['width'] * p['height'] for p in packages])
container_volume = container_length * container_width * container_height
if total_volume > container_volume:
    print("Combined package volume exceeds container capacity!")
    exit()

# Define variables for package positions
x, y, z = {}, {}, {}
for package in packages:
    x[package['id']] = solver.IntVar(0, container_length, f"x_{package['id']}")
    y[package['id']] = solver.IntVar(0, container_width, f"y_{package['id']}")
    z[package['id']] = solver.IntVar(0, container_height, f"z_{package['id']}")

# Add constraints: Packages must fit inside the container
for package in packages:
    solver.Add(x[package['id']] + package['length'] <= container_length)
    solver.Add(y[package['id']] + package['width'] <= container_width)
    solver.Add(z[package['id']] + package['height'] <= container_height)

# Add non-overlap constraints
for i, package1 in enumerate(packages):
    for j, package2 in enumerate(packages):
        if i >= j:
            continue  # Avoid redundant checks

        # Create Boolean variables for non-overlapping conditions
        left = solver.BoolVar(f"left_{package1['id']}_{package2['id']}")
        right = solver.BoolVar(f"right_{package1['id']}_{package2['id']}")
        front = solver.BoolVar(f"front_{package1['id']}_{package2['id']}")
        back = solver.BoolVar(f"back_{package1['id']}_{package2['id']}")
        below = solver.BoolVar(f"below_{package1['id']}_{package2['id']}")
        above = solver.BoolVar(f"above_{package1['id']}_{package2['id']}")

        # Non-overlap in x-direction
        solver.Add(x[package1['id']] + package1['length'] <= x[package2['id']] + left * container_length)
        solver.Add(x[package2['id']] + package2['length'] <= x[package1['id']] + right * container_length)

        # Non-overlap in y-direction
        solver.Add(y[package1['id']] + package1['width'] <= y[package2['id']] + front * container_width)
        solver.Add(y[package2['id']] + package2['width'] <= y[package1['id']] + back * container_width)

        # Non-overlap in z-direction
        solver.Add(z[package1['id']] + package1['height'] <= z[package2['id']] + below * container_height)
        solver.Add(z[package2['id']] + package2['height'] <= z[package1['id']] + above * container_height)

        # Ensure at least one separation condition is satisfied
        solver.Add(left + right + front + back + below + above >= 1)

# Objective function: Maximize volume utilization
solver.Maximize(solver.Sum([
    package['length'] * package['width'] * package['height'] for package in packages
]))

# Solve
status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL:
    print("Solution Found:")
    for package in packages:
        print(f"Package {package['id']} \nPosition: ({x[package['id']].solution_value()}, "
              f"{y[package['id']].solution_value()}, {z[package['id']].solution_value()})")
else:
    print("No solution found.")
    exit()

# Visualization using Plotly
fig = go.Figure()

# Add container
fig.add_trace(go.Mesh3d(
    x=[0, container_length, container_length, 0, 0, container_length, container_length, 0],
    y=[0, 0, container_width, container_width, 0, 0, container_width, container_width],
    z=[0, 0, 0, 0, container_height, container_height, container_height, container_height],
    color='lightblue',
    opacity=0.5
))

# Generate random colors for packages
def random_color():
    return f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

# Add packages to the visualization
for package in packages:
    px_start = x[package['id']].solution_value()
    px_end = px_start + package['length']
    py_start = y[package['id']].solution_value()
    py_end = py_start + package['width']
    pz_start = z[package['id']].solution_value()
    pz_end = pz_start + package['height']
    
    fig.add_trace(go.Mesh3d(
        x=[px_start, px_end, px_end, px_start, px_start, px_end, px_end, px_start],
        y=[py_start, py_start, py_end, py_end, py_start, py_start, py_end, py_end],
        z=[pz_start, pz_start, pz_start, pz_start, pz_end, pz_end, pz_end, pz_end],
        color=random_color(),
        opacity=0.8
    ))

fig.show()
