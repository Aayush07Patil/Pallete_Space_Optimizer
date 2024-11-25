import vtk
from py3dbp import Packer, Bin, Item
import random

# Initialize Packer and add bin and items
packer = Packer()
packer.add_bin(Bin('Pallete_1', 10, 10, 10, 100.0))

packer.add_item(Item('50g [powder 1]', 3.9370, 1.9685, 1.9685, 1))
packer.add_item(Item('50g [powder 2]', 3.9370, 1.9685, 1.9685, 2))
packer.add_item(Item('50g [powder 3]', 3.9370, 1.9685, 1.9685, 3))
packer.add_item(Item('250g [powder 4]', 7.8740, 3.9370, 1.9685, 4))
packer.add_item(Item('250g [powder 5]', 7.8740, 3.9370, 1.9685, 5))
packer.add_item(Item('250g [powder 6]', 7.8740, 3.9370, 1.9685, 6))
packer.add_item(Item('250g [powder 7]', 7.8740, 3.9370, 1.9685, 7))
packer.add_item(Item('250g [powder 8]', 7.8740, 3.9370, 1.9685, 8))
packer.add_item(Item('250g [powder 9]', 7.8740, 3.9370, 1.9685, 9))

# Perform the packing
packer.pack(bigger_first=True, distribute_items=False, number_of_decimals=3)

for b in packer.bins:
    print(":::::::::::", b.string())

    print("FITTED ITEMS:")
    for item in b.items:
        print("====> ", item.string())

    print("UNFITTED ITEMS:")
    for item in b.unfitted_items:
        print("====> ", item.string())

    print("***************************************************")
    print("***************************************************")


# Create a renderer, render window, and interactor
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Create a function to add colored boxes to the render window
def create_box(x, y, z, lx, ly, lz, color):
    # Create a cube
    cube = vtk.vtkCubeSource()
    cube.SetXLength(lx)
    cube.SetYLength(ly)
    cube.SetZLength(lz)
    
    # Create a mapper
    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputConnection(cube.GetOutputPort())
    
    # Create an actor
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    
    # Set position
    cube_actor.SetPosition(x, y, z)
    
    # Set color
    cube_actor.GetProperty().SetColor(color)
    
    # Add the actor to the renderer
    renderer.AddActor(cube_actor)

# Add a bin (as a large box) to represent the space
create_box(0, 0, 0, 5, 5, 10, (0.8, 0.8, 0.8))  # Light grey color for the bin

# Function to generate a random color
def random_color():
    return [random.random() for _ in range(3)]  # Random RGB color

# Loop through bins and items to create 3D objects with unique colors
for b in packer.bins:
    for item in b.items:
        x0, y0, z0 = item.position
        lx, ly, lz = item.width, item.height, item.depth
        
        # Ensure that we convert Decimal values to float
        x0, y0, z0 = float(x0), float(y0), float(z0)
        lx, ly, lz = float(lx), float(ly), float(lz)
        
        # Assign a random color to each item
        color = random_color()
        
        # Create the box for each item with the unique color
        create_box(x0, y0, z0, lx, ly, lz, color)

# Set the background color of the renderer
renderer.SetBackground(0.1, 0.1, 0.1)  # Dark grey background

# Set the camera position and view angle
renderer.GetActiveCamera().SetPosition(10, 10, 20)
renderer.GetActiveCamera().SetFocalPoint(5, 5, 5)
renderer.GetActiveCamera().SetViewUp(0, 0, 1)

# Start the rendering loop
render_window.Render()
render_window_interactor.Start()
