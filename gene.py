import pygad
import numpy as np
import random

# Container and package dimensions
container_dimensions = (10, 10, 10)
packages = [
    {'id': 1, 'dimensions': (2, 2, 2)},
    {'id': 2, 'dimensions': (3, 3, 3)},
]

# Number of packages
num_packages = len(packages)

# Encode a solution: [x1, y1, z1, r1, x2, y2, z2, r2, ...]
def fitness_function(solution, solution_idx):
    used_space = 0
    positions = []
    for i in range(num_packages):
        x, y, z, rotation = solution[i * 4: (i + 1) * 4]
        length, width, height = rotate_package(packages[i]['dimensions'], rotation)
        if (
            0 <= x <= container_dimensions[0] - length
            and 0 <= y <= container_dimensions[1] - width
            and 0 <= z <= container_dimensions[2] - height
            and not overlaps(positions, x, y, z, length, width, height)
        ):
            positions.append((x, y, z, length, width, height))
            used_space += length * width * height
    return used_space

def rotate_package(dimensions, rotation):
    length, width, height = dimensions
    if rotation == 0:
        return length, width, height
    elif rotation == 1:
        return width, height, length
    elif rotation == 2:
        return height, length, width
    elif rotation == 3:
        return width, length, height

def overlaps(existing_positions, x, y, z, length, width, height):
    for pos in existing_positions:
        ex, ey, ez, el, ew, eh = pos
        if not (
            x + length <= ex or ex + el <= x or
            y + width <= ey or ey + ew <= y or
            z + height <= ez or ez + eh <= z
        ):
            return True
    return False

# Genetic algorithm configuration
num_genes = num_packages * 4  # x, y, z, rotation for each package
gene_space = [{'low': 0, 'high': container_dimensions[i % 3]} if i % 4 < 3 else {'low': 0, 'high': 3} for i in range(num_genes)]

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=10,
    fitness_func=fitness_function,
    sol_per_pop=20,
    num_genes=num_genes,
    gene_space=gene_space,
    mutation_type="random",
    mutation_percent_genes=10,
)

# Run the optimization
ga_instance.run()
solution, fitness, _ = ga_instance.best_solution()
print(f"Best Solution: {solution}\nFitness: {fitness}")
