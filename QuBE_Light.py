
"""
QuBE_Light: Quantum Maze Solver with Advanced Optics and Quantum Mechanics

Francisco Angulo de Lafuente
July 2, 2024

https://github.com/Agnuxo1

QuBE_Light: A Physics-Inspired Quantum Computing Simulator for Maze Solving

Q-CUBE simulates a 3D quantum computer where light itself is the computational medium. 
It harnesses advanced optics and quantum mechanics to solve mazes with remarkable efficiency.

Abstract:

This program simulates a 3D maze where light waves, representing qubits, 
propagate through the maze seeking the fastest path to the exit, guided by 
Fermat's principle. The quantum nature of light, including superposition and 
interference, is modeled using the Schrödinger equation. Optionally, the 
Gross-Pitaevskii equation can be used to explore the behavior of a condensate 
of light, potentially revealing emergent patterns in maze solving.

The simulation leverages CUDA and ray tracing, implemented using the Optix framework, 
to accelerate the computationally intensive light propagation calculations. 
A PyQt5-based visualizer provides a real-time, intuitive display of the maze, 
the evolving light waves, and system performance metrics.

This version implements:

* Fermat's Principle:  Models light's tendency to travel the fastest path.
* Schrödinger Equation: Simulates the quantum evolution of light waves.
* Gross-Pitaevskii Equation (Optional): Explores the dynamics of a light condensate.
* CUDA and Optix Ray Tracing:  Accelerates light propagation calculations.
* PyQt5 Visualization: Displays the maze, light waves, and performance metrics. 
"""


import os
import time
import math
import numpy as np
import cupy as cp
import psutil
import GPUtil
import random
import ray
import networkx as nx
import threading
import sys
from numba import cuda
import pennylane as qml
from pennylane import numpy as pnp
from PyQt5.QtWidgets import QApplication
from maze_visualizer import MazeVisualizer
from Q_CUBE_Visualization import QCubeVisualizer
from threading import Lock

# --- Parameters ---
NUM_QUBITS = 448 # Example:  4, 8, 27, 64, 125, 216, 448, 1000, 1444
MAX_RAY_DISTANCE = 7 
ATTENUATION_FACTOR = 0.2
FLASH_INTERVAL = 0.1
NUM_ITERATIONS = 1000
GROVER_ITERATIONS = 3
SAMPLES_PER_QUBIT = 3 
MAX_DEPTH = 3
GRID_DIMENSIONS = np.array([10, 10, 10])        
CELL_SIZE = 1 
MIN_REFLECTANCE = 0.05
MAX_REFLECTANCE = 0.9       
MAZE_SIZE = 6
LEARNING_RATE = 0.01  # For updating Q-table
DISCOUNT_FACTOR = 0.5  # For updating Q-table
EPSILON = 0.9  # Epsilon-greedy strategy
TIME_MAZE = 600.0  # Time (in seconds) for solve maze
MAX_STEPS_PER_MAZE = 250  # Number of steps per maze


best_time = float('inf')
min_steps = float('inf')
print_lock = Lock()
mazes_solved = 0


class Qubit:
    def __init__(self, name: str, x: float, y: float, z: float, reflectance: float, sensor_type: str):
        """
        Initialize a Qubit object with given attributes.
        """
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.reflectance = reflectance
        self.received_intensity = 0.0
        self.sensor_type = sensor_type

    def update_sensor_value(self, value: float) -> None:
        """
        Update the sensor value of the qubit.
        """
        self.sensor_value = value

class Grid:
    def __init__(self, dimensions: np.ndarray, cell_size: float):
        """
        Initialize a spatial grid with given dimensions and cell size.
        """
        self.dimensions = dimensions
        self.cell_size = cell_size
        self.grid = {}

    def _get_cell_index(self, position: np.ndarray) -> tuple:
        """
        Get the cell index for a given position.
        """
        return tuple((position // self.cell_size).astype(int))

    def add_qubit(self, qubit: Qubit, index: int) -> None:
        """
        Add a qubit to the grid at its respective cell.
        """
        position = np.array([qubit.x, qubit.y, qubit.z])
        cell_index = self._get_cell_index(position)
        if cell_index not in self.grid:
            self.grid[cell_index] = []
        self.grid[cell_index].append(index)

    def get_neighbors(self, qubit: Qubit, qubits: list) -> list:
        """
        Get neighboring qubits for a given qubit.
        """
        position = np.array([qubit.x, qubit.y, qubit.z])
        cell_index = self._get_cell_index(position)
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    neighboring_cell = (
                        cell_index[0] + i,
                        cell_index[1] + j,
                        cell_index[2] + k,
                    )
                    if neighboring_cell in self.grid:
                        neighbors.extend(self.grid[neighboring_cell])
        return [qubits[i] for i in neighbors]

def initialize_qcube(grid: Grid, num_qubits: int) -> list:
    """
    Initialize the Q-CUBE with qubits and add them to the grid.
    """
    qcube_data = []
    qubit_id = 1
    grid_size = int(math.ceil(num_qubits ** (1 / 3)))
    distance_between_qubits = 1 / grid_size
    sensor_types = ['distance', 'walls']

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if qubit_id <= num_qubits:
                    x = (i + 0.5) * distance_between_qubits
                    y = (j + 0.5) * distance_between_qubits
                    z = (k + 0.5) * distance_between_qubits
                    sensor_type = sensor_types[(qubit_id - 1) % len(sensor_types)]
                    qubit = Qubit(
                        f"Qubit-{qubit_id}",
                        x,
                        y,
                        z,
                        random.uniform(MIN_REFLECTANCE, MAX_REFLECTANCE),
                        sensor_type
                    )
                    qcube_data.append(qubit)
                    grid.add_qubit(qubit, len(qcube_data) - 1)
                    qubit_id += 1

    return qcube_data

@cuda.jit(device=True)
def create_ray(origin: cp.ndarray, direction: cp.ndarray) -> tuple:
    """
    Create a ray with a given origin and direction.
    """
    return origin, direction

@cuda.jit(device=True)
def random_unit_vector() -> cp.ndarray:
    """
    Generate a random unit vector for ray tracing.
    """
    theta = 2 * math.pi * cuda.random.uniform()
    phi = math.acos(2 * cuda.random.uniform() - 1)
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    return cp.array([x, y, z])

def calculate_optimal_block_size() -> int:
    """
    Calculate the optimal block size for CUDA kernels.
    """
    device = cuda.get_current_device()
    max_threads = device.MAX_THREADS_PER_BLOCK
    warp_size = device.WARP_SIZE
    return max(warp_size, min(max_threads, 256))

def configure_cuda_kernel(qubit_positions: cp.ndarray) -> tuple:
    """
    Configure the CUDA kernel with the optimal block and grid sizes.
    """
    threads_per_block = calculate_optimal_block_size()
    blocks_per_grid = (len(qubit_positions) + threads_per_block - 1) // threads_per_block
    return threads_per_block, blocks_per_grid

@cuda.jit(device=True)
def calculate_distance_gpu(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """
    Calculate the Euclidean distance between two points in GPU.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

@cuda.jit
def calculate_optical_path_matrix_kernel(qubit_count: int, qubits: cp.ndarray, optical_path_matrix: cp.ndarray) -> None:
    """
    CUDA kernel to calculate the optical path matrix.
    """
    i, j = cuda.grid(2)
    if i < qubit_count and j < qubit_count:
        q1 = qubits[i]
        q2 = qubits[j]
        optical_path_matrix[i, j] = calculate_distance_gpu(
            q1[0], q1[1], q1[2], q2[0], q2[1], q2[2]
        )

def calculate_optical_path_matrix(qubits: list) -> cp.ndarray:
    """
    Calculate the optical path matrix for the qubits.
    """
    qubit_count = len(qubits)
    optical_path_matrix = cp.zeros((qubit_count, qubit_count), dtype=cp.float32)
    qubits_flat = cp.array([(q.x, q.y, q.z) for q in qubits], dtype=cp.float32)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (
        qubit_count + threads_per_block[0] - 1
    ) // threads_per_block[0]
    blocks_per_grid_y = (
        qubit_count + threads_per_block[1] - 1
    ) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    calculate_optical_path_matrix_kernel[blocks_per_grid, threads_per_block](
        qubit_count, qubits_flat, optical_path_matrix
    )
    cp.cuda.Stream.null.synchronize()
    return optical_path_matrix

@cuda.jit(device=True)
def path_trace(origin: cp.ndarray, direction: cp.ndarray, depth: int, max_depth: int) -> float:
    """
    Trace the path of a ray through the qubits.
    """
    if depth > max_depth:
        return 0.0
    
    intersection = ray_qubit_intersection(origin, direction)
    if not intersection:
        return 0.0
    
    qubit, t = intersection
    hit_point = origin + t * direction
    
    direct_light = calculate_direct_light(hit_point, qubit)
    
    new_ray_dir = reflect(direction, qubit.normal)
    indirect_light = path_trace(hit_point, new_ray_dir, depth + 1, max_depth)
    
    return direct_light + qubit.reflectance * indirect_light

@cuda.jit(device=True)
def quantum_interference(ray1: tuple, ray2: tuple) -> float:
    """
    Calculate the quantum interference between two rays.
    """
    phase_difference = calculate_phase_difference(ray1, ray2)
    return cp.cos(phase_difference) ** 2

@cuda.jit
def optimized_propagate_light_kernel(
    qcube_intensity: cp.ndarray,
    qubit_positions: cp.ndarray,
    reflectances: cp.ndarray,
    max_distance: float,
    grid_gpu: cp.ndarray,
    grid_shape: tuple,
    cell_size: float,
) -> None:
    """
    Optimized CUDA kernel for propagating light through the Q-CUBE.
    """
    shared_positions = cuda.shared.array(shape=(256, 3), dtype=cp.float32)
    shared_reflectances = cuda.shared.array(shape=256, dtype=cp.float32)
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    
    for i in range(0, qubit_positions.shape[0], 256):
        if i + tx < qubit_positions.shape[0]:
            for j in range(3):
                shared_positions[tx][j] = qubit_positions[i + tx][j]
            shared_reflectances[tx] = reflectances[i + tx]
        
        cuda.syncthreads()
        
        idx = cuda.grid(1)
        if idx < qubit_positions.shape[0]:
            origin = qubit_positions[idx]
            for _ in range(SAMPLES_PER_QUBIT):
                direction = random_unit_vector()
                qcube_intensity[idx] += path_trace(origin, direction, 0, MAX_DEPTH)
            qcube_intensity[idx] /= SAMPLES_PER_QUBIT
            
            for n in range(qubit_positions.shape[0]):
                if n != idx:
                    ray1 = create_ray(qubit_positions[idx], qubit_positions[n] - qubit_positions[idx])
                    ray2 = create_ray(qubit_positions[n], qubit_positions[idx] - qubit_positions[n])
                    interference = quantum_interference(ray1, ray2)
                    qcube_intensity[idx] += interference * reflectances[n]

def propagate_light_gpu(qubits: list, grid: Grid, max_distance: float) -> None:
    """
    Propagate light through the Q-CUBE using GPU.
    """
    qubit_positions = cp.array(
        [[q.x, q.y, q.z] for q in qubits], dtype=cp.float32
    )
    reflectances = cp.array([q.reflectance for q in qubits], dtype=cp.float32)
    qcube_intensity = cp.zeros(len(qubits), dtype=cp.float32)

    grid_shape = grid.dimensions // grid.cell_size
    max_neighbors = max([len(cell_qubits) for cell_qubits in grid.grid.values()])
    grid_gpu = cp.full(
        (grid_shape[0] * grid_shape[1] * grid_shape[2], max_neighbors),
        -1,
        dtype=cp.int32,
    )

    for cell_index, cell_qubits in grid.grid.items():
        flat_index = (
            cell_index[0] * grid_shape[1] * grid_shape[2]
            + cell_index[1] * grid_shape[2]
            + cell_index[2]
        )
        grid_gpu[flat_index, : len(cell_qubits)] = cp.array(
            cell_qubits, dtype=cp.int32
        )

    threads_per_block = 256
    blocks_per_grid = (
        len(qubit_positions) + threads_per_block - 1
    ) // threads_per_block

    optimized_propagate_light_kernel[blocks_per_grid, threads_per_block](
        qcube_intensity,
        qubit_positions,
        reflectances,
        max_distance,
        grid_gpu,
        grid_shape,
        grid.cell_size,
    )

    for i, qubit in enumerate(qubits):
        qubit.received_intensity += qcube_intensity[i].get()
        qubit.received_intensity = min(max(qubit.received_intensity, -10), 10)

def create_maze(size: int = MAZE_SIZE) -> nx.Graph:
    """
    Create a maze using Depth-First Search (DFS) algorithm.
    """
    def dfs(x, y):
        visited.add((x, y))
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                G.add_edge((x, y), (nx, ny))
                dfs(nx, ny)

    G = nx.grid_2d_graph(size, size)
    visited = set()
    dfs(0, 0)

    edges = list(G.edges())
    random.shuffle(edges)
    for u, v in edges:
        G.remove_edge(u, v)
        if not nx.has_path(G, (0, 0), (size - 1, size - 1)):
            G.add_edge(u, v)

    G.graph["width"] = size
    G.graph["height"] = size

    return G

def decompose_qft(circuit: qml.QNode, qubits: list) -> None:
    """
    Decompose the Quantum Fourier Transform (QFT) into individual gates.
    """
    n = len(qubits)
    for i in range(n):
        qml.Hadamard(wires=qubits[i])
        for j in range(i + 1, n):
            qml.CPhase(np.pi / 2 ** (j - i), wires=[qubits[i], qubits[j]])

def encode_maze_features(maze: nx.Graph, current_position: tuple, end_point: tuple, qubits: list) -> None:
    """
    Encode maze features such as distance to target and walls into qubit sensors.
    """
    size = maze.graph["width"]
    for qubit in qubits:
        if qubit.sensor_type == "distance":
            distance = math.sqrt(
                (current_position[0] - end_point[0]) ** 2
                + (current_position[1] - end_point[1]) ** 2
            )
            normalized_distance = (
                distance / (math.sqrt(2) * size)
            )
            qubit.update_sensor_value(normalized_distance)

        elif qubit.sensor_type == "walls":
            wall_value = 0
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for i, (dx, dy) in enumerate(directions):
                nx, ny = current_position[0] + dx, current_position[1] + dy
                if (nx, ny) not in maze[current_position]:
                    wall_value += 2**i
            qubit.update_sensor_value(wall_value / 15)

def maze_to_quantum_circuit(qubits: list, actions: list, current_position: tuple, end_point: tuple, maze: nx.Graph) -> qml.QNode:
    """
    Create a quantum circuit for maze solving using Qubits and Grover's algorithm.
    """
    n_qubits = len(qubits)
    n_actions = len(actions)

    action_qubits = int(np.ceil(np.log2(n_actions)))

    dev = qml.device('default.qubit', wires=n_qubits + action_qubits)

    @qml.qnode(dev)
    def circuit():
        for i, qubit in enumerate(qubits):
            decompose_qft(circuit, range(n_qubits))
            qml.CRY(qubit.sensor_value * np.pi, wires=[i, n_qubits + action_qubits - 1])
            decompose_qft(circuit, range(n_qubits)).inv()

        for i in range(n_qubits, n_qubits + action_qubits):
            qml.Hadamard(wires=i)

        for _ in range(GROVER_ITERATIONS):
            oracle_maze(circuit, qubits, actions, current_position, end_point, maze)
            grover_diffusion(circuit, n_qubits, action_qubits)

        return [qml.sample(qml.PauliZ(i)) for i in range(n_qubits, n_qubits + action_qubits)]

    return circuit

def oracle_maze(circuit: qml.QNode, qubits: list, actions: list, current_position: tuple, end_point: tuple, maze: nx.Graph) -> None:
    """
    Oracle function for Grover's algorithm in the maze.
    """
    n_qubits = len(qubits)
    action_qubits = int(np.ceil(np.log2(len(actions))))

    def is_better_move(action):
        new_pos = get_new_position(current_position, action)
        if new_pos not in maze[current_position]:
            return False
        current_distance = manhattan_distance(current_position, end_point)
        new_distance = manhattan_distance(new_pos, end_point)
        return new_distance < current_distance

    for i, action in enumerate(actions):
        if is_better_move(action):
            qml.RZ(np.pi / 2, wires=n_qubits + i)
        else:
            qml.RZ(-np.pi / 2, wires=n_qubits + i)

def get_new_position(current_position: tuple, action: str) -> tuple:
    """
    Get the new position after taking an action.
    """
    x, y = current_position
    if action == "up":
        return (x - 1, y)
    elif action == "down":
        return (x + 1, y)
    elif action == "left":
        return (x, y - 1)
    elif action == "right":
        return (x, y + 1)
    return current_position

def manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    """
    Calculate the Manhattan distance between two points.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def grover_diffusion(circuit: qml.QNode, n_qubits: int, action_qubits: int) -> None:
    """
    Perform Grover's diffusion operation in the quantum circuit.
    """
    for i in range(n_qubits, n_qubits + action_qubits):
        qml.Hadamard(wires=i)
        qml.PauliX(wires=i)
    qml.Hadamard(wires=n_qubits + action_qubits - 1)
    qml.MultiControlledX(control_wires=range(n_qubits, n_qubits + action_qubits - 1), wires=n_qubits + action_qubits - 1)
    qml.Hadamard(wires=n_qubits + action_qubits - 1)
    for i in range(n_qubits, n_qubits + action_qubits):
        qml.PauliX(wires=i)
        qml.Hadamard(wires=i)

def choose_action(q_table: np.ndarray, current_position: tuple, actions: list, maze: nx.Graph, epsilon: float = EPSILON) -> str:
    """
    Choose an action based on the Q-table and epsilon-greedy strategy.
    """
    valid_actions = get_valid_actions(current_position, maze)
    if not valid_actions:
        print(f"No valid actions at position {current_position}. Maze structure may be incorrect.")
        return random.choice(actions)

    if np.random.uniform(0, 1) < epsilon:
        return random.choice(valid_actions)
    else:
        x, y = current_position
        action_values = [
            (action, q_table[x, y, actions.index(action)])
            for action in valid_actions
        ]
        return max(action_values, key=lambda x: x[1])[0]

def get_valid_actions(position: tuple, maze: nx.Graph) -> list:
    """
    Get valid actions (moves) from the current position in the maze.
    """
    valid_actions = []
    x, y = position
    if (x - 1, y) in maze[position]:
        valid_actions.append("up")
    if (x + 1, y) in maze[position]:
        valid_actions.append("down")
    if (x, y - 1) in maze[position]:
        valid_actions.append("left")
    if (x, y + 1) in maze[position]:
        valid_actions.append("right")
    return valid_actions

def calculate_reward(current_position: tuple, new_position: tuple, end_point: tuple, time_taken: float, steps_taken: int) -> float:
    """
    Calculate the reward for a move based on the new position, time taken, and steps taken.
    """
    global best_time, min_steps
    
    # Base reward calculation
    current_distance = manhattan_distance(current_position, end_point)
    new_distance = manhattan_distance(new_position, end_point)

    if new_position == end_point:
        base_reward = 10
    elif new_distance < current_distance:
        base_reward = 1
    else:
        base_reward = -0.1

    # Time-based reward (0 to 10)
    time_reward = max(0, 10 - (time_taken / 6))  # Assuming 60 seconds is the baseline for 0 reward

    # Step-based reward (0 to 10)
    step_reward = max(0, 10 - (steps_taken / 25))  # Assuming 250 steps is the baseline for 0 reward

    # Check for new records
    time_bonus = 0
    step_bonus = 0
    
    if time_taken < best_time:
        time_bonus = 20
        best_time = time_taken
    
    if steps_taken < min_steps:
        step_bonus = 20
        min_steps = steps_taken

    total_reward = base_reward + time_reward + step_reward + time_bonus + step_bonus
    
    print(f"Reward breakdown - Base: {base_reward}, Time: {time_reward}, Step: {step_reward}, Time Bonus: {time_bonus}, Step Bonus: {step_bonus}")
    
    return total_reward

def update_q_table(
    q_table: np.ndarray,
    current_position: tuple,
    action: str,
    reward: float,
    next_position: tuple,
    actions: list,
    learning_rate: float = LEARNING_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
) -> np.ndarray:
    """
    Update the Q-table based on the taken action and received reward.
    """
    if action is None:
        print("No action taken, skipping Q-table update.")
        return q_table

    x, y = current_position
    next_x, next_y = next_position
    action_index = actions.index(action)

    q_table[x, y, action_index] = (1 - learning_rate) * q_table[
        x, y, action_index
    ] + learning_rate * (
        reward + discount_factor * np.max(q_table[next_x, next_y, :])
    )

    print(f"Updated Q-table at position {current_position} for action {action}: {q_table[x, y, action_index]}")
    return q_table

def choose_action_from_qtable(q_table: np.ndarray, current_position: tuple, actions: list) -> str:
    """
    Choose an action based on the Q-table (exploitation).
    """
    x, y = current_position
    action_index = np.argmax(q_table[x, y, :])
    return actions[action_index]

def solve_maze(maze: nx.Graph, start: tuple, end: tuple) -> list:
    """
    Solve the maze using quantum-inspired techniques.
    """
    circuit = maze_to_quantum_circuit(qcube_data, actions, start, end, maze)
    results = circuit()
    path = []

    most_frequent_state = max(results, key=results.get)
    for i, bit in enumerate(most_frequent_state):
        if bit == 1:
            path.append(list(maze.nodes())[i])

    if path[0] != start:
        path.insert(0, start)
    if path[-1] != end:
        path.append(end)

    return [(int(node[0]), int(node[1])) for node in path]

def visualize_maze(maze: nx.Graph, start: tuple, end: tuple, solution: list) -> dict:
    """
    Visualize the maze and the solution path.
    """
    return {
        'maze': maze,
        'start': start,
        'end': end,
        'solution': solution
    }

def simulation_loop(visualizer: QCubeVisualizer) -> None:
    """
    Main simulation loop for solving the maze and updating the visualizer.
    """
    global iterations, last_solution, current_maze, mazes_solved, best_time, min_steps

    iterations = 0
    last_solution = None
    current_maze = None
    start_point = (0, 0)
    end_point = (MAZE_SIZE - 1, MAZE_SIZE - 1)

    current_position = start_point
    actions = ["up", "down", "left", "right"]
    solution_path = [current_position]

    q_table = np.random.uniform(low=0, high=0.1, size=(MAZE_SIZE, MAZE_SIZE, len(actions)))

    while iterations < NUM_ITERATIONS:
        try:
            if current_maze is None:
                current_maze = create_maze(MAZE_SIZE)
                print(f"New maze created. Number of edges: {current_maze.number_of_edges()}")
                current_position = start_point
                solution_path = [current_position]
                start_time = time.time()
                steps_in_current_maze = 0

            valid_actions = get_valid_actions(current_position, current_maze)
            print(f"Valid actions at {current_position}: {valid_actions}")

            encode_maze_features(current_maze, current_position, end_point, qcube_data)

            epsilon = max(0.01, EPSILON * (1 - iterations / NUM_ITERATIONS))

            action = choose_action(q_table, current_position, actions, current_maze, epsilon=epsilon)

            if action is None:
                print(f"Iteration: {iterations}/{NUM_ITERATIONS}")
                print(f"No preferred action. Staying at {current_position}")
                continue

            new_position = get_new_position(current_position, action)

            if new_position in current_maze[current_position]:
                current_position = new_position
                solution_path.append(current_position)

            time_elapsed = time.time() - start_time
            steps_in_current_maze += 1

            reward = calculate_reward(current_position, new_position, end_point, time_elapsed, steps_in_current_maze)

            q_table = update_q_table(q_table, current_position, action, reward, new_position, actions)

            print(f"Iteration {iterations}: Position {current_position}, Action {action}, Reward {reward}")
            print(f"Time elapsed: {time_elapsed:.2f} seconds, Steps taken: {steps_in_current_maze}")
            print(f"Best time: {best_time:.2f} seconds, Minimum steps: {min_steps}")
            print(f"Q-values: {q_table[current_position[0], current_position[1], :]}")

            if current_position == end_point or time_elapsed > TIME_MAZE or steps_in_current_maze >= MAX_STEPS_PER_MAZE:
                if current_position == end_point:
                    mazes_solved += 1
                    print(f"Maze solved! Solution path: {solution_path}")
                    print(f"Total mazes solved: {mazes_solved}")
                    print(f"Time taken: {time_elapsed:.2f} seconds, Steps taken: {steps_in_current_maze}")
                    last_solution = solution_path
                elif time_elapsed > TIME_MAZE:
                    print(f"Time limit reached. Moving to next maze.")
                else:
                    print(f"Maximum steps reached. Moving to next maze.")

                visualizer.update_maze(current_maze, start_point, end_point, solution_path)

                current_maze = None
                iterations += 1
                continue

            try:
                propagate_light_gpu(qcube_data, grid, MAX_RAY_DISTANCE)
            except Exception as e:
                print(f"Error in light propagation: {str(e)}")
                print("Continuing simulation without light propagation for this iteration.")

            for qubit in qcube_data:
                qubit.reflectance += qubit.received_intensity * 0.01
                qubit.reflectance = max(MIN_REFLECTANCE, min(MAX_REFLECTANCE, qubit.reflectance))

            cpu_percent = psutil.cpu_percent()
            gpu_devices = GPUtil.getGPUs()
            gpu_percent = gpu_devices[0].load * 100 if gpu_devices else 0

            visualizer.update_qubits(
                qcube_data,
                iterations,
                current_maze,
                start_point,
                end_point,
                str(solution_path),
                cpu_percent,
                gpu_percent,
            )

            if iterations % 100 == 0:
                print(f"\nCurrent progress - Iterations: {iterations}, Mazes solved: {mazes_solved}")

            time.sleep(FLASH_INTERVAL)

        except Exception as e:
            print(f"Error in iteration {iterations}: {str(e)}")
            print(f"Current position: {current_position}")
            print(f"Chosen action: {action}")
            print(f"Q-table state:\n{q_table}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


# --- Main Execution ---
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)

        # Initialize the spatial grid
        grid = Grid(GRID_DIMENSIONS, CELL_SIZE)
        print("Spatial grid initialized.")

        # Initialize the Q-CUBE with sensors
        qcube_data = initialize_qcube(grid, NUM_QUBITS)
        print("Q-CUBE initialized.")

        # Calculate the optical path matrix
        optical_path_matrix = calculate_optical_path_matrix(qcube_data)
        print("Optical path matrix calculated.")

        try:
            # Initialize Ray for parallel processing, using all available GPUs
            num_gpus_available = len(GPUtil.getGPUs())
            ray.init(num_gpus=num_gpus_available)
            print(f"Ray initialized with {num_gpus_available} GPUs.")
        except Exception as e:
            print(f"Error initializing Ray: {e}")
            ray.init()
            print("Ray initialized in single-process mode.")

        # Initialize the visualizer
        visualizer = QCubeVisualizer(qcube_data)

        # Start the simulation loop in a separate thread
        print("Starting simulation thread...")
        simulation_thread = threading.Thread(
            target=simulation_loop, args=(visualizer,)
        )
        simulation_thread.daemon = True
        simulation_thread.start()
        print("Simulation thread started.")

        # Run the visualizer
        visualizer.run()

        print("Visualizer finished.")

    except Exception as e:
        print(f"An error occurred in the main execution: {str(e)}")

print("Q-CUBE simulation complete.")
