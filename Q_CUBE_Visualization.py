# Q_CUBE_Visualization.py

import sys
import math
import time
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSplitter
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPainter, QColor
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pyqtgraph as pg
import networkx as nx

class Qubit:
    def __init__(self, x, y, z, reflectance):
        self.x = x
        self.y = y
        self.z = z
        self.reflectance = reflectance
        self.received_intensity = 0.0
        self.color = (0.8, 0.8, 0.8)
        self.flash_intensity = 0.0

    def update(self, reflectance, received_intensity):
        self.reflectance = reflectance
        self.received_intensity = received_intensity
        self.color = (
            min(1.0, self.reflectance),
            min(1.0, self.received_intensity / 10),
            0.8
        )
        self.flash_intensity = max(0, self.flash_intensity - 0.1)

    def flash(self):
        self.flash_intensity = 1.0

class QuantumSimulatorWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.qubits = []
        self.camera_distance = 10
        self.camera_angle_x = 0
        self.camera_angle_y = 0
        self.last_mouse_pos = None
        self.rays = []

    def set_qubits(self, qubits):
        self.qubits = [Qubit(q.x, q.y, q.z, q.reflectance) for q in qubits]

    def initializeGL(self):
        glutInit()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Habilitar el mapeo de entorno esférico
        glEnable(GL_TEXTURE_GEN_S)
        glEnable(GL_TEXTURE_GEN_T)
        glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
        glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 50.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_angle_x, 1, 0, 0)
        glRotatef(self.camera_angle_y, 0, 1, 0)

        self.draw_cube()
        self.draw_connections()
        self.draw_rays()

    def draw_cube(self):
        glPushMatrix()
        for qubit in self.qubits:
            glPushMatrix()
            glTranslatef(qubit.x, qubit.y, qubit.z)
        
            # Configurar el material para el efecto de cristal con resplandor suave
            base_color = (0.3, 0.3, 0.5, 0.6)  # Un azul muy suave y translúcido
            glMaterialfv(GL_FRONT, GL_AMBIENT, (base_color[0]*0.4, base_color[1]*0.4, base_color[2]*0.4, base_color[3]))
            glMaterialfv(GL_FRONT, GL_DIFFUSE, base_color)
            glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
            glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
        
            # Dibujar la esfera translúcida
            glutSolidSphere(0.1, 32, 32)
        
            # Añadir un resplandor interno
            glPushAttrib(GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT)
            glDisable(GL_LIGHTING)
            glDepthMask(GL_FALSE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
            glow_color = (0.5, 0.5, 0.7, 0.3)  # Color del resplandor
            glColor4f(*glow_color)
            glutSolidSphere(0.09, 32, 32)  # Esfera interna ligeramente más pequeña
        
            glPopAttrib()
        
            if qubit.flash_intensity > 0:
                glColor4f(1, 1, 1, qubit.flash_intensity * 0.5)
                glutSolidSphere(0.12, 32, 32)
        
            glPopMatrix()
        glPopMatrix()

    def draw_connections(self):
        glPushMatrix()
        glColor4f(0.5, 0.5, 0.5, 0.3)
        glBegin(GL_LINES)
        for i, qubit1 in enumerate(self.qubits):
            for qubit2 in self.qubits[i + 1:]:
                if abs(qubit1.received_intensity - qubit2.received_intensity) < 0.1:
                    glVertex3f(qubit1.x, qubit1.y, qubit1.z)
                    glVertex3f(qubit2.x, qubit2.y, qubit2.z)
        glEnd()
        glPopMatrix()

    def draw_rays(self):
        glPushMatrix()
        glBegin(GL_LINES)
        for ray in self.rays:
            start, end, intensity = ray
            glColor4f(1, 1, 0, intensity)
            glVertex3f(*start)
            glVertex3f(*end)
        glEnd()
        glPopMatrix()

    def add_ray(self, start, end, intensity):
        self.rays.append((start, end, intensity))

    def update_rays(self):
        self.rays = [(start, end, intensity - 0.1) for start, end, intensity in self.rays if intensity > 0.1]

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            self.camera_angle_y += dx * 0.5
            self.camera_angle_x += dy * 0.5
            self.update()
            self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None

    def wheelEvent(self, event):
        zoom_speed = 0.01
        self.camera_distance -= event.angleDelta().y() * zoom_speed
        self.camera_distance = max(2, min(20, self.camera_distance))
        self.update()

class MazeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.maze = None
        self.start = None
        self.end = None
        self.solution = None

    def set_maze(self, maze, start, end, solution):
        self.maze = maze
        self.start = start
        self.end = end
        self.solution = solution
        self.update()

    def update_maze(self, maze, start, end, solution):
        self.maze = maze
        self.start = start
        self.end = end
        self.solution = solution
        self.update()

    def paintEvent(self, event):
        if not self.maze:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        cell_size = min(width, height) // max(self.maze.graph['width'], self.maze.graph['height'])

        for (x1, y1), (x2, y2) in self.maze.edges():
            painter.setPen(Qt.white)
            painter.drawLine(x1 * cell_size, y1 * cell_size, x2 * cell_size, y2 * cell_size)

        if self.solution:
            painter.setPen(QColor(255, 215, 0))
            for i in range(len(self.solution) - 1):
                x1, y1 = self.solution[i]
                x2, y2 = self.solution[i + 1]
                painter.drawLine(x1 * cell_size + cell_size // 2, y1 * cell_size + cell_size // 2,
                                 x2 * cell_size + cell_size // 2, y2 * cell_size + cell_size // 2)

        painter.setBrush(Qt.green)
        painter.drawEllipse(self.start[0] * cell_size, self.start[1] * cell_size, cell_size, cell_size)

        painter.setBrush(Qt.red)
        painter.drawEllipse(self.end[0] * cell_size, self.end[1] * cell_size, cell_size, cell_size)

        painter.end()  

class ParameterWidget(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.start_time = time.time()

        layout = QVBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.time_label = QLabel()
        self.iterations_label = QLabel()
        self.usage_label = QLabel()

        layout.addWidget(self.title_label)
        layout.addWidget(self.time_label)
        layout.addWidget(self.iterations_label)
        layout.addWidget(self.usage_label)

        self.usage_plot = pg.PlotWidget()
        self.usage_plot.setBackground('w')
        self.usage_plot.setTitle(f"{title} Usage")
        self.usage_plot.setLabel('left', 'Usage', '%')
        self.usage_plot.setLabel('bottom', 'Time', 's')
        self.usage_curve = self.usage_plot.plot(pen='b')
        self.usage_data = []
        layout.addWidget(self.usage_plot)

        self.setLayout(layout)

        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        font = QFont("Arial", 10)
        self.setFont(font)

    def update_parameters(self, iterations, usage_percent):
        current_time = time.time() - self.start_time
        self.time_label.setText(f"Time: {current_time:.2f}s")
        self.iterations_label.setText(f"Iterations: {iterations}")
        self.usage_label.setText(f"Usage: {usage_percent:.2f}%")

        self.usage_data.append({'x': current_time, 'y': usage_percent})
        x = [item['x'] for item in self.usage_data[-100:]]
        y = [item['y'] for item in self.usage_data[-100:]]
        self.usage_curve.setData(x, y)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-CUBE Quantum Processor Simulation")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        main_layout = QHBoxLayout()

        self.cpu_panel = ParameterWidget("CPU")
        
        central_splitter = QSplitter(Qt.Horizontal)
        self.quantum_simulator = QuantumSimulatorWidget()
        self.maze_widget = MazeWidget()
        central_splitter.addWidget(self.quantum_simulator)
        central_splitter.addWidget(self.maze_widget)
        
        self.gpu_panel = ParameterWidget("GPU")

        main_layout.addWidget(self.cpu_panel, 1)
        main_layout.addWidget(central_splitter, 6)
        main_layout.addWidget(self.gpu_panel, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #ffffff;
            }
        """)

    def update_simulation(self, qubits, iterations, maze, start, end, solution, cpu_percent, gpu_percent):
        """
        Updates the visualization with the current state of the simulation.

        Args:
            qubits (list): The list of qubits.
            iterations (int): The current iteration number.
            maze (networkx.Graph): The current maze.
            start (tuple): The starting position.
            end (tuple): The ending position.
            solution (list): The solution path.
            cpu_percent (float): The CPU usage percentage.
            gpu_percent (float): The GPU usage percentage. 
        """
        self.update_qubits(qubits, iterations, maze, start, end, solution, cpu_percent, gpu_percent)
        self.app.processEvents()  # Process pending Qt events to ensure updates
        
        # Update qubits
        for sim_qubit, qubit in zip(self.quantum_simulator.qubits, qubits):
            sim_qubit.update(qubit.reflectance, qubit.received_intensity)
            if random.random() < 0.1:
                sim_qubit.flash()
        
        # Add random rays
        if random.random() < 0.3:
            start = random.choice(self.quantum_simulator.qubits)
            end = random.choice(self.quantum_simulator.qubits)
            self.quantum_simulator.add_ray((start.x, start.y, start.z), (end.x, end.y, end.z), 1.0)
        
        self.quantum_simulator.update_rays()
        self.quantum_simulator.update()
        
        # Update maze
        if maze is not None:
            self.maze_widget.update_maze(maze, start, end, solution)
        
        # Update CPU and GPU panels
        self.cpu_panel.update_parameters(iterations, cpu_percent)
        self.gpu_panel.update_parameters(iterations, gpu_percent)

class QCubeVisualizer:
    """
    Manages the visualization of the Q-CUBE and the maze solving process.
    Uses PyQt5 to create the graphical user interface and handles updates 
    from the simulation loop. 
    """
    def __init__(self, qubits):
        self.app = QApplication(sys.argv)  # Create a Qt application
        self.window = MainWindow()  # Create the main window
        self.window.quantum_simulator.set_qubits(qubits)  # Set the qubits in the simulator widget
        self.window.show()  # Show the main window

    def update_maze(self, maze, start, end, solution):
        """
        Updates the maze widget in the visualization window with the new maze, 
        start and end points, and the solution path.
        """
        self.window.maze_widget.set_maze(maze, start, end, solution) 
        self.app.processEvents()  # Process pending Qt events

    def update_qubits(self, qubits, iterations, maze, start, end, solution, cpu_percent, gpu_percent):
        """
        Updates the visualization of the qubits and other parameters in the Q-CUBE simulator.
        """
        # Update the visual representation of the qubits
        for sim_qubit, qubit in zip(self.window.quantum_simulator.qubits, qubits):
            sim_qubit.update(qubit.reflectance, qubit.received_intensity)
            if random.random() < 0.1:
                sim_qubit.flash()  # Make some qubits flash randomly

        # Add random rays to visualize interaction
        if random.random() < 0.3:
            start_qubit = random.choice(self.window.quantum_simulator.qubits)
            end_qubit = random.choice(self.window.quantum_simulator.qubits)
            self.window.quantum_simulator.add_ray(
                (start_qubit.x, start_qubit.y, start_qubit.z), 
                (end_qubit.x, end_qubit.y, end_qubit.z), 
                1.0
            )
        
        self.window.quantum_simulator.update_rays()  # Update the rays
        self.window.quantum_simulator.update()  # Update the simulator widget

        # Update CPU and GPU panels in the visualization window
        self.window.cpu_panel.update_parameters(iterations, cpu_percent) 
        self.window.gpu_panel.update_parameters(iterations, gpu_percent)

    def run(self):
        """
        Starts the Qt event loop, which handles user interactions and updates 
        the graphical interface. 
        """
        sys.exit(self.app.exec_())

# Example for testing the visualizer (if this is the main script)
if __name__ == "__main__":
    class DummyQubit:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.reflectance = 0.5
            self.received_intensity = 0.0

    dummy_qubits = [DummyQubit(x, y, z) for x in range(2) for y in range(2) for z in range(2)]
    visualizer = QCubeVisualizer(dummy_qubits)
    visualizer.run()