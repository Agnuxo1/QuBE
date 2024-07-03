
# maze_visualizer.py

import pygame
import networkx as nx



class MazeVisualizer:
    def __init__(self, maze_size, cell_size=40):
        self.maze_size = maze_size
        self.cell_size = cell_size
        self.width = maze_size * cell_size
        self.height = maze_size * cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Q-CUBE Maze Solver")
        
        self.colors = {
            'background': (255, 255, 255),  # White
            'wall': (0, 0, 0),              # Black
            'path': (255, 0, 0),            # Red
            'start': (0, 255, 0),           # Green
            'end': (0, 0, 255),             # Blue
        }
        

    def draw_maze(self, maze, start, end, solution=None):
        self.screen.fill(self.colors['background'])
        
        # Draw walls
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                x = j * self.cell_size
                y = i * self.cell_size
                
                if not maze.has_edge((i, j), (i, j+1)):  # Right wall
                    pygame.draw.line(self.screen, self.colors['wall'], 
                                     (x + self.cell_size, y), 
                                     (x + self.cell_size, y + self.cell_size))
                
                if not maze.has_edge((i, j), (i+1, j)):  # Bottom wall
                    pygame.draw.line(self.screen, self.colors['wall'], 
                                     (x, y + self.cell_size), 
                                     (x + self.cell_size, y + self.cell_size))
        
        # Draw outer walls
        pygame.draw.rect(self.screen, self.colors['wall'], 
                         (0, 0, self.width, self.height), 2)
        
        # Draw solution path
        if solution:
            path = self.solution_to_path(maze, solution)
            for i in range(len(path) - 1):
                start_pos = path[i]
                end_pos = path[i + 1]
                pygame.draw.line(self.screen, self.colors['path'],
                                 (start_pos[1] * self.cell_size + self.cell_size // 2,
                                  start_pos[0] * self.cell_size + self.cell_size // 2),
                                 (end_pos[1] * self.cell_size + self.cell_size // 2,
                                  end_pos[0] * self.cell_size + self.cell_size // 2),
                                 5)
        
        # Draw start and end
        self.draw_cell(start, self.colors['start'])
        self.draw_cell(end, self.colors['end'])
        
        pygame.display.flip()

    def draw_cell(self, pos, color):
        pygame.draw.rect(self.screen, color,
                         (pos[1] * self.cell_size + 5, 
                          pos[0] * self.cell_size + 5,
                          self.cell_size - 10, 
                          self.cell_size - 10))

    def solution_to_path(self, maze, solution):
        path = []
        nodes = list(maze.nodes())
        for i, bit in enumerate(solution):
            if bit == '1':
                path.append(nodes[i])
        return path

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        pygame.quit()
