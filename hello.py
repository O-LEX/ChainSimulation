import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Node:
    def __init__(self, position, weight, is_fixed=False):
        self.position = np.array(position)
        self.weight = weight
        self.is_fixed = is_fixed

class Constraint:
    def __init__(self, node0, node1):
        self.node0 = node0
        self.node1 = node1
        self.dS = np.linalg.norm(node1.position - node0.position)
        self.C = 0.0
        
        dim = node0.position.shape[0]
        self.dC = np.zeros(2 * dim)
        self.ddC = np.zeros((2 * dim, 2 * dim))
    
    def update(self):
        direction = self.node1.position - self.node0.position
        norm = np.linalg.norm(direction)
        normalized_direction = direction / norm if norm > 0 else direction

        self.C = norm / self.dS - 1.0

        dim = self.node0.position.shape[0]
        self.dC[:dim] = -normalized_direction / self.dS
        self.dC[dim:] = normalized_direction / self.dS

        outer_prod = np.outer(normalized_direction, normalized_direction)
        identity = np.eye(dim)

        self.ddC[:dim, :dim] = (outer_prod - identity) / self.dS
        self.ddC[dim:, dim:] = (outer_prod - identity) / self.dS
        self.ddC[:dim, dim:] = outer_prod / self.dS
        self.ddC[dim:, :dim] = outer_prod / self.dS

class Chain:
    def __init__(self):
        self.dim = 0
        self.nodes = []
        self.free_nodes = []
        self.free_to_global_map = {}
        self.global_to_free_map = {}
        self.ndof_free = 0
        self.ndof_global = 0
        self.constraints = []
        self.constraint_stiffness = 0.0
        self.gravity = []

    def setNodes(self, nodes):
        self.nodes = nodes
    
    def setConstraints(self, constraints):
        self.constraints = constraints
    
    def setConstraintStiffness(self, stiffness):
        self.constraint_stiffness = stiffness
    
    def setGravity(self, gravity):
        self.gravity = np.array(gravity)
    
    def initialize(self):
        self.dim = self.nodes[0].position.shape[0]
        self.ndof_global = len(self.nodes) * self.dim
        
        free_idx = 0
        for global_idx, node in enumerate(self.nodes):
            if not node.is_fixed:
                self.global_to_free_map[global_idx] = free_idx
                self.free_to_global_map[free_idx] = global_idx
                free_idx += 1
        
        self.ndof_free = free_idx * self.dim
        self.free_nodes = [node for node in self.nodes if not node.is_fixed]
    
    def updateNode(self, x):
        for free_idx, global_idx in self.free_to_global_map.items():
            node = self.nodes[global_idx]
            node.position = x[free_idx * self.dim:(free_idx + 1) * self.dim]

    def calcEnergyDerivatives(self):
        self.updateConstraints()
        
        global_dEgrav = self.calcdEgrav()
        free_dEgrav = self.globalToFreeVector(global_dEgrav)
        
        global_dC = self.calcdC()
        free_dC = self.globalToFreeVector(global_dC)
        
        global_ddC = self.calcddC()
        free_ddC = self.globalToFreeMatrix(global_ddC)
        
        dE = free_dEgrav + free_dC
        ddE = free_ddC
        
        return dE, ddE
    
    def calcdEgrav(self):
        grad = np.zeros(self.ndof_global)
        for i, node in enumerate(self.nodes):
            grad[i * self.dim:(i + 1) * self.dim] = -node.weight * self.gravity
        return grad
    
    def updateConstraints(self):
        for constraint in self.constraints:
            constraint.update()
    
    def calcdC(self):
        gradient = np.zeros(self.ndof_global)
        for constraint in self.constraints:
            i0 = self.nodes.index(constraint.node0)
            i1 = self.nodes.index(constraint.node1)
            
            gradient[i0 * self.dim:(i0 + 1) * self.dim] += self.constraint_stiffness * constraint.C * constraint.dC[:self.dim]
            gradient[i1 * self.dim:(i1 + 1) * self.dim] += self.constraint_stiffness * constraint.C * constraint.dC[self.dim:]
        return gradient
    
    def calcddC(self):
        hessian = np.zeros((self.ndof_global, self.ndof_global))
        for constraint in self.constraints:
            i0 = self.nodes.index(constraint.node0)
            i1 = self.nodes.index(constraint.node1)
            
            grad_outer = np.outer(constraint.dC, constraint.dC)
            
            hessian[i0 * self.dim:(i0 + 1) * self.dim, i0 * self.dim:(i0 + 1) * self.dim] += \
                self.constraint_stiffness * (constraint.C * constraint.ddC[:self.dim, :self.dim] + grad_outer[:self.dim, :self.dim])
            
            hessian[i1 * self.dim:(i1 + 1) * self.dim, i1 * self.dim:(i1 + 1) * self.dim] += \
                self.constraint_stiffness * (constraint.C * constraint.ddC[self.dim:, self.dim:] + grad_outer[self.dim:, self.dim:])
            
            hessian[i0 * self.dim:(i0 + 1) * self.dim, i1 * self.dim:(i1 + 1) * self.dim] += \
                self.constraint_stiffness * (constraint.C * constraint.ddC[:self.dim, self.dim:] + grad_outer[:self.dim, self.dim:])
            
            hessian[i1 * self.dim:(i1 + 1) * self.dim, i0 * self.dim:(i0 + 1) * self.dim] += \
                self.constraint_stiffness * (constraint.C * constraint.ddC[self.dim:, :self.dim] + grad_outer[self.dim:, :self.dim])
        return hessian
    
    def globalToFreeVector(self, global_vector):
        free_vector = np.zeros(self.ndof_free)
        for free_idx, global_idx in self.free_to_global_map.items():
            free_vector[free_idx * self.dim:(free_idx + 1) * self.dim] = \
                global_vector[global_idx * self.dim:(global_idx + 1) * self.dim]
        return free_vector
    
    def globalToFreeMatrix(self, global_matrix):
        free_matrix = np.zeros((self.ndof_free, self.ndof_free))
        for free_i, global_i in self.free_to_global_map.items():
            for free_j, global_j in self.free_to_global_map.items():
                free_matrix[free_i * self.dim:(free_i + 1) * self.dim, free_j * self.dim:(free_j + 1) * self.dim] = \
                    global_matrix[global_i * self.dim:(global_i + 1) * self.dim, global_j * self.dim:(global_j + 1) * self.dim]
        return free_matrix

class Solver:
    def __init__(self, dt=0.01, iterations=10, tolerance=1e-6):
        self.dt = dt
        self.iterations = iterations
        self.tolerance = tolerance
        
        self.x = None
        self.v = None
        self.invMass = None
    
    def initialize(self, chain):
        dim = chain.nodes[0].position.shape[0]
        
        self.x = np.zeros(chain.ndof_free)
        self.v = np.zeros(chain.ndof_free)
        self.invMass = np.zeros(chain.ndof_free)
        
        for i, node in enumerate(chain.nodes):
            if not node.is_fixed:
                free_i = chain.global_to_free_map[i]
                self.x[free_i*dim:(free_i+1)*dim] = node.position
                self.invMass[free_i*dim:(free_i+1)*dim] = 1.0 / node.weight
    
    def solve(self, chain):
        x_next = self.x.copy()
        iter_count = 0
        
        while iter_count < self.iterations:
            chain.updateNode(x_next)
            dE, ddE = chain.calcEnergyDerivatives()
            residual = x_next - self.x - self.dt * self.v + self.dt * self.dt * self.invMass * dE
            jacobian = np.eye(chain.ndof_free) + self.dt * self.dt * self.invMass * ddE
            
            # Solve for the update step
            dx = -solve(jacobian, residual)
            x_next += dx
            iter_count += 1
            
            # Check for convergence
            if np.linalg.norm(dx) < self.tolerance:
                break
        
        # Update velocity and position
        self.v = (x_next - self.x) / self.dt
        self.x = x_next
        
        return iter_count

def main():
    # Create nodes
    nodes = [Node([i, 0], 1.0) for i in range(20)]
    nodes[0].is_fixed = True
    nodes[0].weight = 0.5
    nodes[-1].weight = 0.5

    # Create constraints
    constraints = [Constraint(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
    
    # Initialize the chain
    chain = Chain()
    chain.setNodes(nodes)
    chain.setConstraints(constraints)
    chain.setConstraintStiffness(1e3)
    chain.setGravity([0, -9.8])
    chain.initialize()
    
    # Initialize the solver
    solver = Solver(dt=0.01, iterations=10, tolerance=1e-6)
    solver.initialize(chain)
    
    # Simulation parameters
    dt = solver.dt
    sim_time = 100.0
    steps = int(sim_time / dt)
    
    position_history = []
    dim = nodes[0].position.shape[0]
    
    # Record initial positions
    initial_positions = np.zeros(len(nodes) * dim)
    for i, node in enumerate(nodes):
        initial_positions[i*dim:(i+1)*dim] = node.position
    position_history.append(initial_positions)
    
    # Run the simulation
    for step in range(steps):
        iterations = solver.solve(chain)
        
        # Record current positions
        current_positions = np.zeros(len(nodes) * dim)
        for i, node in enumerate(nodes):
            current_positions[i*dim:(i+1)*dim] = node.position
        
        position_history.append(current_positions)

        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Time: {step * dt:.2f}s, Iterations: {iterations}")
    
    print("Simulation complete")
    
    # Visualize the simulation
    visualize_simulation(position_history, dim, len(nodes), dt)

def visualize_simulation(position_history, dim, num_nodes, dt):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-3, 0.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Chain Simulation")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    
    line, = ax.plot([], [], 'o-', lw=2, markersize=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        positions = position_history[i]
        x_pos = [positions[j*dim] for j in range(num_nodes)]
        y_pos = [positions[j*dim+1] for j in range(num_nodes)]
        
        line.set_data(x_pos, y_pos)
        time_text.set_text(f'Time: {i*dt:.2f}s')
        return line, time_text

    # Mouse scroll event handler
    def on_scroll(event):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        scale_factor = 0.9 if event.button == 'up' else 1.1  # Zoom in: 0.9, Zoom out: 1.1

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_range = (x_max - x_min) * scale_factor
        y_range = (y_max - y_min) * scale_factor

        ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
        ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
        fig.canvas.draw()  # Update the plot

    # Keyboard event handler
    def on_key(event):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        step = 0.5  # Pan step size

        if event.key == 'up':
            ax.set_ylim(y_min + step, y_max + step)
        elif event.key == 'down':
            ax.set_ylim(y_min - step, y_max - step)
        elif event.key == 'left':
            ax.set_xlim(x_min - step, x_max - step)
        elif event.key == 'right':
            ax.set_xlim(x_min + step, x_max + step)
        fig.canvas.draw()  # Update the plot

    # Register events
    fig.canvas.mpl_connect('scroll_event', on_scroll)  # Mouse scroll
    fig.canvas.mpl_connect('key_press_event', on_key)  # Keyboard

    frames = range(0, len(position_history), 10)
    
    anim = FuncAnimation(fig, animate, frames=frames,
                         init_func=init, blit=True, interval=50)
    
    plt.show()

if __name__ == "__main__":
    main()