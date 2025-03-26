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
    def __init__(self, node0, node1, stiffness=1.0):
        self.node0 = node0
        self.node1 = node1
        self.li = np.linalg.norm(node0.position - node1.position)
        self.C = 0.0
        self.stiffness = stiffness
        
        dim = node0.position.shape[0]
        self.dC = np.zeros(2 * dim)
        self.ddC = np.zeros((2 * dim, 2 * dim))
    
    def update(self):
        current_length = np.linalg.norm(self.node0.position - self.node1.position)
        self.C = current_length / self.li - 1.0
        
        direction = self.node0.position - self.node1.position
        norm = np.linalg.norm(direction)
        normalized_direction = direction / norm if norm > 0 else direction
        
        dim = self.node0.position.shape[0]
        self.dC[:dim] = normalized_direction / self.li 
        self.dC[dim:] = -normalized_direction / self.li  
        
        outer_prod = np.outer(normalized_direction, normalized_direction)
        identity = np.eye(dim)
        
        self.ddC[:dim, :dim] = (outer_prod - identity) / self.li
        self.ddC[dim:, dim:] = (outer_prod - identity) / self.li
        self.ddC[:dim, dim:] = -outer_prod / self.li
        self.ddC[dim:, :dim] = -outer_prod / self.li

class Chain:
    def __init__(self):
        self.nodes = []
        self.free_nodes = []
        self.global_to_free_map = {}
        self.ndof_free = 0
        self.ndof_global = 0
        self.constraints = []
        self.gravity = np.array([0, -9.81])

    def setNodes(self, nodes):
        self.nodes = nodes
    
    def setConstraints(self, constraints):
        self.constraints = constraints
    
    def initialize(self):
        dim = self.nodes[0].position.shape[0]
        self.ndof_global = len(self.nodes) * dim
        
        self.free_nodes = [node for node in self.nodes if not node.is_fixed]
        self.ndof_free = len(self.free_nodes) * dim
        
        free_idx = 0
        for i, node in enumerate(self.nodes):
            if not node.is_fixed:
                self.global_to_free_map[i] = free_idx
                free_idx += 1
    
    def updateNode(self, x):
        dim = self.nodes[0].position.shape[0]
        
        for i, node in enumerate(self.nodes):
            if not node.is_fixed:
                free_i = self.global_to_free_map[i]
                node.position = x[free_i*dim:(free_i+1)*dim]
    
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
        dim = self.nodes[0].position.shape[0]
        grad = np.zeros(self.ndof_global)
        
        for i, node in enumerate(self.nodes):
            grad[i*dim:(i+1)*dim] = - node.weight * self.gravity
            
        return grad
    
    def updateConstraints(self):
        for constraint in self.constraints:
            constraint.update()
    
    def calcdC(self):
        dim = self.nodes[0].position.shape[0]
        gradient = np.zeros(self.ndof_global)
        
        for constraint in self.constraints:
            i0 = self.nodes.index(constraint.node0)
            i1 = self.nodes.index(constraint.node1)
            
            gradient[i0*dim:(i0+1)*dim] += constraint.stiffness * constraint.C * constraint.dC[:dim]
            gradient[i1*dim:(i1+1)*dim] += constraint.stiffness * constraint.C * constraint.dC[dim:2*dim]
            
        return gradient
    
    def calcddC(self):
        dim = self.nodes[0].position.shape[0]
        hessian = np.zeros((self.ndof_global, self.ndof_global))
        
        for constraint in self.constraints:
            i0 = self.nodes.index(constraint.node0)
            i1 = self.nodes.index(constraint.node1)
            
            grad_outer = np.outer(constraint.dC[:2*dim], constraint.dC[:2*dim])
            
            hessian[i0*dim:(i0+1)*dim, i0*dim:(i0+1)*dim] += constraint.stiffness * (
                constraint.C * constraint.ddC[:dim,:dim] + grad_outer[:dim,:dim])
            
            hessian[i1*dim:(i1+1)*dim, i1*dim:(i1+1)*dim] += constraint.stiffness * (
                constraint.C * constraint.ddC[dim:2*dim,dim:2*dim] + grad_outer[dim:2*dim,dim:2*dim])
            
            hessian[i0*dim:(i0+1)*dim, i1*dim:(i1+1)*dim] += constraint.stiffness * (
                constraint.C * constraint.ddC[:dim,dim:2*dim] + grad_outer[:dim,dim:2*dim])
            hessian[i1*dim:(i1+1)*dim, i0*dim:(i0+1)*dim] += constraint.stiffness * (
                constraint.C * constraint.ddC[dim:2*dim,:dim] + grad_outer[dim:2*dim,:dim])
            
        return hessian
    
    def globalToFreeVector(self, global_vector):
        dim = self.nodes[0].position.shape[0]
        free_vector = np.zeros(self.ndof_free)
        
        for i, node in enumerate(self.nodes):
            if not node.is_fixed:
                free_i = self.global_to_free_map[i]
                free_vector[free_i*dim:(free_i+1)*dim] = global_vector[i*dim:(i+1)*dim]
        
        return free_vector
    
    def freeToGlobalVector(self, free_vector):
        dim = self.nodes[0].position.shape[0]
        global_vector = np.zeros(self.ndof_global)
        
        for i, node in enumerate(self.nodes):
            if not node.is_fixed:
                free_i = self.global_to_free_map[i]
                global_vector[i*dim:(i+1)*dim] = free_vector[free_i*dim:(free_i+1)*dim]
            else:
                global_vector[i*dim:(i+1)*dim] = node.position
        
        return global_vector
    
    def globalToFreeMatrix(self, global_matrix):
        dim = self.nodes[0].position.shape[0]
        free_matrix = np.zeros((self.ndof_free, self.ndof_free))
        
        for i, node_i in enumerate(self.nodes):
            if not node_i.is_fixed:
                free_i = self.global_to_free_map[i]
                for j, node_j in enumerate(self.nodes):
                    if not node_j.is_fixed:
                        free_j = self.global_to_free_map[j]
                        free_matrix[free_i*dim:(free_i+1)*dim, free_j*dim:(free_j+1)*dim] = global_matrix[i*dim:(i+1)*dim, j*dim:(j+1)*dim]
        
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
            
            dx = -solve(jacobian, residual)
            x_next += dx
            iter_count += 1
            
            if np.linalg.norm(dx) < self.tolerance:
                break
        
        self.v = (x_next - self.x) / self.dt
        self.x = x_next
        
        return iter_count

def main():
    nodes = [
        Node([0, 0], 1.0, is_fixed=True),
        Node([1, 0], 1.0),
        Node([2, 0], 1.0),
        Node([3, 0], 1.0),
    ]
    
    constraints = [
        Constraint(nodes[0], nodes[1], stiffness=1000.0),
        Constraint(nodes[1], nodes[2], stiffness=1000.0),
        Constraint(nodes[2], nodes[3], stiffness=1000.0),
    ]
    
    chain = Chain()
    chain.setNodes(nodes)
    chain.setConstraints(constraints)
    chain.initialize()
    
    solver = Solver(dt=0.01, iterations=10, tolerance=1e-6)
    solver.initialize(chain)
    
    dt = solver.dt
    sim_time = 100.0
    steps = int(sim_time / dt)
    
    position_history = []
    dim = nodes[0].position.shape[0]
    
    initial_positions = np.zeros(len(nodes) * dim)
    for i, node in enumerate(nodes):
        initial_positions[i*dim:(i+1)*dim] = node.position
    position_history.append(initial_positions)
    
    for step in range(steps):
        iterations = solver.solve(chain)
        
        current_positions = np.zeros(len(nodes) * dim)
        for i, node in enumerate(nodes):
            current_positions[i*dim:(i+1)*dim] = node.position
        
        position_history.append(current_positions)

        if step % 10 == 0:
            print(f"Time: {step * dt:.2f}s, Iterations: {iterations}")
            print([node.position for node in nodes])
    
    print("Simulation complete")
    
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
    
    line, = ax.plot([], [], 'o-', lw=2)
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
    
    frames = range(0, len(position_history), 10)
    
    anim = FuncAnimation(fig, animate, frames=frames,
                         init_func=init, blit=True, interval=50)
    
    plt.show()

if __name__ == "__main__":
    main()

