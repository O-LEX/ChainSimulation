import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Node:
    def __init__(self, position, weight):
        self.position = np.array(position)
        self.weight = weight

class Constraint:
    def __init__(self, node0, node1):
        self.node0 = node0
        self.node1 = node1
        self.li = np.linalg.norm(node0.position - node1.position)
        self.c = 0.0
        
        dim = node0.position.shape[0]
        self.dc = np.zeros(2 * dim)
        self.ddc = np.zeros((2 * dim, 2 * dim))
    
    def update(self):
        self.c = np.linalg.norm(self.node0.position - self.node1.position) - self.li
        
        direction = self.node0.position - self.node1.position
        norm = np.linalg.norm(direction)
        normalized_direction = direction / norm if norm > 0 else direction
        
        dim = self.node0.position.shape[0]
        self.dc[:dim] = normalized_direction
        self.dc[dim:] = -normalized_direction
        
        outer_prod = np.outer(normalized_direction, normalized_direction)
        identity = np.eye(dim)
        
        self.ddc[:dim, :dim] = outer_prod - identity
        self.ddc[dim:, dim:] = outer_prod - identity
        self.ddc[:dim, dim:] = -outer_prod
        self.ddc[dim:, :dim] = -outer_prod

class Chain:
    def __init__(self):
        self.nodes = []
        self.ndof = 0
        self.constraints = []
        self.gravity = np.array([0, -9.81])
        self.dE = None
        self.ddE = None
        
        self.x = None
        self.v = None
        self.invMass = None

    def setNodes(self, nodes):
        self.nodes = nodes
    
    def setConstraints(self, constraints):
        self.constraints = constraints
    
    def initialize(self):
        self.ndof = len(self.nodes) * self.nodes[0].position.shape[0]
        self.x = np.zeros(self.ndof)
        self.v = np.zeros(self.ndof)
        self.invMass = np.zeros(self.ndof)
        self.dE = np.zeros(self.ndof)
        self.ddE = np.zeros((self.ndof, self.ndof))
    
        dim = self.nodes[0].position.shape[0]
        for i, node in enumerate(self.nodes):
            self.x[i*dim:(i+1)*dim] = node.position
            self.invMass[i*dim:(i+1)*dim] = 1.0 / node.weight
    
    def update(self):
        self.dE = np.zeros(self.ndof)
        self.ddE = np.zeros((self.ndof, self.ndof))

        dEgrav = self.calcdEgrav()
        
        dEcons = self.calcdEcons()
        ddEcons = self.calcddEcons()

        self.dE = dEgrav
        self.ddE = np.zeros((self.ndof, self.ndof))
    
    def calcdEgrav(self):
        dim = self.nodes[0].position.shape[0]
        size = len(self.nodes) * dim
        grad = np.zeros(size)
        
        for i, node in enumerate(self.nodes):
            grad[i*dim:(i+1)*dim] = - node.weight * self.gravity
            
        return grad
    
    def updateConstraints(self):
        for constraint in self.constraints:
            constraint.update()
    
    def calcdEcons(self):
        self.updateConstraints()
        
        dim = self.nodes[0].position.shape[0]
        size = len(self.nodes) * dim
        gradient = np.zeros(size)
        
        for constraint in self.constraints:
            i0 = self.nodes.index(constraint.node0)
            i1 = self.nodes.index(constraint.node1)
            
            gradient[i0*dim:(i0+1)*dim] += constraint.c * constraint.dc[:dim]
            gradient[i1*dim:(i1+1)*dim] += constraint.c * constraint.dc[dim:2*dim]
            
        return gradient
    
    def calcddEcons(self):
        dim = self.nodes[0].position.shape[0]
        size = len(self.nodes) * dim
        hessian = np.zeros((size, size))
        
        for constraint in self.constraints:
            i0 = self.nodes.index(constraint.node0)
            i1 = self.nodes.index(constraint.node1)
            
            grad_outer = np.outer(constraint.dc[:2*dim], constraint.dc[:2*dim])
            
            hessian[i0*dim:(i0+1)*dim, i0*dim:(i0+1)*dim] += constraint.c * constraint.ddc[:dim,:dim] + grad_outer[:dim,:dim]
            
            hessian[i1*dim:(i1+1)*dim, i1*dim:(i1+1)*dim] += constraint.c * constraint.ddc[dim:2*dim,dim:2*dim] + grad_outer[dim:2*dim,dim:2*dim]
            
            hessian[i0*dim:(i0+1)*dim, i1*dim:(i1+1)*dim] += constraint.c * constraint.ddc[:dim,dim:2*dim] + grad_outer[:dim,dim:2*dim]
            hessian[i1*dim:(i1+1)*dim, i0*dim:(i0+1)*dim] += constraint.c * constraint.ddc[dim:2*dim,:dim] + grad_outer[dim:2*dim,:dim]
            
        return hessian
    
    def solve(self, dt=0.01, iterations=10, newton_tol=1e-6):
        x_next = self.x.copy()
        iter = 0 

        while iter < iterations:
            self.update()     
            forces = -self.dE
            residual = x_next - self.x - dt * self.v - dt * dt * self.invMass * forces

            jacobian = np.eye(self.ndof) + dt * dt * self.invMass * self.ddE

            dx = solve(jacobian, -residual)
            x_next += dx
            iter += 1
            
        self.v = (x_next - self.x) / dt
        self.x = x_next


def main():
    # Create nodes
    nodes = [
        Node([0, 0], 1.0),  # Fixed point (will be made static later)
        Node([1, 0], 1.0),  # First movable point
        Node([2, 0], 1.0),  # Second movable point
        Node([3, 0], 1.0),  # Third movable point
    ]
    
    # Create constraints between adjacent nodes
    constraints = [
        Constraint(nodes[0], nodes[1]),
        Constraint(nodes[1], nodes[2]),
    ]
    
    # Create the chain
    chain = Chain()
    chain.setNodes(nodes)
    chain.setConstraints(constraints)
    chain.initialize()
    
    # Simulation parameters
    dt = 0.01
    sim_time = 2.0  # Simulate for 2 seconds
    steps = int(sim_time / dt)
    
    # 位置の履歴を保存するリスト
    position_history = [chain.x.copy()]
    
    # Simulation loop
    for step in range(steps):
        chain.solve(dt)
        position_history.append(chain.x.copy())

        if step % 10 == 0:
            print(f"Time: {step * dt:.2f}s")
            print(chain.x)
    
    print("Simulation complete")
    
    # シミュレーション結果の可視化
    visualize_simulation(position_history, nodes[0].position.shape[0], len(nodes), dt)

def visualize_simulation(position_history, dim, num_nodes, dt):
    """シミュレーション結果を可視化する"""
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
    
    # アニメーション速度調整（10フレームごとに1フレーム表示）
    frames = range(0, len(position_history), 10)
    
    anim = FuncAnimation(fig, animate, frames=frames,
                         init_func=init, blit=True, interval=50)
    
    plt.show()

if __name__ == "__main__":
    main()

