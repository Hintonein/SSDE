import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class PINNSolver(nn.Module):
    def __init__(self):
        super(PINNSolver, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        super().load_state_dict(torch.load('./numerical/vanderpol.pth', map_location='cpu'))

    def forward(self, x):
        t = torch.tanh(self.fc1(x))
        t = torch.tanh(self.fc2(t))
        t = self.fc3(t)
        return t * (1 - x) * x + x
    
class FDMSolver:
    def __init__(self, n_points=101, mu=1.0, left=0, right=1):
        """
        Initialize the finite difference solver
        
        Parameters
        ----------
        n_points: Number of grid points
        mu: The parameter μ in the equation
        """
        self.n_points = n_points
        self.mu = mu
        self.x = np.linspace(left, right, n_points)
        self.dx = (right - left) / (n_points - 1)
        self.u = np.linspace(left, right, n_points)  # 使用线性插值作为初始猜测
        
    def compute_residual(self, u):
        """
        Using central difference format to compute:
            d²u/dx² - μ(1-u²)du/dx + u
        """
        d2u_dx2 = np.zeros_like(u)
        du_dx = np.zeros_like(u)
        d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (self.dx**2)
        du_dx[1:-1] = (u[2:] - u[:-2]) / (2*self.dx)
        residual = d2u_dx2 - self.mu * (1 - u**2) * du_dx + u
        residual[0] = u[0] - 0  # u(0) = 0
        residual[-1] = u[-1] - 1    # u(1) = 1 
        return residual
    
    def compute_jacobian(self, u):
        n = len(u)
        J = np.zeros((n, n))
        for i in range(1, n-1):
            J[i, i-1] = 1.0 / (self.dx**2)
            J[i, i] = -2.0 / (self.dx**2)
            J[i, i+1] = 1.0 / (self.dx**2)
            J[i, i] += 1.0  
            J[i, i] += self.mu * 2 * u[i] * (u[i+1] - u[i-1]) / (2*self.dx)
            J[i, i-1] += -self.mu * (1 - u[i]**2) / (2*self.dx)
            J[i, i+1] += self.mu * (1 - u[i]**2) / (2*self.dx)
        J[0, 0] = 1.0
        J[-1, -1] = 1.0
        return J
    
    def solve(self, max_iter=50, tol=1e-6):
        u = self.u.copy()
        for iter in tqdm(range(max_iter), desc="Newton iterations"):
            r = self.compute_residual(u)
            if np.max(np.abs(r)) < tol:
                print(f"Converged after {iter+1} iterations")
                break
            J = self.compute_jacobian(u)
            du = np.linalg.solve(J, -r)
            u += du
            
        self.u = u
        return u, np.mean(r**2)