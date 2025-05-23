from ssde.gradients import hessian, jacobian

def poisson1D_forward(y, X):
    return hessian(y, X, i=0, j=0)

def haros_forward(y, X):
    return hessian(y, X, i=0, j=0) + y

def van_der_pol_forward(y, X, mu = 1):
    return hessian(y, X, i=0, j=0) - mu * (1 - y**2) * jacobian(y, X, i=0, j=0) + y 

# 2D poisson's Equation
def poisson2D_forward(y, X):
    return hessian(y, X, i=0, j=0) + hessian(y, X, i=1, j=1)

# 3D poisson's Equation
def poisson3D_forward(y, X):
    return hessian(y, X, i=0, j=0) + hessian(y, X, i=1, j=1) +  hessian(y, X, i=2, j=2)

# 1D Advection Equation
def advection1D_forward(y,X):
    # first dimension is time, while others are spatial dimensions
    return jacobian(y, X, i=0, j=0) + jacobian(y, X, i=0, j=1)

# 2D Advection Equation
def advection2D_forward(y,X):
    # first dimension is time, while others are spatial dimensions
    return jacobian(y, X, i=0, j=0) + jacobian(y, X, i=0, j=1) +  jacobian(y, X, i=0, j=2)

# 3D Advection Equation
def advection3D_forward(y,X):
    # first dimension is time, while others are spatial dimensions
    return jacobian(y, X, i=0, j=0) + jacobian(y, X, i=0, j=1) +  jacobian(y, X, i=0, j=2) + jacobian(y, X, i=0, j=3)

# 1D Heat Equation
def heat1D_forward(y, X, a = 1):
    return jacobian(y, X, i=0, j=0) - a**2 * (hessian(y, X, i=1, j=1))

def heat2D_forward(y, X, a = 1):
    return jacobian(y, X, i=0, j=0) - a**2 * (hessian(y, X, i=1, j=1) + hessian(y, X, i=2, j=2))

def heat3D_forward(y, X, a = 1):
    return jacobian(y, X, i=0, j=0) - a**2 * (hessian(y, X, i=1, j=1) + hessian(y, X, i=2, j=2) + hessian(y, X, i=3, j=3))

def wave2D_forward(y, X):
    return hessian(y, X, i=0, j=0) -  (hessian(y, X, i=1, j=1) + hessian(y, X, i=2, j=2)) + y + y**3

def wave3D_forward(y, X):
    return hessian(y, X, i=0, j=0) -  (hessian(y, X, i=1, j=1) + hessian(y, X, i=2, j=2) + hessian(y, X, i=3, j=3)) - y ** 2


function_map = {
    "haros": haros_forward,
    "van_der_pol": van_der_pol_forward,
    "poisson1d": poisson1D_forward,
    "poisson2d": poisson2D_forward,
    "poisson3d": poisson3D_forward,
    "advection1d": advection1D_forward,
    "advection2d": advection2D_forward,
    "advection3d": advection3D_forward,
    "heat1d": heat1D_forward,
    "heat2d": heat2D_forward,
    "heat3d": heat3D_forward,
    "wave2d": wave2D_forward,
    "wave3d": wave3D_forward,
}
