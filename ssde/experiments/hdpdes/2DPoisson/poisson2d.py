import torch
import numpy as np
import sympy as sp
import pandas as pd
from scipy.optimize import least_squares
from ssde import PDESymbolicSolver
from ssde.execute import cython_recursion_execute as ce
from ssde.execute import python_execute as pe
from ssde.const import make_const_optimizer
from ssde.program import Program
from ssde.utils import rect, cube
from ssde.pde import function_map
from ssde.task.recursion.recursion import make_regression_metric
# import pickle

def calculate_source(solution):
    ''' Calculate the source term of the PDE '''
    # solution: sympy expr of the solution
    real_params = dict()
    for symbol in sp.preorder_traversal(solution):
        if isinstance(symbol, sp.Symbol):
            exec('%s = sp.Symbol("%s")' % (symbol.name, symbol.name))
            if symbol.name not in real_params:
                real_params[symbol.name] = None
    real_params = sorted(list(real_params.keys()))
    print(f'real_params:{real_params}')
    source = 0
    for i in real_params:
        source += sp.diff(solution, i, 2)
    print(f'source:{source}')
    solution_func = sp.lambdify(real_params, solution, modules='numpy')
    source_func = sp.lambdify(real_params, source, modules='numpy')
    return solution_func, source_func, real_params

def opti_consts(nxexpr):
    """
    Returns objective of x2_traversal's consts optimization.

    Including bc constraints.
    """
    for token in x1_traversal:
        if token.name == 'Nxexpr':
            token.value = nxexpr.reshape(-1,1)
    y_bc_hat = ce(x1_traversal, X_bc)
    return (y_bc_hat-y_bc).ravel()

def pde_r(consts):
    """
    Returns objective of new_traversal's consts optimization.

    Including all physical constraints.
    """
    for i in range(len(consts)):
        new_traversal[consts_index[i]].torch_value = consts[i]
    y = pe(new_traversal, X_combine_torch)
    f = function_map['poisson2d'](y, X_combine_torch)
    y_hat = [f, y[-n_boundary:,0:1]]
    r = metric(y_input_torch,y_hat)
    obj = -r
    return obj



def test(pred_expr, real_expr,xmin,xmax,seed):
    np.random.seed(seed)
    pred_expr = sp.sympify(pred_expr)
    real_expr = sp.sympify(real_expr)
    real_params = dict()
    for symbol in sp.preorder_traversal(real_expr):
        if isinstance(symbol, sp.Symbol):
            exec('%s = sp.Symbol("%s")' % (symbol.name, symbol.name))
            if symbol.name not in real_params:
                real_params[symbol.name] = None
    real_params =  sorted(list(real_params.keys()))

    real_func = sp.lambdify(real_params, real_expr, modules='numpy')
    n_vars = len(real_params)
    pred_params = dict()
    for symbol in sp.preorder_traversal(pred_expr):
        if isinstance(symbol, sp.Symbol):
            exec('%s = sp.Symbol("%s")' % (symbol.name, symbol.name))
            if symbol.name not in pred_params:
                pred_params[symbol.name] = None
    pred_params = sorted(list(pred_params.keys()))
    if len(pred_params) != len(real_params):
        return False
    pred_func = sp.lambdify(pred_params, pred_expr, modules='numpy')


    X = np.random.uniform(xmin, xmax, (1000, n_vars))
    if n_vars == 2:
        n_boundary = 200
        X_bc = rect(xmin=[xmin, xmin],
                    xmax=[xmax, xmax],
                    n=n_boundary,
                    method='uniform')
        X_bc_temp = X_bc.transpose(1, 0)
    elif n_vars == 3:
        n_boundary = 200
        X_bc = cube(xmin,
                    xmax,
                    n_boundary,
                    method='uniform')
        X_bc_temp = X_bc.transpose(1, 0)
    else:
        n_boundary = 2
        X_bc = np.array([[xmin],[xmax]])
        X_bc_temp = X_bc.transpose(1, 0)
    X_combine = np.concatenate([X, X_bc], axis=0)
    X_com_temp = X_combine.transpose(1, 0)
    

    real_y = real_func(*X_com_temp)
    pred_y = pred_func(*X_com_temp)

    # 和数值解误差
    mse = np.mean((real_y - pred_y)**2)

    # pde_error
    ddy_x_pred = 0
    for i in range(len(pred_params)):
        dy_x = sp.diff(pred_expr, pred_params[i], 2)
        ddy_x_pred += dy_x
    ddy_x_pred_func = sp.lambdify(pred_params, ddy_x_pred, modules='numpy')

    ddy_x_real = 0
    for i in range(len(real_params)):
        dy_x = sp.diff(real_expr, real_params[i], 2)
        ddy_x_real += dy_x
    ddy_x_real_func = sp.lambdify(real_params, ddy_x_real, modules='numpy')
    
        
    pde_mse = np.mean((ddy_x_real_func(*X_com_temp) - ddy_x_pred_func(*X_com_temp))**2)
    bc_mse = np.mean((pred_y[-n_boundary:] - real_y[-n_boundary:])**2)
    pde_mrmse = (np.sqrt(pde_mse) + np.sqrt(bc_mse))/2

    return  sp.simplify(pred_expr), mse, pde_mrmse


def round_constants(expr, precision=2):
    if expr.is_Atom or expr.is_Piecewise:
        if expr.is_number:
            return round(float(expr), precision)
        else:
            return expr
    args = [round_constants(arg, precision) for arg in expr.args]
    return expr.func(*args)

real_expression_dict = {
    'nguyen-9': "sin(x1)+sin(x2**2)",
    'nguyen-10': "2*sin(x1)*cos(x2)",
    'nguyen-11': "x1**x2",
    'nguyen-12': "x1**4 -x1**3 + 0.5 * x2**2 -x2",
    'jin-1': "2.5*x1**4 - 1.3*x1**3 + 0.5*x2**2 - 1.7*x2"
    }

domain_dict = {
    'nguyen-9': [0.5, 1.5],
    'nguyen-10': [0.5, 1.5],
    'nguyen-11': [0.5, 1.5],
    'nguyen-12': [0.5, 1.5],
    'jin-1': [-1,1]
}

if __name__ == '__main__':
    # Generate some data on the bcs
    np.random.seed(10)
    test_case = 'jin-1'
    solution = sp.sympify(real_expression_dict[test_case])
    left_bc, right_bc = domain_dict[test_case]
    solution_func, source_func, real_params = calculate_source(solution)

    # The second-order derivative of x2 is calculated by finite difference with respect to the boundary conditions in the x2 direction
    # Then the first-order derivative of x1 in the boundary direction is derived.

    # region: sample for 1d opti(x2 direction)
    """ 
    Boundary conditions in the x2 direction, used for refine
    the first half of the rows are the obtained second-order differences of x1(on the border)
    and the second half are directly values
    """
    n_x1bc, n_x2_bc = 2, 51
    d_x2 = (right_bc - left_bc)/(n_x2_bc-1)
    X1_bc = np.array([left_bc,right_bc]).reshape(2,1)
    X2_bc_all = np.linspace(left_bc, right_bc, n_x2_bc).reshape(n_x2_bc,1)

    y_bc_all = []
    source_bc_all = []
    for i in X2_bc_all:
        temp = np.tile(i, (n_x1bc, 1))
        X_bc_temp = np.concatenate([X1_bc, temp], axis=1).transpose(1, 0)
        y_bc_all.append(solution_func(*X_bc_temp).reshape(-1, 1))
        source_bc_all.append(source_func(*X_bc_temp).reshape(-1, 1))
    X_bc1 = X1_bc
    y_bc_all = np.concatenate(y_bc_all, axis=1)
    source_bc_all = np.concatenate(source_bc_all, axis=1)
    # Convolution along the column direction calculates the second-order derivative of x2
    y_dx2_all = []
    for i in range(y_bc_all.shape[0]):
        y_dx2_all.append(np.convolve(y_bc_all[0,:], [1, -2, 1], 'valid').reshape(1,-1)/d_x2**2)
    y_dx2_all = np.concatenate(y_dx2_all, axis=0)
    # Calculate the first derivative of x1
    y_dx1_all = source_bc_all[:,1:-1] - y_dx2_all
    # Randomly select 10 points from y_bc_all, with an index range of (1,49) for a total of 49 points
    n_samples = 10
    index = np.random.choice(np.arange(1,50), n_samples, replace=False)
    y_bc1 = np.concatenate([y_dx1_all[:,index-1],y_bc_all[:,index]], axis=0)
    # endregion

    # region: sample for 1d opti(x1 direction)
    """
    Boundary conditions in the x1 direction, used for initial selection
    """
    n_x1bc, n_x2_bc = 10, 2
    X1_bc = np.random.uniform(left_bc, right_bc, (n_x1bc,1))
    X2_bc = np.array([left_bc,right_bc]).reshape(2,1)

    y_bc2 = []
    for i in X2_bc:
        temp = np.tile(i, (n_x1bc, 1))
        X_bc_temp = np.concatenate([X1_bc, temp], axis=1).transpose(1, 0)
        y_bc2.append(solution_func(*X_bc_temp).reshape(-1, 1))
    X_bc2 = X1_bc
    y_bc2 = np.concatenate(y_bc2, axis=1)

    # endregion

    X_input = [X_bc1, X_bc2]
    y_input = [y_bc1, y_bc2]

    model = PDESymbolicSolver(f"./configs/{test_case}/config_poisson_gp.json")
    x1_model = model.fit(X_input, y_input) # Should solve in ~10 hours
    x1_traversal = x1_model.program_.traversal
    x1_exp = x1_model.program_.sympy_expr
    print('Identified var x1\'s parametirc expression:')
    print(x1_exp)
    # with open('x1_traversal_poisson2d.pkl', 'wb') as f:
    #     pickle.dump(x1_traversal, f)

    # region:Compute the value of const (actually the label of x2)
    # with open('./models/jin-1/x1_traversal_poisson2d.pkl', 'rb') as f:
    #     x1_traversal = pickle.load(f)
    n_boundary = 20
    X_bc = rect(xmin=[left_bc, left_bc],
                        xmax=[right_bc, right_bc],
                        n=n_boundary,
                        method='uniform')
    y_bc = solution_func(*X_bc.T).reshape(-1, 1)


    consts = np.ones(n_boundary)
    res = least_squares(opti_consts, consts, method='lm')
    # consts_real = (0.5 * X_bc[:,1]**2 - 1.7 * X_bc[:,1])
    # print('abs error:', np.abs(consts_real-res.x).mean())
    X_bc = X_bc
    y_bc = np.array(res.x, dtype=np.float64).reshape(-1,1)
    # print("Shape of label vs. x2:", y_bc.shape)
    # endregion

    X_input = [None, X_bc]
    y_input = [None, y_bc]
    x2_model = PDESymbolicSolver(f"./configs/{test_case}/config_poisson_gp.json")
    # Fit the model
    config_x2 = x2_model.fit(X_input, y_input, start_n_var=2) # Should solve in ~10 hours
    x2_traversal = config_x2.program_.traversal
    x2_exp = x2_model.program_.sympy_expr
    print('Identified var x2\'s parametirc expression:')
    print(x2_exp)
    # with open('x2_traversal_possion2d.pkl', 'wb') as f:
    #     pickle.dump(x2_traversal, f)

    # with open('./models/jin-1/x2_traversal_poisson2d.pkl', 'rb') as f:
    #     x2_traversal = pickle.load(f)
    # replace the nxexpr with x2_traversal
    new_traversal = []
    for token in x1_traversal:
        if token.name == 'Nxexpr':
            new_traversal.extend(x2_traversal)
        else:
            new_traversal.append(token)
    print(new_traversal)
    ini_consts = []
    for token in new_traversal:
        if token.name == 'const':
            ini_consts.append(token.value)
    ini_consts = [torch.tensor(i, requires_grad=True) for i in ini_consts]

    # region: refine constants in the replaced traversal
    n_samples, n_boundary = 100, 100
    X = np.random.uniform(left_bc, right_bc, (n_samples, 2))
    X_bc = rect(xmin=[left_bc, left_bc],
            xmax=[right_bc, right_bc],
            n=n_samples,
            method='uniform')
    X_bc_temp = X_bc.transpose(1, 0)
    X_combine = np.concatenate([X, X_bc], axis=0)
    X_combine_torch = torch.tensor(X_combine, dtype=torch.float32, requires_grad=True)
    X_com_temp = X_combine.transpose(1, 0)
    y = source_func(*X_com_temp).reshape(-1, 1)
    y_bc = solution_func(*X_bc_temp).reshape(-1, 1)
    y_input = [y, y_bc]
    y_input_torch = [torch.tensor(i, dtype=torch.float32, requires_grad=True) for i in y_input]

    consts_index = [i for i in range(len(new_traversal)) if new_traversal[i].name == 'const']
    metric,_,_ = make_regression_metric("neg_smse_torch", y_input)

    optimized_consts, smse = make_const_optimizer('torch')(pde_r, ini_consts)
    for i in range(len(optimized_consts)):
        new_traversal[consts_index[i]].value = optimized_consts[i]
    test_p = Program()
    test_p.traversal = new_traversal
    pred_expr = test_p.sympy_expr
    sym_expr = sp.simplify(sp.N(sp.expand(sp.sympify(pred_expr)),4))
    print(f'Identified solution: {sym_expr}')

    # region: 测试
    mse_ls, pde_mrmse_ls= [], []
    solution = sp.sympify("2.5*x1**4 - 1.3*x1**3 + 0.5*x2**2 - 1.7*x2")
    for i in range(20):
        _, mse, mrmse = test(pred_expr, solution, left_bc, right_bc,i)
        mse_ls.append(mse)
        pde_mrmse_ls.append(mrmse)

    # 计算均值
    mse_mean = np.mean(mse_ls)
    pde_mrmse_mean = np.mean(pde_mrmse_ls)
    # 计算标准误差
    std_error_mse = np.std(mse_ls) / np.sqrt(20)
    std_error_mrmse = np.std(pde_mrmse_ls) / np.sqrt(20)
    confidence_level = 0.95
    critical_value = 1.96
    # 计算置信区间
    margin_of_mrmse = critical_value * std_error_mrmse
    margin_of_mse = critical_value * std_error_mse
    # 最后一行输出转为文字
    pde_mrmse_ls.append(f'{pde_mrmse_mean:.4e}±{margin_of_mrmse:.4e}')
    mse_ls.append(f'{mse_mean:.4e}±{margin_of_mse:.4e}')


    # 以 pd 格式输出
    df = pd.DataFrame({'MSE': mse_ls[-1:], 
                        'PDE_MRMSE': pde_mrmse_ls[-1:]
                        })

    # 设置显示宽度和列宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    print('The result of the test:')
    print(df)
    # endregion