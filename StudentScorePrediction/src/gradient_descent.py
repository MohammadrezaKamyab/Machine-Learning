import numpy as np
import math

def compute_model_output (x, w, b):
    """Computes Predicted Values of Linear regression Model

    Args:
        x (ndarray(m,)): Data, m examples
        w (Scalar): Parameter for the model
        b (Scalar): Parameter for the model
        
    Returns:
        f_wb (ndarray(m,)) : model predictions
    """
    return w * x - b

def compute_cost (x, y , w, b):
    """Computes cost of Linear regression Model

    Args:
        x (ndarray(m,)): Data, m examples
        y (ndarray(m,)): Target values
        w (Scalar): Parameter for the model
        b (Scalar): Parameter for the model
        
    Returns:
        total_cost (float): the cost of using w, b as the parameters for linear regression
    """
    return np.sum(((w * x + b) - y) ** 2) * (1 / (2 * x.shape[0]))

def compute_gradient (x, y , w , b):
    """computes gradient for linear regression

    Args:
        x (ndarray(m,)): Data, m examples
        y (ndarray(m,)): Target values
        w (Scalar): Parameter for the model
        b (Scalar): Parameter for the model
        
    Returns
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """
    m = x.shape[0]
    dj_dw = np.sum(((w * x + b) - y) * x) / m
    dj_db = np.sum((w * x + b) - y) / m
    return dj_dw, dj_db

def gradient_descent (x, y, w_in, b_in, alpha, num_iter, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
        x (ndarray (m,))  : Data, m examples 
        y (ndarray (m,))  : target values
        w_in,b_in (scalar): initial values of model parameters  
        alpha (float):     Learning rate
        num_iters (int):   number of iterations to run gradient descent
        cost_function:     function to call to produce cost
        gradient_function: function to call to produce gradient
      
    Returns:
        w (scalar): Updated value of parameter after running gradient descent
        b (scalar): Updated value of parameter after running gradient descent
        J_history (List): History of cost values
        p_history (list): History of parameters [w,b] 
    """
      
    J_history = list()
    p_history = list()
    b = b_in
    w = w_in
    
    for i in range(num_iter):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        tmp_w = w - alpha * dj_dw
        tmp_b = b - alpha * dj_db
        w = tmp_w
        b = tmp_b

        if i < 100000:
            J_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])
        
        if i % math.ceil(num_iter / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
        
    return w, b, J_history, p_history