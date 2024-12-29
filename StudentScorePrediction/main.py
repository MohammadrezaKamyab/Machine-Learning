import numpy as np
import matplotlib.pyplot as plt
from src.gradient_descent import gradient_descent
from src.gradient_descent import compute_cost
from src.gradient_descent import compute_gradient
from src.visualize import plot_regression_line, plot_cost_history

# 1. Generate Synthetic Dataset
# Generating dataset for students hour and their grades
np.random.seed(50)
hours = np.sort(np.random.randint(1,20,20))
grades = np.sort(np.random.randint(0,100,hours.shape[0]))

w = 0  
b = 0 
alpha = 0.1  
num_iters = 1000  

# 3. Apply Gradient Descent
w, b, J_history, p_history = gradient_descent(hours, grades, w, b, alpha, num_iters,compute_cost, compute_gradient)

# 4. Visualize Results
plot_regression_line(hours, grades, w, b)
plot_cost_history(J_history)

# 5. Print Final Results
print(f"Final Weight (w): {w:.4f}")
print(f"Final Bias (b): {b:.4f}")
print(f"Final Cost: {J_history[-1]:.4f}")

