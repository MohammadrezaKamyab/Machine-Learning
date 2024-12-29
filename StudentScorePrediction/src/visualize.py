import matplotlib.pyplot as plt

def plot_regression_line(x, y, w, b):
    
    predictions = w * x + b
    plt.scatter(x, y, color="red", label="Actual Data")
    plt.plot(x, predictions, color="blue", label="Regression Line")
    plt.title("Study Hours vs grade")
    plt.xlabel("Study Hours")
    plt.ylabel("grade")
    plt.legend()
    plt.show()

def plot_cost_history(J_history):
    
    plt.plot(range(len(J_history)), J_history, color="green")
    plt.title("Cost Function History")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
