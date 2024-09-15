import matplotlib.pyplot as plt

def plot_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history, label="Training Loss")
    plt.title("Loss over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_actual_vs_predicted(y_actual, y_predicted, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_predicted, label="Data")
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', label="Perfect Fit")
    plt.title(title)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()
